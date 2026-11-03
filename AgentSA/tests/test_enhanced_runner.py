import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentsa.run_enhanced import workflow
from agentsa.utils import read_json


class EnhancedRunnerTests(unittest.TestCase):
    def fixture(self, directory):
        return {'agent': {'variant': 'target_aware_dense_v1', 'output': str(directory/'agent')},
                'experiment': {'output': str(directory/'experiment')},
                'attack': {'fake_ratio': .05}, 'project_root': str(directory),
                '_config_path': str(directory/'config.yaml'),
                'llm': {'api_key_env': 'ENHANCED_RUNNER_TEST_KEY'}}

    def test_missing_key_noninteractive_sends_no_requests(self):
        with tempfile.TemporaryDirectory() as temp:
            config=self.fixture(Path(temp))
            with patch('agentsa.run_enhanced.load_config',return_value=config), \
                 patch('agentsa.run_enhanced.load_dataset',return_value={'targets':[1920],'manifest':{'users':19856}}), \
                 patch('agentsa.run_enhanced.completed_generation',return_value=False), \
                 patch.dict(os.environ,{},clear=True), patch('sys.stdin.isatty',return_value=False), \
                 patch('subprocess.run') as run:
                with self.assertRaisesRegex(RuntimeError,'missing'):
                    workflow('config.yaml')
                run.assert_not_called()

    def test_failed_generation_prevents_training_and_key_is_not_saved(self):
        with tempfile.TemporaryDirectory() as temp:
            config=self.fixture(Path(temp))
            with patch('agentsa.run_enhanced.load_config',return_value=config), \
                 patch('agentsa.run_enhanced.load_dataset',return_value={'targets':[1920],'manifest':{'users':19856}}), \
                 patch('agentsa.run_enhanced.completed_generation',return_value=False), \
                 patch.dict(os.environ,{},clear=True), patch('sys.stdin.isatty',return_value=True), \
                 patch('getpass.getpass',return_value='test-secret-not-to-save'), \
                 patch('subprocess.run') as run:
                with self.assertRaisesRegex(RuntimeError,'completion manifest'):
                    workflow('config.yaml')
                self.assertEqual(run.call_count,2)
                self.assertNotIn('ENHANCED_RUNNER_TEST_KEY',os.environ)
            status=Path(temp)/'experiment/workflow_status.json'
            self.assertNotIn('test-secret-not-to-save',status.read_text())
            self.assertEqual(read_json(status)['status'],'failed')

    def test_completed_generation_needs_no_key_and_only_runs_training(self):
        with tempfile.TemporaryDirectory() as temp:
            config=self.fixture(Path(temp))
            with patch('agentsa.run_enhanced.load_config',return_value=config), \
                 patch('agentsa.run_enhanced.load_dataset',return_value={'targets':[1920],'manifest':{'users':19856}}), \
                 patch('agentsa.run_enhanced.completed_generation',return_value=True), \
                 patch.dict(os.environ,{},clear=True), patch('subprocess.run') as run:
                result=workflow('config.yaml')
                self.assertEqual(result['status'],'complete')
                run.assert_called_once()
                self.assertIn('--match-agent-budget',run.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
