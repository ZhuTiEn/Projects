import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from agentsa.agents import TargetAwareExposure, FakeUserAgent, generate_agent_attack, valid_ids
from agentsa.data import load_dataset, prepare
from agentsa.experiment import target_coverage, run_experiments
from agentsa.llm import MockClient
from agentsa.utils import load_config, read_json, validate_config
from test_core import test_config, create_data, ROOT


class AttackRevisionTests(unittest.TestCase):
    def test_target_aware_exposure_uses_metadata_neighbors_and_masks(self):
        counts = np.array([5, 5, 100, 0])
        vectors = np.array([[1., 0.], [.99, .01], [0., 1.], [1., 0.]])
        environment = TargetAwareExposure(counts, 1, vectors, 0, 1, 1.)
        self.assertEqual(environment.expose(set(), 0, 3, 42), [1])
        self.assertEqual(environment.expose({1}, 0, 3, 42), [])
        mixed = TargetAwareExposure(counts, 1, vectors, 0, 1, .5)
        self.assertEqual(set(mixed.expose(set(), 0, 3, 42)), {1, 2})
        self.assertEqual(mixed.expose(set(), 0, 3, 42), mixed.expose(set(), 0, 3, 42))

    def test_dense_selection_is_enforced_and_capped_by_remaining(self):
        with self.assertRaisesRegex(ValueError, 'Too few'):
            valid_ids({'item_ids': []}, 'item_ids', [1, 2], maximum=2, minimum=2)
        settings = load_config(ROOT/'configs/smoke.yaml')['agent']
        settings.update(min_selected_per_round=2, max_selected_per_round=4)
        class Encoder:
            def relevant(self, *args):
                return []
        agent = FakeUserAgent(1, {}, [], MockClient(), Encoder(), settings)
        items = [dict(item_id=i, title=str(i), category='', description='') for i in range(5)]
        rows = agent.act([1, 2, 3, 4], items, 0, 1)
        self.assertEqual(len(rows), 1)

    def test_invalid_config_and_formal_separation(self):
        config = load_config(ROOT/'configs/formal_target_aware.yaml')
        self.assertEqual(config['attack']['fake_ratio'], .05)
        self.assertEqual(config['training']['learning_rate'], .0003)
        self.assertEqual(config['agent']['filler_cap'], 20)
        original = load_config(ROOT/'configs/formal_original.yaml')
        self.assertNotEqual(config['agent']['output'], original['agent']['output'])
        config['agent']['min_selected_per_round'] = 5
        with self.assertRaisesRegex(ValueError, 'min_selected'):
            validate_config(config)

    def test_coverage_retains_failures_and_averages_seeds(self):
        rows=[]
        for target in (1, 2):
            for seed in (11, 22):
                row=dict(model='nmf', method='agent', mock_validation=False, budget_protocol='matched_agent',
                         agent_variant='target_aware_dense_v1', target=target, seed=seed)
                for k in (10, 50, 100):
                    row[f'clean_hr_at_{k}']=0.
                    row[f'poison_hr_at_{k}']=.02 if target==1 and seed==11 else 0.
                rows.append(row)
        report=target_coverage(rows, .01)[0]
        self.assertEqual(report['targets'], 2)
        self.assertEqual(report['fraction_targets_any_hit_at_10'], .5)
        self.assertEqual(report['fraction_targets_minimum_hr_at_10'], .5)
        self.assertEqual(report['per_target'][1]['poison_hr_at_10'], 0.)

    def test_dense_variant_end_to_end_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=test_config(Path(tmp))
            config['agent'].update(variant='target_aware_dense_v1', exposure_policy='target_aware',
                                  min_selected_per_round=2, max_selected_per_round=2,
                                  semantic_pool_size=5, semantic_fraction=.5)
            create_data(Path(tmp))
            prepare(config)
            dataset=load_dataset(config)
            with patch('requests.post', side_effect=AssertionError('No network allowed')):
                path=generate_agent_attack(config,dataset,dataset['targets'][0],mock=True)
                manifest=read_json(path.with_name('manifest.json'))
                self.assertEqual(manifest['filler_lengths'],[4]*4)
                result=run_experiments(config,dataset,['nmf'],['average'],[11],dataset['targets'],
                                       include_agent=True,match_agent=True,mock=True)
            self.assertEqual(len(result),2)
            for row in result:
                self.assertEqual(row['fake_ratings'],20)
                self.assertLessEqual(row['poison_hr_at_10'],row['poison_hr_at_50'])
                self.assertLessEqual(row['poison_hr_at_50'],row['poison_hr_at_100'])
            coverage=Path(tmp)/config['experiment']['output']/'mock_validation'/'target_coverage.json'
            self.assertTrue(coverage.exists())


if __name__ == '__main__':
    unittest.main()
