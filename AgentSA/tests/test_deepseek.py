"""DeepSeek wire-format checks; every HTTP request is mocked."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentsa.llm import ChatClient, connection_info
from agentsa.utils import load_config, read_json

ROOT = Path(__file__).resolve().parents[1]


class Response:
    status_code = 200

    def __init__(self, content='{"ok":true}', finish_reason="stop"):
        self.content = content
        self.finish_reason = finish_reason

    def json(self):
        return {"model": "deepseek-flash", "choices": [{"message": {"content": self.content},
                    "finish_reason": self.finish_reason}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                          "prompt_cache_hit_tokens": 4, "prompt_cache_miss_tokens": 6}}


class DeepSeekTests(unittest.TestCase):
    def setUp(self):
        self.settings = load_config(ROOT / "configs/deepseek.yaml")["llm"]
        self.env = patch.dict(os.environ, {"DEEPSEEK_API_KEY": "fixture-key-not-real",
                                          "DEEPSEEK_BASE_URL": "", "DEEPSEEK_MODEL": ""})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_defaults_json_format_and_non_thinking(self):
        with tempfile.TemporaryDirectory() as tmp, patch("agentsa.llm.requests.post", return_value=Response()) as post:
            client = ChatClient(self.settings, tmp)
            self.assertEqual(client.generate("test", 'Return JSON: {"ok":true}', {}), {"ok": True})
            self.assertEqual(post.call_args.args[0], "https://api.deepseek.com/chat/completions")
            body = post.call_args.kwargs["json"]
            self.assertEqual(body["model"], "deepseek-flash")
            self.assertEqual(body["thinking"], {"type": "disabled"})
            self.assertEqual(body["response_format"], {"type": "json_object"})
            self.assertIn("temperature", body)
            self.assertNotIn("reasoning_effort", body)
            saved = read_json(next((Path(tmp) / "cache").glob("*.json")))
            self.assertEqual(saved["usage"]["prompt_cache_hit_tokens"], 4)
            self.assertEqual(saved["finish_reason"], "stop")

    def test_environment_override_and_thinking_identity(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
                "DEEPSEEK_MODEL": "deepseek-v4-pro", "DEEPSEEK_BASE_URL": "https://api.deepseek.com/v1/"}), \
                patch("agentsa.llm.requests.post", return_value=Response()) as post:
            plain = ChatClient(self.settings, tmp)
            settings = {**self.settings, "thinking": "enabled", "reasoning_effort": "high"}
            thinking = ChatClient(settings, tmp)
            self.assertNotEqual(plain.identity, thinking.identity)
            thinking.generate("test", 'Return JSON: {"ok":true}', {})
            self.assertEqual(post.call_args.args[0], "https://api.deepseek.com/v1/chat/completions")
            body = post.call_args.kwargs["json"]
            self.assertEqual(body["model"], "deepseek-v4-pro")
            self.assertEqual(body["thinking"], {"type": "enabled"})
            self.assertEqual(body["reasoning_effort"], "high")
            self.assertNotIn("temperature", body)

    def test_blank_json_response_explicitly_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch("agentsa.llm.requests.post", return_value=Response("   ")):
            client = ChatClient(self.settings, tmp)
            with self.assertRaisesRegex(ValueError, "blank"):
                client.generate("test", "Return JSON", {})

    def test_truncated_response_rejected_even_if_valid_json_and_cached(self):
        with tempfile.TemporaryDirectory() as tmp, patch("agentsa.llm.requests.post", return_value=Response(finish_reason="length")) as post:
            client = ChatClient(self.settings, tmp)
            for _ in range(2):
                with self.assertRaisesRegex(ValueError, "truncated"):
                    client.generate("test", "Return JSON", {})
            self.assertEqual(post.call_count, 1)

    def test_missing_key_has_actionable_error_and_no_request(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}), \
                patch("agentsa.llm.requests.post") as post:
            with self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY"):
                ChatClient(self.settings, tmp)
            post.assert_not_called()
            self.assertFalse(connection_info(self.settings)["credential_present"])


if __name__ == "__main__":
    unittest.main()
