import json
from pathlib import Path
import tempfile
import unittest

from scripts.policy_eval.resume_io import (
    append_checkpoint,
    atomic_write_json,
    job_key,
    load_checkpoint,
)


class ResumeIoTests(unittest.TestCase):
    def test_truncated_final_line_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text('{"job_key":"a"}\n{"job_key":', encoding="utf-8")
            rows = load_checkpoint(path, required_keys={"job_key"})
            self.assertEqual(rows, {"a": {"job_key": "a"}})

    def test_job_key_changes_with_config(self):
        a = job_key("manifest", "prompt", "base_raw", 42, "cfg-a")
        b = job_key("manifest", "prompt", "base_raw", 42, "cfg-b")
        self.assertNotEqual(a, b)

    def test_append_rejects_conflicting_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            append_checkpoint(path, {"job_key": "a", "value": 1})
            with self.assertRaises(ValueError):
                append_checkpoint(path, {"job_key": "a", "value": 2})

    def test_atomic_json_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "status.json"
            atomic_write_json(path, {"status": "running"})
            self.assertEqual(json.loads(path.read_text()), {"status": "running"})
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
