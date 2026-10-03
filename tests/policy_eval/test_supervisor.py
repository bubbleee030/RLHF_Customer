import unittest

from scripts.policy_eval.supervisor import (
    SupervisorState,
    classify_worker_failure,
    container_cache_mount,
    container_pythonpath,
    docker_gpu_request,
    retry_delay,
)


class SupervisorTests(unittest.TestCase):
    def test_completed_stage_is_not_relaunched(self):
        state = SupervisorState(completed_stages=["preflight", "generation"])
        self.assertEqual(state.next_stage(), "scoring")

    def test_backoff_is_capped(self):
        self.assertEqual(retry_delay(1), 1)
        self.assertEqual(retry_delay(20), 300)

    def test_oom_restarts_once_then_becomes_permanent_at_batch_one(self):
        self.assertEqual(classify_worker_failure("CUDA out of memory", 1), "transient")
        self.assertEqual(classify_worker_failure("CUDA out of memory", 2), "permanent")

    def test_low_space_is_permanent_and_timeout_is_transient(self):
        self.assertEqual(classify_worker_failure("low space: 10 bytes", 1), "permanent")
        self.assertEqual(classify_worker_failure("request timed out", 50), "transient")

    def test_blocked_state_has_exact_reason(self):
        state = SupervisorState()
        state.block("disk reserve violated")
        self.assertEqual(state.status, "blocked")
        self.assertEqual(state.block_reason, "disk reserve violated")

    def test_successful_completion_clears_stale_block_reason(self):
        state = SupervisorState()
        state.block("missing manifest")
        state.mark_complete()
        self.assertEqual(state.status, "complete")
        self.assertIsNone(state.block_reason)
        self.assertIsNone(state.current_stage)

    def test_container_pythonpath_includes_historical_ppo_module_directory(self):
        self.assertEqual(
            container_pythonpath(),
            "/deps:/eval_code:/workspace:/workspace/scripts/ppo_lag",
        )

    def test_multiple_gpu_device_ids_keep_csv_quotes_in_argv(self):
        self.assertEqual(docker_gpu_request("0"), "device=0")
        self.assertEqual(docker_gpu_request("0,1"), '\"device=0,1\"')

    def test_hf_cache_is_mounted_read_only_at_historical_container_path(self):
        self.assertEqual(
            container_cache_mount("/home/ubuntu/.cache/huggingface"),
            "/home/ubuntu/.cache/huggingface:/root/.cache/huggingface:ro",
        )


if __name__ == "__main__":
    unittest.main()
