import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT = Path(__file__).parents[1] / "scripts" / "e2_0003_same_source_ab.py"
SPEC = importlib.util.spec_from_file_location("e2_0003_same_source_ab", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SameSourceHarnessTest(unittest.TestCase):
    def test_prompt_matches_e3_fingerprint(self):
        self.assertEqual(MODULE.prompt_fingerprint(MODULE.make_prompt()), "bce3932a8fc5d121")

    def test_metrics_parse_events_properties_and_result(self):
        metrics = (
            "prompt_tokens=1024\ncontext_capacity_tokens=1280\n"
            "event=MODEL_ALLOC_READY iteration=0 epoch_ms=1000\n"
            'result={"tokens_valid":"PASS","ttft_ms":12.5,"token_ids":[1,2]}\n'
        )
        events, properties, result = MODULE.parse_metrics(metrics)
        self.assertEqual(events["MODEL_ALLOC_READY"], 1000)
        self.assertEqual(properties["context_capacity_tokens"], "1280")
        self.assertEqual(result["token_ids"], [1, 2])
        self.assertEqual(result["ttft_ms"], 12.5)

    def test_phase_memory_requires_nearby_successful_sample(self):
        events = {"PREFILL_END": 1000}
        samples = [{"returncode": 0, "epoch_ms": 1100,
                    "parsed": {"vram_used_bytes": 1234}}]
        self.assertEqual(MODULE.nearest_phase_memory(events, samples, "PREFILL_END")[
            "vram_used_bytes"], 1234)
        samples[0]["epoch_ms"] = 3000
        self.assertIsNone(MODULE.nearest_phase_memory(events, samples, "PREFILL_END")[
            "vram_used_bytes"])

    def test_cooldown_rejects_temperature_at_warning_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "snapshots").mkdir()
            args = SimpleNamespace(cooldown_timeout_seconds=1, gpu_index=0,
                                   expected_bdf="0000:06:00.0")
            state = {"returncode": 0, "pci_bdf": args.expected_bdf, "junction_c": 80.0,
                     "gpu_use_percent": 0.0, "kfd_idle": True, "vram_used_bytes": 0}
            with patch.object(MODULE, "capture_state", return_value=state), \
                    patch.object(MODULE.time, "monotonic", side_effect=(0, 0, 2)), \
                    patch.object(MODULE.time, "sleep"):
                result = MODULE.wait_cool(args, root, "warning", 79.0, 0)
        self.assertFalse(result["ready"])

    def test_route_env_is_the_only_candidate_selector(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "bench"
            model = Path(directory) / "model.gguf"
            binary.touch()
            model.touch()
            args = SimpleNamespace(container_runtime="podman", binary=binary, model=model,
                                   expected_bdf="0000:06:00.0", image="image",
                                   context_profile="8K", prompt_tokens=1024, generate_tokens=32,
                                   capture_raw_logits=False)
            control = MODULE.container_command(args, "control", 1, Path(directory))
            candidate = MODULE.container_command(args, "mmq_only", 1, Path(directory))
        self.assertIn("MIINFER_EXPERIMENTAL_MMQ_GATEUP_ONLY=0", control)
        self.assertIn("MIINFER_EXPERIMENTAL_MMQ_GATEUP_ONLY=1", candidate)
        self.assertEqual(control[control.index("/bench/run")], candidate[candidate.index("/bench/run")])

    def test_raw_logit_capture_is_opt_in_and_route_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "bench"
            model = root / "model.gguf"
            binary.touch()
            model.touch()
            args = SimpleNamespace(container_runtime="podman", binary=binary, model=model,
                                   expected_bdf="0000:06:00.0", image="image",
                                   context_profile="8K", prompt_tokens=2048, generate_tokens=16,
                                   capture_raw_logits=True)
            command = MODULE.container_command(args, "mmq_only", 1, root)
        self.assertEqual(command[-2:], ["--capture-raw-logits", "/results/mmq_only-pair-1.logits.f32"])

    def test_layer_zero_trace_is_opt_in_and_targets_first_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "bench"
            model = root / "model.gguf"
            binary.touch()
            model.touch()
            args = SimpleNamespace(container_runtime="podman", binary=binary, model=model,
                                   expected_bdf="0000:06:00.0", image="image",
                                   context_profile="8K", prompt_tokens=2048, generate_tokens=5,
                                   capture_layer_zero_trace=True)
            command = MODULE.container_command(args, "control", 1, root)
        self.assertIn("MIINFER_LC_OP_TRACE_PREFIX=/results/control-optrace", command)
        self.assertIn("MIINFER_LC_DECODE_TRACE_POSITION=2048", command)
        self.assertLess(command.index("MIINFER_LC_DECODE_TRACE_POSITION=2048"), command.index("image"))

    def test_context_workload_is_passed_through_to_the_same_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "bench"
            model = root / "model.gguf"
            binary.touch()
            model.touch()
            args = SimpleNamespace(container_runtime="podman", binary=binary, model=model,
                                   expected_bdf="0000:06:00.0", image="image",
                                   context_profile="8K", prompt_tokens=8192, generate_tokens=16)
            command = MODULE.container_command(args, "mmq_only", 1, root)
        self.assertEqual(command[command.index("8K") + 1:command.index("8K") + 5],
                         ["--generate", "16", "--prompt-tokens", "8192"])

    def test_sampler_contract_is_explicit_and_rejects_legacy_penalty(self):
        self.assertEqual(MODULE.SAMPLER_CONFIG["repetition_penalty"], "1")
        self.assertEqual(MODULE.SAMPLER_CONFIG["sampler"], "greedy_argmax")
        config = dict(MODULE.SAMPLER_CONFIG)
        self.assertTrue(MODULE.sampler_config_valid(config))
        config["repetition_penalty"] = "1.15"
        self.assertFalse(MODULE.sampler_config_valid(config))

    def test_workload_matches_qualification_pair(self):
        self.assertEqual(MODULE.PROMPT_TOKENS, 1024)
        self.assertEqual(MODULE.OUTPUT_TOKENS, 32)
        self.assertEqual(MODULE.CONTEXT_TOKENS, 1280)


if __name__ == "__main__":
    unittest.main()
