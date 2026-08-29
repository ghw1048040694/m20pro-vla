from pathlib import Path
import unittest

try:
    import torch
    from m20pro_vla.policies import M20MuJoCoVLA, TEXT_TOKEN_LENGTH, encode_text
except ImportError:  # The base package remains testable without optional VLA deps.
    torch = None
    M20MuJoCoVLA = None
    TEXT_TOKEN_LENGTH = None
    encode_text = None


ROOT = Path(__file__).resolve().parents[1]


class ProjectLayoutTest(unittest.TestCase):
    def test_canonical_mujoco_cli_and_compatibility_entrypoint_exist(self):
        self.assertTrue((ROOT / "src/m20pro_vla/cli.py").is_file())
        self.assertTrue((ROOT / "src/m20pro_vla/runtime/run_context.py").is_file())
        self.assertTrue((ROOT / "scripts/mujoco/train_m20_mujoco_vla.py").is_file())
        self.assertTrue((ROOT / "scripts/train_m20_mujoco_vla.py").is_file())
        self.assertTrue((ROOT / "scripts/mujoco/evaluate_m20_rough_terrain_controller.py").is_file())
        self.assertTrue((ROOT / "docs/PROJECT_LAYOUT.md").is_file())

    @unittest.skipUnless(torch is not None, "optional VLA dependencies are not installed")
    def test_compact_policy_emits_body_command_shape(self):
        model = M20MuJoCoVLA().eval()
        with torch.no_grad():
            output = model(
                torch.zeros(1, 6, 48, 80),
                torch.zeros(1, 72),
                torch.zeros(1, 45),
                torch.from_numpy(encode_text("Go to the red cube.")).unsqueeze(0),
            )
        self.assertEqual(tuple(output.shape), (1, 4))

    @unittest.skipUnless(torch is not None, "optional VLA dependencies are not installed")
    def test_text_encoder_preserves_utf8_and_ascii_contract(self):
        ascii_tokens = encode_text("go to the red cube")
        utf8_tokens = encode_text("找到红色方块")
        self.assertEqual(ascii_tokens[0], ord("g") + 1)
        self.assertEqual(len(ascii_tokens), TEXT_TOKEN_LENGTH)
        self.assertGreater(int((utf8_tokens > 0).sum()), 0)
        self.assertGreater(int(utf8_tokens.max()), 128)
        self.assertFalse((utf8_tokens == encode_text("")).all())

    @unittest.skipUnless(torch is not None, "optional VLA dependencies are not installed")
    def test_existing_checkpoint_contracts_are_loadable(self):
        loaded = 0
        for checkpoint_path in sorted((ROOT / ".runtime/checkpoints").glob("*/best.pt")):
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model = M20MuJoCoVLA.for_state_dict(payload["model_state_dict"])
            try:
                model.load_state_dict(payload["model_state_dict"])
            except RuntimeError:
                continue
            loaded += 1
        self.assertGreater(loaded, 0, "No current checkpoint matched the compact policy contract")


if __name__ == "__main__":
    unittest.main()
