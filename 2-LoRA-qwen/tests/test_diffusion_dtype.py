"""Regression checks for FP32 NF4 outputs promoting the next Euler state."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from sharm_image21.diffusion import conditioning_output_dtype


class FrozenDiffuserDtypeChecks(unittest.TestCase):
    def test_tuple_output_follows_conditioning_dtype_and_preserves_auxiliary_values(self):
        values = torch.randn(1, 16, 64)
        result = conditioning_output_dtype(None, (), {"encoder_hidden_states": torch.ones(1, 3, 16, dtype=torch.bfloat16)}, (values, "aux"))
        self.assertEqual(result[0].dtype, torch.bfloat16)
        self.assertEqual(result[1], "aux")
        torch.testing.assert_close(result[0], values.to(torch.bfloat16), rtol=0, atol=0)

    def test_model_output_and_float32_conditioning_supported(self):
        result = SimpleNamespace(sample=torch.ones(2, dtype=torch.bfloat16))
        actual = conditioning_output_dtype(None, (), {"encoder_hidden_states": torch.ones(1)}, result)
        self.assertIs(actual, result)
        self.assertEqual(actual.sample.dtype, torch.float32)
        self.assertIs(conditioning_output_dtype(None, (), {}, result), result)

    def test_multiple_euler_steps_keep_bf16_state(self):
        # A CPU surrogate exercises the output boundary; a real 20-step GPU
        # trajectory must additionally be run before accepting the runtime fix.
        class Model(torch.nn.Module):
            def forward(self, hidden_states, encoder_hidden_states):
                if hidden_states.dtype != encoder_hidden_states.dtype:
                    raise ValueError("mixed image/text projection dtype")
                return (hidden_states.float()*0.1,)

        model = Model()
        model.register_forward_hook(conditioning_output_dtype, with_kwargs=True)
        source = torch.ones(1, 8, 4, dtype=torch.bfloat16)
        conditioning = torch.ones(1, 3, 4, dtype=torch.bfloat16)
        for _ in range(20):
            target = model(hidden_states=source, encoder_hidden_states=conditioning)[0]
            source = (source.float()-0.05*target.float()).to(target.dtype)
            self.assertEqual(source.dtype, torch.bfloat16)
            self.assertTrue(torch.isfinite(source).all())


if __name__ == "__main__":
    unittest.main()
