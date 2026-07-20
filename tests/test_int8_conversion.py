from __future__ import annotations

import unittest

import numpy as np

from scripts.convert_model_int8 import FP16_MATRIX_TENSORS, quantize_per_output_channel


class Int8ConversionTests(unittest.TestCase):
    def test_per_output_channel_quantization_reconstructs_rows(self) -> None:
        values = np.asarray([[-2.0, -0.25, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        quantized, scales = quantize_per_output_channel(values)

        self.assertEqual(quantized.dtype, np.dtype("int8"))
        self.assertEqual(scales.dtype, np.dtype("float32"))
        self.assertEqual(scales.shape, (2,))
        self.assertTrue(np.all(np.abs(quantized) <= 127))
        reconstructed = quantized.astype(np.float32) * scales[:, None]
        self.assertTrue(np.all(np.abs(reconstructed[0] - values[0]) <= scales[0] / 2 + 1e-7))
        np.testing.assert_array_equal(reconstructed[1], values[1])

    def test_rejects_non_matrix_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank-2"):
            quantize_per_output_channel(np.ones(4, dtype=np.float32))

    def test_vocabulary_facing_matrices_remain_fp16(self) -> None:
        self.assertEqual(
            FP16_MATRIX_TENSORS,
            frozenset({"model.embed_tokens.weight", "lm_head.weight"}),
        )


if __name__ == "__main__":
    unittest.main()
