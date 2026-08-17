"""Small deterministic checks; run with `python -m unittest tests/test_core.py`."""
from __future__ import annotations

import unittest

import numpy as np

from src.corruption import missing_frames, random_missing
from src.evaluation import metrics
from src.tensor_completion import complete_tensor


class TensorCompletionTests(unittest.TestCase):
    def setUp(self) -> None:
        # Exact rank-one toy video: this lets the low-rank model be checked
        # without claiming anything about a real-video experiment.
        a = np.array([0.2, 0.6, 1.0])
        b = np.array([0.3, 0.7, 1.0, 0.5])
        c = np.array([0.5, 0.8, 0.4, 0.9, 0.6])
        self.tensor = np.einsum("i,j,k->ijk", a, b, c)

    def test_missing_frame_mask(self) -> None:
        corrupted = missing_frames(self.tensor, gap_length=2, start=1)
        self.assertFalse(corrupted.observed_mask[:, :, 1:3].any())
        self.assertTrue(corrupted.evaluation_mask[:, :, 1:3].all())
        self.assertTrue(np.array_equal(corrupted.observed[corrupted.observed_mask], self.tensor[corrupted.observed_mask]))

    def test_cp_completion_keeps_observations(self) -> None:
        corrupted = random_missing(self.tensor, 0.25, np.random.default_rng(4))
        completed = complete_tensor(corrupted.observed, corrupted.observed_mask, "cp", rank=1, max_iterations=15, tolerance=1e-6)
        self.assertTrue(np.array_equal(completed.tensor[corrupted.observed_mask], corrupted.observed[corrupted.observed_mask]))
        self.assertLess(metrics(self.tensor, completed.tensor, corrupted.evaluation_mask)["rmse"], 0.12)


if __name__ == "__main__":
    unittest.main()
