"""Automated Python unit and leakage-safety tests for dataset pipeline, DSP features, and models."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "src"))

from ser_research.data_pipeline import compute_escalation_onset
from ser_research.features import N_FRAMES, N_MELS, SAMPLE_RATE, extract_all_features
from ser_research.models import (
    ContextLstm,
    ContextTransformer,
    Conv3DSer,
    FasterRCNN1D,
    GoogLeNetSer,
    ResNetSer,
    RetinaNet1D,
    SpecCVAE,
    SpecUNet1D,
    YOLO1D,
)


class TestSerPipelineAndModels(unittest.TestCase):
    def test_feature_extraction_shapes_and_finiteness(self) -> None:
        t = np.linspace(0, 2.0, 2 * SAMPLE_RATE, endpoint=False)
        sig = 0.5 * np.sin(2 * np.pi * 220.0 * t) + 0.25 * np.sin(2 * np.pi * 440.0 * t)
        feats = extract_all_features(sig)
        self.assertEqual(feats["log_mel"].shape, (N_MELS, N_FRAMES))
        self.assertEqual(feats["mfcc_delta"].shape, (40, N_FRAMES))
        self.assertEqual(feats["functionals"].shape, (64,))
        self.assertTrue(np.all(np.isfinite(feats["log_mel"])))
        self.assertTrue(np.all(np.isfinite(feats["functionals"])))
        self.assertGreater(float(feats["f0_mean"]), 180.0)
        self.assertLess(float(feats["f0_mean"]), 260.0)

    def test_escalation_onset_definition(self) -> None:
        emos = ["NEU", "NEU", "NEU", "ANG", "DIS", "ANG"]
        ints = ["XX", "XX", "XX", "HI", "XX", "HI"]
        self.assertEqual(compute_escalation_onset(emos, ints), 3)
        # Chronic negative has no calm baseline -> should NOT trigger false onset
        chronic_e = ["ANG", "DIS", "ANG", "ANG", "DIS", "ANG"]
        chronic_i = ["HI", "XX", "HI", "HI", "XX", "HI"]
        self.assertIsNone(compute_escalation_onset(chronic_e, chronic_i))

    def test_speaker_leakage_zero_overlap_if_built(self) -> None:
        feat_file = ROOT / "data" / "features" / "crema_d_features.npz"
        if not feat_file.exists():
            self.skipTest("crema_d_features.npz not generated yet")
        data = np.load(feat_file, allow_pickle=False)
        splits = data["splits"]
        actors = data["actor_ids"]
        tr_a = set(actors[splits == "train"].tolist())
        va_a = set(actors[splits == "val"].tolist())
        te_a = set(actors[splits == "test"].tolist())
        self.assertTrue(tr_a.isdisjoint(va_a))
        self.assertTrue(tr_a.isdisjoint(te_a))
        self.assertTrue(va_a.isdisjoint(te_a))
        self.assertEqual(len(tr_a) + len(va_a) + len(te_a), 91)

    def test_model_forward_shapes(self) -> None:
        x2d = torch.randn(2, 1, 64, 128)
        for cls in (ResNetSer, GoogLeNetSer):
            lg, em = cls()(x2d)
            self.assertEqual(lg.shape, (2, 6))
        lg3, _ = Conv3DSer()(torch.randn(2, 1, 3, 64, 128))
        self.assertEqual(lg3.shape, (2, 6))

        xs = torch.randn(2, 1, 64, 512)
        self.assertEqual(SpecUNet1D()(xs).shape, (2, 3, 512))
        self.assertEqual(YOLO1D()(xs).shape, (2, 16, 5))
        c_r, b_r, a_r = RetinaNet1D()(xs)
        self.assertEqual(c_r.shape[0], 2)
        self.assertEqual(b_r.shape[1], a_r.shape[0])
        _, rpn_o, _ = FasterRCNN1D().forward_rpn(xs)
        self.assertEqual(rpn_o.shape, (2, 64 * 3))

        seq = torch.randn(2, 10, 130)
        clg, cesc, attn = ContextTransformer()(seq)
        self.assertEqual(clg.shape, (2, 10, 6))
        self.assertEqual(cesc.shape, (2, 10, 1))
        self.assertEqual(attn.shape, (2, 10, 10))
        l_lg, l_esc = ContextLstm()(seq)
        self.assertEqual(l_lg.shape, (2, 10, 6))

        rec, mu, lv = SpecCVAE()(x2d, torch.eye(6)[:2])
        self.assertEqual(rec.shape, (2, 1, 64, 128))


if __name__ == "__main__":
    unittest.main()
