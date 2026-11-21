"""Master runner for all Speech Emotion Recognition & Conversation Analytics experiments."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data" / "cache" / "mpl"))
(ROOT / "data" / "cache" / "mpl").mkdir(parents=True, exist_ok=True)

import numpy as np

from ser_research.experiments_context_esc import run_context_and_export
from ser_research.experiments_segdet import run_segdet_experiments
from ser_research.experiments_utts import run_utterance_experiments
from ser_research.plots import generate_all_plots

FEAT_DIR = ROOT / "data" / "features"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"


def main() -> None:
    t_start = time.perf_counter()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading cached features & dialogs...")
    crema = dict(np.load(FEAT_DIR / "crema_d_features.npz", allow_pickle=False))
    ravdess = dict(np.load(FEAT_DIR / "ravdess_features.npz", allow_pickle=False))
    stream_npz = dict(np.load(FEAT_DIR / "stream_benchmark.npz", allow_pickle=False))
    crema_dialogs = json.loads((FEAT_DIR / "crema_dialogs.json").read_text())
    ravdess_dialogs = json.loads((FEAT_DIR / "ravdess_dialogs.json").read_text())

    utt_bundle = run_utterance_experiments(crema, ravdess, RESULTS_DIR)
    segdet_res = run_segdet_experiments(stream_npz)
    ctx_esc_res = run_context_and_export(crema, ravdess, crema_dialogs, ravdess_dialogs, utt_bundle, MODELS_DIR)

    # Strip large raw tensors before serializing summary JSON
    summary = {
        "total_runtime_s": round(time.perf_counter() - t_start, 2),
        "e1_features": utt_bundle["e1_features"],
        "e2_architectures": utt_bundle["e2_architectures"],
        "e2_t4_resnet": utt_bundle["e2_t4_resnet"],
        "e3_optimizers": utt_bundle["e3_optimizers"],
        "e4_regularization": utt_bundle["e4_regularization"],
        "e8_segdet": segdet_res,
        "e6_e7_context": ctx_esc_res["e6_e7_context"],
        "h2_bootstrap_vs_nocontext": ctx_esc_res["h2_bootstrap_vs_nocontext"],
        "h2_bootstrap_vs_hmm": ctx_esc_res["h2_bootstrap_vs_hmm"],
        "e8_context_windows": ctx_esc_res["e8_context_windows"],
        "ravdess_dialog_zeroshot": ctx_esc_res["ravdess_dialog_zeroshot"],
        "e11_escalation": ctx_esc_res["e11_escalation"],
        "e12_calibration": utt_bundle["e12_calibration"],
        "e14_generative": utt_bundle["e14_generative"],
        "e9_fairness": utt_bundle["e9_fairness"],
        "e10_cross_corpus_ravdess": utt_bundle["e10_cross_corpus_ravdess"],
        "e13_robustness": utt_bundle["e13_robustness"],
        "e13_efficiency_python": utt_bundle["e13_efficiency_python"],
        "onnx_parity": ctx_esc_res["onnx_parity"],
        "sample_attention": ctx_esc_res["sample_attention"],
        "norm_stats": utt_bundle["norm_stats"],
    }

    (RESULTS_DIR / "experiment_results.json").write_text(json.dumps(summary, indent=2))
    generate_all_plots(summary, utt_bundle, RESULTS_DIR / "figures")
    print(f"ALL EXPERIMENTS COMPLETED in {summary['total_runtime_s']}s. Results written to results/experiment_results.json")


if __name__ == "__main__":
    main()
