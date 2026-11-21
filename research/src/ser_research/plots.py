"""Publication-quality SVG figure generation from verified experimental results."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EMOTIONS_6 = ["NEU", "HAP", "SAD", "ANG", "FEA", "DIS"]


def generate_all_plots(all_results: dict[str, Any], bundle: dict[str, Any], fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.facecolor": "#0f172a",
        "axes.facecolor": "#111827",
        "axes.edgecolor": "#475569",
        "axes.labelcolor": "#e2e8f0",
        "xtick.color": "#cbd5e1",
        "ytick.color": "#cbd5e1",
        "text.color": "#f8fafc",
        "grid.color": "#334155",
        "font.size": 9,
    })

    # 1. E1 & E2 Model Benchmark
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    e1 = all_results["e1_features"]
    names1 = list(e1.keys())
    uars1 = [e1[k]["uar"] * 100 for k in names1]
    axes[0].barh(names1, uars1, color="#3b82f6")
    axes[0].set_title("E1: Acoustic Feature Representation (CNN-Basic)")
    axes[0].set_xlabel("Speaker-Independent UAR (%)")
    axes[0].grid(axis="x", alpha=0.4)

    e2 = all_results["e2_architectures"]
    names2 = list(e2.keys())
    uars2 = [e2[k].get("uar_mean", e2[k]["uar"]) * 100 for k in names2]
    f1s2 = [e2[k].get("macro_f1_mean", e2[k]["macro_f1"]) * 100 for k in names2]
    y_pos = np.arange(len(names2))
    axes[1].barh(y_pos - 0.18, uars2, height=0.35, label="UAR (%)", color="#10b981")
    axes[1].barh(y_pos + 0.18, f1s2, height=0.35, label="Macro-F1 (%)", color="#f59e0b")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(names2)
    axes[1].set_title("E2: Architecture Benchmark (CREMA-D 6-Class)")
    axes[1].set_xlabel("Score (%)")
    axes[1].legend(facecolor="#1e293b", edgecolor="#475569")
    axes[1].grid(axis="x", alpha=0.4)
    fig.tight_layout()
    fig.savefig(fig_dir / "e1_e2_model_benchmark.svg", format="svg")
    plt.close(fig)

    # 2. E6/E7 Conversational Context & Agreement Stratification
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ctx = all_results["e6_e7_context"]
    c_names = list(ctx.keys())
    c_uars = [ctx[k].get("uar_mean", ctx[k]["uar"]) * 100 for k in c_names]
    axes[0].barh(c_names, c_uars, color=["#64748b", "#94a3b8", "#38bdf8", "#0284c7", "#22c55e", "#16a34a"])
    axes[0].set_title("E6/E7: Conversational Context UAR (CREMA-Dialog)")
    axes[0].set_xlabel("UAR (%)")
    axes[0].grid(axis="x", alpha=0.4)

    bins = ["Low (<0.50)", "Medium (0.50-0.75)", "High (>=0.75)"]
    nc_bins = [ctx["No-Context (ResNet-SER)"]["agreement_stratified"].get(b, {}).get("uar", 0) * 100 for b in bins]
    hmm_bins = [ctx["HMM-Forward (Control)"]["agreement_stratified"].get(b, {}).get("uar", 0) * 100 for b in bins]
    tf_bins = [ctx["ResNet + Causal Transformer"]["agreement_stratified"].get(b, {}).get("uar", 0) * 100 for b in bins]
    xp = np.arange(len(bins))
    axes[1].bar(xp - 0.25, nc_bins, width=0.25, label="No-Context", color="#64748b")
    axes[1].bar(xp, hmm_bins, width=0.25, label="HMM Control", color="#f59e0b")
    axes[1].bar(xp + 0.25, tf_bins, width=0.25, label="Causal Transformer", color="#22c55e")
    axes[1].set_xticks(xp)
    axes[1].set_xticklabels(bins)
    axes[1].set_ylabel("UAR (%)")
    axes[1].set_title("H2 Ambiguity Test: Context Gain by Human Listener Agreement")
    axes[1].legend(facecolor="#1e293b", edgecolor="#475569")
    axes[1].grid(axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(fig_dir / "e6_e7_context_and_agreement.svg", format="svg")
    plt.close(fig)

    # 3. E8 Segmentation & Detection Benchmark
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    sd = all_results["e8_segdet"]
    sd_names = list(sd.keys())
    maps = [sd[k]["mAP@0.5"] * 100 for k in sd_names]
    ef1s = [sd[k]["event_f1"] * 100 for k in sd_names]
    lats = [sd[k]["latency_ms_per_stream"] for k in sd_names]
    y_p = np.arange(len(sd_names))
    axes[0].barh(y_p - 0.18, maps, height=0.35, label="mAP@0.5 (%)", color="#a855f7")
    axes[0].barh(y_p + 0.18, ef1s, height=0.35, label="Event F1 (%)", color="#ec4899")
    axes[0].set_yticks(y_p)
    axes[0].set_yticklabels(sd_names)
    axes[0].set_title("E8-SegDet: Accuracy (mAP@0.5 & Event F1)")
    axes[0].legend(facecolor="#1e293b", edgecolor="#475569")
    axes[0].grid(axis="x", alpha=0.4)

    axes[1].scatter(lats, maps, s=110, color="#38bdf8", edgecolors="#f8fafc")
    for name, lx, my in zip(sd_names, lats, maps):
        axes[1].annotate(name.split(" ")[0], (lx, my), xytext=(5, 4), textcoords="offset points", fontsize=8)
    axes[1].set_xlabel("Median Inference Latency (ms / 10s stream)")
    axes[1].set_ylabel("mAP@0.5 (%)")
    axes[1].set_title("Accuracy vs Speed Trade-Off (Seg/Det)")
    axes[1].grid(alpha=0.4)
    fig.tight_layout()
    fig.savefig(fig_dir / "e8_segdet_benchmark.svg", format="svg")
    plt.close(fig)

    # 4. Conv Feature Maps & Grad-CAM Saliency
    gc = bundle["gradcam_sample"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].imshow(gc["input_spec"], origin="lower", aspect="auto", cmap="magma")
    axes[0].set_title("Input Log-Mel (Angry Utterance)")
    axes[1].imshow(gc["conv_fmap0"], origin="lower", aspect="auto", cmap="viridis")
    axes[1].set_title("ResNet Stage-4 Feature Map #0")
    axes[2].imshow(gc["input_spec"], origin="lower", aspect="auto", cmap="gray", alpha=0.65)
    axes[2].imshow(
        np.kron(np.array(gc["gradcam"]), np.ones((16, 16))),
        origin="lower", aspect="auto", cmap="jet", alpha=0.45
    )
    axes[2].set_title("Grad-CAM Saliency (Time-Freq Focus)")
    fig.tight_layout()
    fig.savefig(fig_dir / "conv_activations_gradcam.svg", format="svg")
    plt.close(fig)

    # 5. Latent Space & Activation Behavior (Step vs Sigmoid vs Tanh)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    gen_info = all_results["e14_generative"]
    pca_res = np.array(gen_info["pca_resnet_emb"])
    pca_cvae = np.array(gen_info["pca_cvae_latent"])
    pca_lbl = np.array(gen_info["pca_labels"])
    colors = ["#94a3b8", "#22c55e", "#3b82f6", "#ef4444", "#a855f7", "#f59e0b"]
    for c_i, ename in enumerate(EMOTIONS_6):
        m = pca_lbl == c_i
        axes[0].scatter(pca_res[m, 0], pca_res[m, 1], s=14, alpha=0.75, label=ename, color=colors[c_i])
        axes[1].scatter(pca_cvae[m, 0], pca_cvae[m, 1], s=14, alpha=0.75, label=ename, color=colors[c_i])
    axes[0].set_title("ResNet 128-d Embedding Space (PCA)")
    axes[1].set_title("CVAE 16-d Latent Space z (PCA)")
    axes[0].legend(fontsize=7, facecolor="#1e293b", edgecolor="#475569")

    xs = np.linspace(-4, 4, 200)
    axes[2].plot(xs, (xs >= 0).astype(float), label="Step (0/1)", lw=2)
    axes[2].plot(xs, 1.0 / (1.0 + np.exp(-xs)), label="Sigmoid (0..1)", lw=2)
    axes[2].plot(xs, np.tanh(xs), label="Tanh (-1..1)", lw=2)
    axes[2].set_title("Step vs Sigmoid vs Tanh Behavior")
    axes[2].legend(fontsize=8, facecolor="#1e293b", edgecolor="#475569")
    axes[2].grid(alpha=0.35)
    fig.tight_layout()
    fig.savefig(fig_dir / "latent_space_and_activations.svg", format="svg")
    plt.close(fig)

    # 6. Confusion Matrix & Temperature Calibration Reliability Diagram
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    cal = all_results["e12_calibration"]
    cm = np.array(cal["after"]["confusion_matrix_norm"])
    im = axes[0].imshow(cm, cmap="Blues", vmin=0.0, vmax=0.75)
    axes[0].set_xticks(range(6))
    axes[0].set_yticks(range(6))
    axes[0].set_xticklabels(EMOTIONS_6)
    axes[0].set_yticklabels(EMOTIONS_6)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title("ResNet-SER Normalized Confusion Matrix")
    for r in range(6):
        for c in range(6):
            axes[0].text(c, r, f"{cm[r, c]*100:.1f}%", ha="center", va="center",
                         color="white" if cm[r, c] > 0.35 else "#cbd5e1", fontsize=8)

    b_conf = cal["before"]["reliability_bins"]["confidence"]
    b_acc = cal["before"]["reliability_bins"]["accuracy"]
    a_conf = cal["after"]["reliability_bins"]["confidence"]
    a_acc = cal["after"]["reliability_bins"]["accuracy"]
    axes[1].plot([0, 1], [0, 1], "--", color="#94a3b8", label="Perfect Calibration")
    axes[1].plot(b_conf, b_acc, "o-", label=f"Uncalibrated (ECE={cal['before']['ece']:.3f})", color="#ef4444")
    axes[1].plot(a_conf, a_acc, "s-", label=f"Temp-Scaled T={cal['temperature']:.2f} (ECE={cal['after']['ece']:.3f})", color="#22c55e")
    axes[1].set_xlabel("Predicted Confidence")
    axes[1].set_ylabel("Empirical Accuracy")
    axes[1].set_title("Reliability Diagram (Before vs After Calibration)")
    axes[1].legend(fontsize=8, facecolor="#1e293b", edgecolor="#475569")
    axes[1].grid(alpha=0.35)
    fig.tight_layout()
    fig.savefig(fig_dir / "confusion_matrix_and_calibration.svg", format="svg")
    plt.close(fig)

    # 7. Escalation Detectors & Subgroup Fairness / Noise Robustness
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    dets = all_results["e11_escalation"]["detectors"]
    d_names = list(dets.keys())
    auprcs = [dets[k]["auprc"] for k in d_names]
    aurocs = [dets[k]["auroc"] for k in d_names]
    yp = np.arange(len(d_names))
    axes[0].barh(yp - 0.18, auprcs, height=0.35, label="AUPRC", color="#f97316")
    axes[0].barh(yp + 0.18, aurocs, height=0.35, label="AUROC", color="#38bdf8")
    axes[0].set_yticks(yp)
    axes[0].set_yticklabels(d_names, fontsize=8)
    axes[0].set_title("E11: Frustration Escalation Detection Benchmark")
    axes[0].legend(facecolor="#1e293b", edgecolor="#475569")
    axes[0].grid(axis="x", alpha=0.4)

    rob = all_results["e13_robustness"]
    r_names = list(rob.keys())
    r_uars = [rob[k]["uar"] * 100 for k in r_names]
    axes[1].plot(range(len(r_names)), r_uars, "o-", lw=2.2, color="#22c55e")
    axes[1].set_xticks(range(len(r_names)))
    axes[1].set_xticklabels(r_names, rotation=22, ha="right", fontsize=8)
    axes[1].set_ylabel("UAR (%)")
    axes[1].set_title("E13: Acoustic Noise & Telephony Codec Robustness")
    axes[1].grid(alpha=0.4)
    fig.tight_layout()
    fig.savefig(fig_dir / "escalation_and_robustness.svg", format="svg")
    plt.close(fig)
    print(f"Generated 7 SVG figures in {fig_dir}")
