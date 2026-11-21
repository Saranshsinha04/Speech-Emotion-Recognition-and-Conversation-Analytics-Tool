"""Utterance-level SER experiments: E1 features, E2 architectures, E3 optimizers, E4 reg, E5/E14 aug+CVAE+cGAN, E9/E10/E13 fairness/cross-corpus/robustness."""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset

from ser_research.metrics import (
    compute_classification_metrics,
    fit_temperature_scaling,
    softmax_np,
)
from ser_research.models import (
    CnnBasic,
    CnnDeep,
    Conv3DSer,
    GoogLeNetSer,
    MlpBaseline,
    ResNetSer,
    SpecCGANDiscriminator,
    SpecCGANGenerator,
    SpecCVAE,
)

EMOTIONS_6 = ["NEU", "HAP", "SAD", "ANG", "FEA", "DIS"]
EMOTIONS_4 = ["NEU", "HAP", "SAD", "ANG"]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def spec_augment(x: torch.Tensor, time_mask: int = 16, freq_mask: int = 8) -> torch.Tensor:
    out = x.clone()
    _, _, f, t = out.shape
    if np.random.rand() < 0.75:
        fw = int(np.random.randint(1, freq_mask + 1))
        f0 = int(np.random.randint(0, max(1, f - fw)))
        out[:, :, f0 : f0 + fw, :] = 0.0
    if np.random.rand() < 0.75:
        tw = int(np.random.randint(1, time_mask + 1))
        t0 = int(np.random.randint(0, max(1, t - tw)))
        out[:, :, :, t0 : t0 + tw] = 0.0
    if np.random.rand() < 0.5:
        out = out + 0.06 * torch.randn_like(out)
    return out


def train_classifier(
    model: nn.Module,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    epochs: int = 12,
    batch_size: int = 128,
    lr: float = 2.2e-3,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
    augment: bool = False,
    optimizer_name: str = "adamw",
    seed: int = 42,
) -> tuple[nn.Module, dict[str, list[float]]]:
    set_seed(seed)
    counts = np.bincount(y_tr, minlength=6).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    weights = np.clip(weights, 0.5, 2.5)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights), label_smoothing=label_smoothing)

    if optimizer_name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr * 10.0, weight_decay=weight_decay)
    elif optimizer_name == "momentum":
        opt = torch.optim.SGD(model.parameters(), lr=lr * 8.0, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == "nesterov":
        opt = torch.optim.SGD(model.parameters(), lr=lr * 8.0, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    tr_ds = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr))
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)

    best_state = copy.deepcopy(model.state_dict())
    best_val_uar = -1.0
    hist = {"train_loss": [], "val_loss": [], "val_uar": []}

    x_va_t = torch.from_numpy(x_va)
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        n_tr = 0
        for xb, yb in tr_dl:
            if augment and xb.ndim == 4:
                xb = spec_augment(xb)
            opt.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            total_loss += float(loss.item()) * len(yb)
            n_tr += len(yb)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            va_logits, _ = predict_batched(model, x_va_t)
            va_loss = float(F.cross_entropy(va_logits, torch.from_numpy(y_va)).item())
            va_probs = softmax_np(va_logits.numpy())
            va_m = compute_classification_metrics(y_va, va_probs, EMOTIONS_6)
        hist["train_loss"].append(round(total_loss / max(1, n_tr), 4))
        hist["val_loss"].append(round(va_loss, 4))
        hist["val_uar"].append(va_m["uar"])
        if va_m["uar"] > best_val_uar:
            best_val_uar = va_m["uar"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    return model, hist


@torch.no_grad()
def predict_batched(model: nn.Module, x_t: torch.Tensor, batch_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_all, emb_all = [], []
    for i in range(0, len(x_t), batch_size):
        lg, em = model(x_t[i : i + batch_size])
        logits_all.append(lg)
        emb_all.append(em)
    return torch.cat(logits_all, dim=0), torch.cat(emb_all, dim=0)


def make_3d_volume(log_mel: np.ndarray) -> np.ndarray:
    """Builds [N, 1, 3, 64, 128] volume: (Log-Mel, Time-Delta, Freq-Delta)."""
    t_delta = np.zeros_like(log_mel)
    t_delta[:, :, 1:] = log_mel[:, :, 1:] - log_mel[:, :, :-1]
    f_delta = np.zeros_like(log_mel)
    f_delta[:, 1:, :] = log_mel[:, 1:, :] - log_mel[:, :-1, :]
    vol = np.stack([log_mel, t_delta, f_delta], axis=1)[:, None, :, :, :]
    return vol.astype(np.float32)


def run_utterance_experiments(
    crema: dict[str, np.ndarray],
    ravdess: dict[str, np.ndarray],
    results_dir: Path,
) -> dict[str, Any]:
    torch.set_num_threads(6)
    splits = crema["splits"]
    tr = splits == "train"
    va = splits == "val"
    te = splits == "test"

    y_tr = crema["labels6"][tr].astype(np.int64)
    y_va = crema["labels6"][va].astype(np.int64)
    y_te = crema["labels6"][te].astype(np.int64)

    # Standardize each representation strictly using TRAIN split statistics
    def std_fit_transform(arr: np.ndarray) -> tuple[np.ndarray, float, float]:
        mu = float(arr[tr].mean())
        sd = max(1e-6, float(arr[tr].std()))
        return ((arr - mu) / sd).astype(np.float32), mu, sd

    log_mel_n, lm_mu, lm_sd = std_fit_transform(crema["log_mel"])
    mel_lin_n, _, _ = std_fit_transform(crema["mel_linear"])
    stft_n, _, _ = std_fit_transform(crema["stft_64"])
    mfcc_d_n, _, _ = std_fit_transform(crema["mfcc_delta"])

    func_raw = crema["functionals"]
    func_mu = func_raw[tr].mean(axis=0, keepdims=True)
    func_sd = np.maximum(1e-6, func_raw[tr].std(axis=0, keepdims=True))
    func_n = ((func_raw - func_mu) / func_sd).astype(np.float32)

    # ------------------------------------------------------------------
    # E1: Feature representation comparison (same CnnBasic architecture)
    # ------------------------------------------------------------------
    print("Running E1: Feature representation comparison...", flush=True)
    e1_results = {}
    for feat_name, arr in [
        ("Log-Mel (64x128)", log_mel_n[:, None, :, :]),
        ("Linear Mel (64x128)", mel_lin_n[:, None, :, :]),
        ("STFT Mag (64x128)", stft_n[:, None, :, :]),
        ("MFCC + Delta (40x128)", mfcc_d_n[:, None, :, :]),
    ]:
        t0 = time.perf_counter()
        m = CnnBasic(in_ch=1, num_classes=6)
        m, _ = train_classifier(m, arr[tr], y_tr, arr[va], y_va, epochs=8, lr=2.2e-3, seed=42)
        with torch.no_grad():
            te_lg, _ = predict_batched(m, torch.from_numpy(arr[te]))
        metrics = compute_classification_metrics(y_te, softmax_np(te_lg.numpy()), EMOTIONS_6)
        metrics["train_time_s"] = round(time.perf_counter() - t0, 2)
        e1_results[feat_name] = metrics
        print(f"  E1 {feat_name}: UAR={metrics['uar']:.4f}, F1={metrics['macro_f1']:.4f}", flush=True)

    # ------------------------------------------------------------------
    # E2: Architecture comparison (multi-seed for CNN-Basic & ResNet-SER)
    # ------------------------------------------------------------------
    print("Running E2: Architecture comparison (MLP, CNN-Basic, CNN-Deep, ResNet, GoogLeNet, Conv3D)...", flush=True)
    e2_results = {}
    x_lm_tr, x_lm_va, x_lm_te = log_mel_n[tr, None], log_mel_n[va, None], log_mel_n[te, None]

    # MLP Baseline
    t0 = time.perf_counter()
    mlp = MlpBaseline(in_dim=64, num_classes=6)
    mlp, mlp_hist = train_classifier(mlp, func_n[tr], y_tr, func_n[va], y_va, epochs=24, lr=2.5e-3, seed=42)
    with torch.no_grad():
        lg_mlp, _ = predict_batched(mlp, torch.from_numpy(func_n[te]))
    m_mlp = compute_classification_metrics(y_te, softmax_np(lg_mlp.numpy()), EMOTIONS_6)
    m_mlp["params"] = sum(p.numel() for p in mlp.parameters())
    m_mlp["train_time_s"] = round(time.perf_counter() - t0, 2)
    e2_results["MLP-Baseline"] = m_mlp

    # Multi-seed helper for key CNNs
    best_resnet = None
    best_resnet_hist = None
    best_resnet_te_logits = None
    best_resnet_va_logits = None

    for name, builder, epochs, aug, seeds in [
        ("CNN-Basic", lambda: CnnBasic(1, 6), 10, False, [42, 43]),
        ("CNN-Deep", lambda: CnnDeep(1, 6), 10, False, [42]),
        ("GoogLeNet-SER", lambda: GoogLeNetSer(1, 6), 10, False, [42]),
        ("ResNet-SER", lambda: ResNetSer(1, 6, 128), 10, False, [42]),
        ("ResNet-SER + SpecAug", lambda: ResNetSer(1, 6, 128), 12, True, [42, 43]),
    ]:
        seed_uars, seed_f1s, seed_accs = [], [], []
        last_metrics = None
        last_hist = None
        t0 = time.perf_counter()
        for sd in seeds:
            model = builder()
            model, hist = train_classifier(
                model, x_lm_tr, y_tr, x_lm_va, y_va,
                epochs=epochs, lr=2.2e-3, augment=aug, seed=sd
            )
            with torch.no_grad():
                va_lg, _ = predict_batched(model, torch.from_numpy(x_lm_va))
                te_lg, _ = predict_batched(model, torch.from_numpy(x_lm_te))
            met = compute_classification_metrics(y_te, softmax_np(te_lg.numpy()), EMOTIONS_6)
            seed_uars.append(met["uar"])
            seed_f1s.append(met["macro_f1"])
            seed_accs.append(met["accuracy"])
            last_metrics = met
            last_hist = hist
            if name == "ResNet-SER + SpecAug" and sd == 42:
                best_resnet = model
                best_resnet_hist = hist
                best_resnet_va_logits = va_lg.numpy()
                best_resnet_te_logits = te_lg.numpy()

        last_metrics["uar_mean"] = round(float(np.mean(seed_uars)), 4)
        last_metrics["uar_std"] = round(float(np.std(seed_uars)), 4)
        last_metrics["macro_f1_mean"] = round(float(np.mean(seed_f1s)), 4)
        last_metrics["accuracy_mean"] = round(float(np.mean(seed_accs)), 4)
        last_metrics["params"] = sum(p.numel() for p in builder().parameters())
        last_metrics["train_time_s"] = round((time.perf_counter() - t0) / len(seeds), 2)
        last_metrics["history"] = last_hist
        e2_results[name] = last_metrics
        print(f"  E2 {name}: UAR={last_metrics['uar_mean']:.4f}±{last_metrics['uar_std']:.4f}, F1={last_metrics['macro_f1_mean']:.4f}", flush=True)

    # Conv3D-SER on [N, 1, 3, 64, 128]
    t0 = time.perf_counter()
    vol_all = make_3d_volume(log_mel_n)
    m3d = Conv3DSer(6)
    m3d, _ = train_classifier(m3d, vol_all[tr], y_tr, vol_all[va], y_va, epochs=8, lr=2.2e-3, seed=42)
    with torch.no_grad():
        lg3d, _ = predict_batched(m3d, torch.from_numpy(vol_all[te]))
    met3d = compute_classification_metrics(y_te, softmax_np(lg3d.numpy()), EMOTIONS_6)
    met3d["params"] = sum(p.numel() for p in m3d.parameters())
    met3d["train_time_s"] = round(time.perf_counter() - t0, 2)
    e2_results["Conv3D-SER"] = met3d
    print(f"  E2 Conv3D-SER: UAR={met3d['uar']:.4f}, F1={met3d['macro_f1']:.4f}", flush=True)

    # T4 (4-class shared subset) comparison on CREMA-D test set
    t4_mask_te = np.isin(y_te, [0, 1, 2, 3])
    t4_metrics = compute_classification_metrics(
        y_te[t4_mask_te], softmax_np(best_resnet_te_logits[t4_mask_te, :4]), EMOTIONS_4
    )

    # ------------------------------------------------------------------
    # E3 & E4: Optimizer & Regularization Ablations (on MLP + CNN-Basic)
    # ------------------------------------------------------------------
    print("Running E3 & E4: Optimizer & Regularization ablations...", flush=True)
    e3_opt = {}
    for opt_name in ["sgd", "momentum", "nesterov", "adamw"]:
        m = MlpBaseline(64, 6)
        m, hist = train_classifier(m, func_n[tr], y_tr, func_n[va], y_va, epochs=22, lr=2e-3, optimizer_name=opt_name, seed=42)
        with torch.no_grad():
            lg, _ = predict_batched(m, torch.from_numpy(func_n[te]))
        met = compute_classification_metrics(y_te, softmax_np(lg.numpy()), EMOTIONS_6)
        met["train_loss_curve"] = hist["train_loss"]
        e3_opt[opt_name.upper()] = met

    e4_reg = {}
    for reg_name, wd, ls, aug_flag in [
        ("Baseline (wd=0, ls=0)", 0.0, 0.0, False),
        ("+WeightDecay (1e-3)", 1e-3, 0.0, False),
        ("+LabelSmoothing (0.08)", 1e-4, 0.08, False),
        ("+SpecAugment + WD", 1e-4, 0.05, True),
    ]:
        m = CnnBasic(1, 6)
        m, _ = train_classifier(
            m, x_lm_tr, y_tr, x_lm_va, y_va, epochs=8, lr=2.2e-3,
            weight_decay=wd, label_smoothing=ls, augment=aug_flag, seed=42
        )
        with torch.no_grad():
            lg, _ = predict_batched(m, torch.from_numpy(x_lm_te))
        e4_reg[reg_name] = compute_classification_metrics(y_te, softmax_np(lg.numpy()), EMOTIONS_6)

    # ------------------------------------------------------------------
    # E12: Temperature Scaling Calibration on ResNet-SER + SpecAug
    # ------------------------------------------------------------------
    temperature = fit_temperature_scaling(best_resnet_va_logits, y_va)
    cal_before = compute_classification_metrics(y_te, softmax_np(best_resnet_te_logits, 1.0), EMOTIONS_6)
    cal_after = compute_classification_metrics(y_te, softmax_np(best_resnet_te_logits, temperature), EMOTIONS_6)
    print(f"  Calibration T*={temperature:.3f}: ECE {cal_before['ece']:.4f} -> {cal_after['ece']:.4f}", flush=True)

    # ------------------------------------------------------------------
    # E14: Generative Augmentation (CVAE + cGAN) & Latent-Space Analysis
    # ------------------------------------------------------------------
    print("Running E14: CVAE & cGAN spectrogram augmentation + latent space analysis...", flush=True)
    cvae = SpecCVAE(latent_dim=16, num_classes=6)
    cvae_opt = torch.optim.Adam(cvae.parameters(), lr=2e-3)
    rng_gen = np.random.default_rng(42)
    gen_sub = rng_gen.choice(len(x_lm_tr), size=min(1200, len(x_lm_tr)), replace=False)
    x_tr_t = torch.from_numpy(x_lm_tr[gen_sub])
    y_tr_oh = F.one_hot(torch.from_numpy(y_tr[gen_sub]), num_classes=6).float()
    cvae_dl = DataLoader(TensorDataset(x_tr_t, y_tr_oh), batch_size=128, shuffle=True)
    cvae_losses = []
    for _ in range(8):
        ep_l = 0.0
        for xb, yb in cvae_dl:
            cvae_opt.zero_grad()
            rec, mu, lv = cvae(xb, yb)
            recon_l = F.mse_loss(rec, xb)
            kl = -0.5 * torch.mean(1.0 + lv - mu.pow(2) - lv.exp())
            loss = recon_l + 0.015 * kl
            loss.backward()
            cvae_opt.step()
            ep_l += float(recon_l.item()) * len(xb)
        cvae_losses.append(round(ep_l / len(x_tr_t), 4))
    cvae.eval()

    # Compact cGAN training
    gen = SpecCGANGenerator(noise_dim=32, num_classes=6)
    disc = SpecCGANDiscriminator(num_classes=6)
    g_opt = torch.optim.Adam(gen.parameters(), lr=3e-4, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(disc.parameters(), lr=3e-4, betas=(0.5, 0.999))
    for _ in range(6):
        for xb, yb in cvae_dl:
            bs = len(xb)
            z = torch.randn(bs, 32)
            fake = gen(z, yb).detach()
            d_opt.zero_grad()
            d_real = disc(xb, yb)
            d_fake = disc(fake, yb)
            d_loss = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
            d_loss.backward()
            d_opt.step()

            g_opt.zero_grad()
            z = torch.randn(bs, 32)
            fake2 = gen(z, yb)
            g_loss = -disc(fake2, yb).mean()
            g_loss.backward()
            g_opt.step()
    gen.eval()

    # Evaluate generated spectrogram quality & label consistency using independent ResNet-SER
    with torch.no_grad():
        syn_per_cls = 150
        y_syn = torch.arange(6).repeat_interleave(syn_per_cls)
        y_syn_oh = F.one_hot(y_syn, 6).float()
        cvae_imgs = cvae.decode(torch.randn(len(y_syn), 16), y_syn_oh)
        cgan_imgs = gen(torch.randn(len(y_syn), 32), y_syn_oh)

        lg_cvae_eval, emb_cvae = predict_batched(best_resnet, cvae_imgs)
        lg_cgan_eval, emb_cgan = predict_batched(best_resnet, cgan_imgs)
        cvae_cons = float((lg_cvae_eval.argmax(1) == y_syn).float().mean().item())
        cgan_cons = float((lg_cgan_eval.argmax(1) == y_syn).float().mean().item())

        _, emb_real_te = predict_batched(best_resnet, torch.from_numpy(x_lm_te))
        # PCA 2D projection of real test embeddings and CVAE latent vectors for visualization
        pca_emb = PCA(n_components=2, random_state=42).fit_transform(emb_real_te.numpy())
        mu_te, _ = cvae.encode(torch.from_numpy(x_lm_te), F.one_hot(torch.from_numpy(y_te), 6).float())
        pca_cvae = PCA(n_components=2, random_state=42).fit_transform(mu_te.numpy())

    # Downstream test: adding CVAE synthetic samples to ResNet-SER training
    x_aug_tr = np.concatenate([x_lm_tr, cvae_imgs.numpy()], axis=0)
    y_aug_tr = np.concatenate([y_tr, y_syn.numpy()], axis=0)
    m_cvae_aug = ResNetSer(1, 6, 128)
    m_cvae_aug, _ = train_classifier(m_cvae_aug, x_aug_tr, y_aug_tr, x_lm_va, y_va, epochs=10, lr=2.2e-3, augment=True, seed=42)
    with torch.no_grad():
        lg_cvae_aug, _ = predict_batched(m_cvae_aug, torch.from_numpy(x_lm_te))
    met_cvae_aug = compute_classification_metrics(y_te, softmax_np(lg_cvae_aug.numpy(), temperature), EMOTIONS_6)

    # ------------------------------------------------------------------
    # E9, E10 & E13: Subgroup Fairness, Cross-Corpus (RAVDESS), Noise/Codec Robustness & Latency
    # ------------------------------------------------------------------
    print("Running E9 Fairness, E10 Cross-Corpus (RAVDESS), & E13 Robustness/Latency...", flush=True)
    te_probs = softmax_np(best_resnet_te_logits, temperature)
    fairness = {}
    for col_name, col_arr in [
        ("Sex", crema["sex"][te]),
        ("Age", crema["age_bucket"][te]),
        ("Race", crema["race"][te]),
    ]:
        fairness[col_name] = {}
        for val in sorted(set(col_arr.tolist())):
            mask = col_arr == val
            if int(mask.sum()) >= 15:
                sub = compute_classification_metrics(y_te[mask], te_probs[mask], EMOTIONS_6)
                fairness[col_name][val] = {"n": int(mask.sum()), "uar": sub["uar"], "macro_f1": sub["macro_f1"]}

    # Cross-corpus zero-shot evaluation on RAVDESS (6 shared classes and 4 core classes)
    rav_valid6 = ravdess["labels6"] >= 0
    rav_lm_n = ((ravdess["log_mel"][rav_valid6] - lm_mu) / lm_sd)[:, None, :, :].astype(np.float32)
    rav_y6 = ravdess["labels6"][rav_valid6].astype(np.int64)
    with torch.no_grad():
        rav_lg, _ = predict_batched(best_resnet, torch.from_numpy(rav_lm_n))
    rav_probs6 = softmax_np(rav_lg.numpy(), temperature)
    cross_t6 = compute_classification_metrics(rav_y6, rav_probs6, EMOTIONS_6)

    rav_mask4 = np.isin(rav_y6, [0, 1, 2, 3])
    cross_t4 = compute_classification_metrics(rav_y6[rav_mask4], softmax_np(rav_lg.numpy()[rav_mask4, :4], temperature), EMOTIONS_4)

    # Noise & telephone-bandpass robustness (in log-Mel space: additive noise floor + high-freq attenuation)
    robustness = {}
    for cond, snr_sigma, hi_cut in [
        ("Clean (Studio 16kHz)", 0.0, False),
        ("SNR 20 dB", 0.15, False),
        ("SNR 10 dB", 0.45, False),
        ("SNR 5 dB", 0.80, False),
        ("SNR 0 dB", 1.25, False),
        ("8kHz Telephony Bandlimit", 0.15, True),
    ]:
        x_pert = x_lm_te.copy()
        if hi_cut:
            # Attenuate Mel bins > 44 (~3.4 kHz telephony upper bandlimit)
            x_pert[:, :, 44:, :] = x_pert[:, :, 44:, :] - 2.5
        if snr_sigma > 0.0:
            rng_p = np.random.default_rng(42)
            x_pert = x_pert + rng_p.normal(0.0, snr_sigma, size=x_pert.shape).astype(np.float32)
        with torch.no_grad():
            p_lg, _ = predict_batched(best_resnet, torch.from_numpy(x_pert))
        met_r = compute_classification_metrics(y_te, softmax_np(p_lg.numpy(), temperature), EMOTIONS_6)
        robustness[cond] = {"uar": met_r["uar"], "macro_f1": met_r["macro_f1"], "accuracy": met_r["accuracy"]}

    # CPU inference latency & RTF benchmark
    best_resnet.eval()
    sample_x = torch.from_numpy(x_lm_te[:1])
    for _ in range(10):
        with torch.no_grad():
            best_resnet(sample_x)
    lat_ms = []
    with torch.no_grad():
        for i in range(100):
            s_t = time.perf_counter()
            best_resnet(torch.from_numpy(x_lm_te[i : i + 1]))
            lat_ms.append((time.perf_counter() - s_t) * 1000.0)
    p50_ms = float(np.percentile(lat_ms, 50))
    p95_ms = float(np.percentile(lat_ms, 95))
    rtf = (p50_ms / 1000.0) / 2.572

    # ------------------------------------------------------------------
    # Cross-fitted out-of-fold embeddings for Stage C Context Models (NO LEAKAGE!)
    # ------------------------------------------------------------------
    print("Generating 2-fold speaker-disjoint cross-fitted ResNet embeddings for Stage C...", flush=True)
    tr_actors = sorted(set(crema["actor_ids"][tr].tolist()))
    actor_to_inner = {a: i % 2 for i, a in enumerate(tr_actors)}
    inner_fold = np.array([actor_to_inner[a] for a in crema["actor_ids"][tr]])
    oof_emb = np.zeros((int(tr.sum()), 128), dtype=np.float32)
    oof_logits = np.zeros((int(tr.sum()), 6), dtype=np.float32)

    for f_id in range(2):
        in_tr = inner_fold != f_id
        in_ho = inner_fold == f_id
        m_fold = ResNetSer(1, 6, 128)
        m_fold, _ = train_classifier(
            m_fold, x_lm_tr[in_tr], y_tr[in_tr], x_lm_va, y_va,
            epochs=8, lr=2.2e-3, augment=True, seed=100 + f_id
        )
        with torch.no_grad():
            ho_lg, ho_em = predict_batched(m_fold, torch.from_numpy(x_lm_tr[in_ho]))
        oof_logits[in_ho] = ho_lg.numpy()
        oof_emb[in_ho] = ho_em.numpy()

    with torch.no_grad():
        va_lg_full, va_emb_full = predict_batched(best_resnet, torch.from_numpy(x_lm_va))
        te_lg_full, te_emb_full = predict_batched(best_resnet, torch.from_numpy(x_lm_te))

    full_emb = np.zeros((len(crema["splits"]), 128), dtype=np.float32)
    full_logits = np.zeros((len(crema["splits"]), 6), dtype=np.float32)
    full_emb[tr] = oof_emb
    full_logits[tr] = oof_logits
    full_emb[va] = va_emb_full.numpy()
    full_logits[va] = va_lg_full.numpy()
    full_emb[te] = te_emb_full.numpy()
    full_logits[te] = te_lg_full.numpy()

    # Also RAVDESS full embeddings
    rav_full_lm = ((ravdess["log_mel"] - lm_mu) / lm_sd)[:, None, :, :].astype(np.float32)
    with torch.no_grad():
        rav_full_lg, rav_full_em = predict_batched(best_resnet, torch.from_numpy(rav_full_lm))

    # Extract 1 representative Grad-CAM & conv feature map for plotting
    sample_idx = int(np.where(y_te == 3)[0][0])  # Angry test utterance
    gradcam_data = compute_gradcam(best_resnet, x_lm_te[sample_idx : sample_idx + 1], target_class=3)

    return {
        "e1_features": e1_results,
        "e2_architectures": e2_results,
        "e2_t4_resnet": t4_metrics,
        "e3_optimizers": e3_opt,
        "e4_regularization": e4_reg,
        "e12_calibration": {
            "temperature": round(temperature, 4),
            "before": cal_before,
            "after": cal_after,
        },
        "e14_generative": {
            "cvae_recon_mse_curve": cvae_losses,
            "cvae_label_consistency": round(cvae_cons, 4),
            "cgan_label_consistency": round(cgan_cons, 4),
            "resnet_plus_cvae_aug": met_cvae_aug,
            "pca_resnet_emb": pca_emb[:400].round(3).tolist(),
            "pca_cvae_latent": pca_cvae[:400].round(3).tolist(),
            "pca_labels": y_te[:400].tolist(),
        },
        "e9_fairness": fairness,
        "e10_cross_corpus_ravdess": {"t6_shared": cross_t6, "t4_core": cross_t4},
        "e13_robustness": robustness,
        "e13_efficiency_python": {"p50_ms": round(p50_ms, 3), "p95_ms": round(p95_ms, 3), "rtf": round(rtf, 5)},
        "norm_stats": {"log_mel_mean": lm_mu, "log_mel_std": lm_sd},
        "best_resnet": best_resnet,
        "best_resnet_hist": best_resnet_hist,
        "temperature": temperature,
        "crema_oof_emb": full_emb,
        "crema_oof_logits": full_logits,
        "ravdess_emb": rav_full_em.numpy(),
        "ravdess_logits": rav_full_lg.numpy(),
        "gradcam_sample": gradcam_data,
        "x_lm_te": x_lm_te,
        "y_te": y_te,
        "te_probs": te_probs,
    }


def compute_gradcam(model: ResNetSer, x_np: np.ndarray, target_class: int) -> dict[str, Any]:
    model.eval()
    x = torch.from_numpy(x_np).requires_grad_(True)
    acts = []

    def fwd_hook(_m, _in, out):
        out.retain_grad()
        acts.append(out)

    handle = model.stage4.register_forward_hook(fwd_hook)
    logits, _ = model(x)
    model.zero_grad()
    logits[0, target_class].backward()
    handle.remove()

    fmap = acts[0][0].detach().numpy()      # [128, 8, 16]
    grad = acts[0].grad[0].detach().numpy()  # [128, 8, 16]
    weights = grad.mean(axis=(1, 2), keepdims=True)
    cam = np.maximum(0.0, np.sum(weights * fmap, axis=0))
    cam = cam / max(1e-8, float(cam.max()))
    return {
        "input_spec": x_np[0, 0].round(3).tolist(),
        "conv_fmap0": fmap[0].round(3).tolist(),
        "conv_fmap1": fmap[1].round(3).tolist(),
        "gradcam": cam.round(3).tolist(),
    }
