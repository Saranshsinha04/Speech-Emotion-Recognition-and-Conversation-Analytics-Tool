"""Stage C conversational context experiments (E6/E7/E8), escalation benchmark (E11), and ONNX export."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

from ser_research.data_pipeline import compute_escalation_onset
from ser_research.features import load_and_validate_wav, preprocess_signal
from ser_research.metrics import (
    binary_ranking_metrics,
    cluster_bootstrap_diff,
    compute_classification_metrics,
    jensen_shannon_divergence,
    softmax_np,
)
from ser_research.models import ContextLstm, ContextTransformer

EMOTIONS_6 = ["NEU", "HAP", "SAD", "ANG", "FEA", "DIS"]


def build_dialog_tensors(
    dialogs: list[dict[str, Any]],
    embeddings: np.ndarray,
    acoustic_logits: np.ndarray,
    temperature: float,
) -> dict[str, Any]:
    max_t = max(len(d["turns"]) for d in dialogs)
    n_d = len(dialogs)
    x_seq = np.zeros((n_d, max_t, 130), dtype=np.float32)
    y_seq = np.full((n_d, max_t), -100, dtype=np.int64)
    esc_seq = np.zeros((n_d, max_t), dtype=np.float32)
    mask = np.zeros((n_d, max_t), dtype=bool)
    aco_probs = np.zeros((n_d, max_t, 6), dtype=np.float32)
    agreements = np.zeros((n_d, max_t), dtype=np.float32)
    speakers = np.zeros((n_d, max_t), dtype=np.int64)  # 0=CUSTOMER, 1=AGENT

    for i, dlg in enumerate(dialogs):
        tau = dlg["escalation_onset_customer_turn"]
        cust_turn = 0
        for t, turn in enumerate(dlg["turns"]):
            d_idx = turn["dataset_idx"]
            is_cust = turn["speaker"] == "CUSTOMER"
            spk_oh = np.array([1.0, 0.0], dtype=np.float32) if is_cust else np.array([0.0, 1.0], dtype=np.float32)
            x_seq[i, t, :128] = embeddings[d_idx]
            x_seq[i, t, 128:] = spk_oh
            y_seq[i, t] = turn["label6"]
            mask[i, t] = True
            aco_probs[i, t] = softmax_np(acoustic_logits[d_idx : d_idx + 1], temperature)[0]
            agreements[i, t] = turn["agreement"]
            speakers[i, t] = 0 if is_cust else 1
            if is_cust:
                if tau is not None and cust_turn >= tau:
                    esc_seq[i, t] = 1.0
                cust_turn += 1

    return {
        "x_seq": x_seq,
        "y_seq": y_seq,
        "esc_seq": esc_seq,
        "mask": mask,
        "aco_probs": aco_probs,
        "agreements": agreements,
        "speakers": speakers,
        "dialogs": dialogs,
    }


def fit_hmm_smoother(tr_data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Fits empirical class prior & speaker-conditional turn transition matrix for HMM Viterbi/Forward control."""
    trans = np.ones((6, 6), dtype=np.float64) * 0.5  # Laplace smoothing
    prior = np.ones(6, dtype=np.float64)
    for i in range(len(tr_data["dialogs"])):
        # Separate customer transition sequence
        cust_y = tr_data["y_seq"][i][(tr_data["mask"][i]) & (tr_data["speakers"][i] == 0)]
        if len(cust_y) > 0:
            prior[cust_y[0]] += 1.0
        for a, b in zip(cust_y[:-1], cust_y[1:]):
            trans[a, b] += 1.0
    prior /= prior.sum()
    trans /= trans.sum(axis=1, keepdims=True)
    return np.log(prior + 1e-12), np.log(trans + 1e-12)


def apply_hmm_forward(probs_seq: np.ndarray, mask: np.ndarray, speakers: np.ndarray, log_prior: np.ndarray, log_trans: np.ndarray) -> np.ndarray:
    """Applies HMM forward filtering independently on Customer & Agent turn tracks."""
    out = probs_seq.copy()
    for i in range(len(probs_seq)):
        for spk in (0, 1):
            idxs = np.where(mask[i] & (speakers[i] == spk))[0]
            if len(idxs) == 0:
                continue
            log_alpha = log_prior + np.log(probs_seq[i, idxs[0]] + 1e-9)
            log_alpha -= np.max(log_alpha)
            p = np.exp(log_alpha)
            out[i, idxs[0]] = p / p.sum()
            for step in range(1, len(idxs)):
                t = idxs[step]
                # log sum_prev exp(log_alpha[prev] + log_trans[prev, curr])
                scores = log_alpha[:, None] + log_trans
                mx = np.max(scores, axis=0)
                pred = mx + np.log(np.sum(np.exp(scores - mx[None, :]), axis=0))
                log_alpha = pred + np.log(probs_seq[i, t] + 1e-9)
                log_alpha -= np.max(log_alpha)
                p = np.exp(log_alpha)
                out[i, t] = p / p.sum()
    return out


def sequence_metrics(
    y_seq: np.ndarray,
    probs_seq: np.ndarray,
    mask: np.ndarray,
    agreements: np.ndarray,
) -> dict[str, Any]:
    y_flat = y_seq[mask]
    p_flat = probs_seq[mask]
    base = compute_classification_metrics(y_flat, p_flat, EMOTIONS_6)

    # Sequence-level exact accuracy, normalized edit distance, and JSD of conversation distribution
    exact_matches, edit_sims, jsds = [], [], []
    for i in range(len(y_seq)):
        m = mask[i]
        yt = y_seq[i, m]
        yp = np.argmax(probs_seq[i, m], axis=1)
        exact_matches.append(float(np.mean(yt == yp)))
        # Normalized edit distance (Levenshtein)
        L = len(yt)
        dp = np.zeros((L + 1, L + 1), dtype=np.int32)
        dp[:, 0] = np.arange(L + 1)
        dp[0, :] = np.arange(L + 1)
        for r in range(1, L + 1):
            for c in range(1, L + 1):
                cost = 0 if yt[r - 1] == yp[c - 1] else 1
                dp[r, c] = min(dp[r - 1, c] + 1, dp[r, c - 1] + 1, dp[r - 1, c - 1] + cost)
        edit_sims.append(1.0 - dp[L, L] / max(1, L))

        gold_dist = np.bincount(yt, minlength=6).astype(np.float64) + 1e-6
        pred_dist = np.mean(probs_seq[i, m], axis=0) + 1e-6
        jsds.append(jensen_shannon_divergence(gold_dist, pred_dist))

    # Agreement stratification (Ambiguous <0.50, Medium 0.50..0.75, High >=0.75)
    agr_flat = agreements[mask]
    strat = {}
    for label, cond in [
        ("Low (<0.50)", agr_flat < 0.50),
        ("Medium (0.50-0.75)", (agr_flat >= 0.50) & (agr_flat < 0.75)),
        ("High (>=0.75)", agr_flat >= 0.75),
    ]:
        if int(cond.sum()) >= 20:
            sub = compute_classification_metrics(y_flat[cond], p_flat[cond], EMOTIONS_6)
            strat[label] = {"n": int(cond.sum()), "uar": sub["uar"], "macro_f1": sub["macro_f1"]}

    base["turn_accuracy"] = round(float(np.mean(exact_matches)), 4)
    base["norm_edit_similarity"] = round(float(np.mean(edit_sims)), 4)
    base["conversation_jsd"] = round(float(np.mean(jsds)), 4)
    base["agreement_stratified"] = strat
    return base


def train_context_model(
    model: nn.Module,
    tr_data: dict[str, Any],
    va_data: dict[str, Any],
    epochs: int = 32,
    lr: float = 1e-3,
    seed: int = 42,
    context_limit: int | None = None,
) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    x_tr = torch.from_numpy(tr_data["x_seq"])
    y_tr = torch.from_numpy(tr_data["y_seq"])
    e_tr = torch.from_numpy(tr_data["esc_seq"])
    m_tr = torch.from_numpy(tr_data["mask"])

    x_va = torch.from_numpy(va_data["x_seq"])
    y_va = va_data["y_seq"]
    m_va = va_data["mask"]

    best_state = copy.deepcopy(model.state_dict())
    best_uar = -1.0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(x_tr))
        for i in range(0, len(x_tr), 24):
            idx = perm[i : i + 24]
            xb = x_tr[idx]
            yb = y_tr[idx]
            eb = e_tr[idx]
            mb = m_tr[idx]
            if context_limit is not None and xb.shape[1] > context_limit:
                xb = xb[:, -context_limit:, :]
                yb = yb[:, -context_limit:]
                eb = eb[:, -context_limit:]
                mb = mb[:, -context_limit:]
            opt.zero_grad()
            out = model(xb)
            lg, esc = out[0], out[1]
            cls_loss = F.cross_entropy(lg.reshape(-1, 6), yb.reshape(-1), ignore_index=-100)
            esc_loss = F.binary_cross_entropy(esc.squeeze(-1)[mb], eb[mb])
            (cls_loss + 0.5 * esc_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            va_lg = model(x_va)[0].numpy()
            va_probs = softmax_np(va_lg[m_va])
            va_met = compute_classification_metrics(y_va[m_va], va_probs, EMOTIONS_6)
        if va_met["uar"] > best_uar:
            best_uar = va_met["uar"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    return model


def evaluate_escalation_detectors(
    tr_data: dict[str, Any],
    te_data: dict[str, Any],
    te_ctx_probs: np.ndarray,
    te_ctx_esc_head: np.ndarray,
) -> dict[str, Any]:
    """Evaluates D1..D6 escalation detectors on customer tracks and runs (W, rho, k, m) sensitivity."""
    neg_idx = [2, 3, 5]  # SAD, ANG, DIS

    def extract_customer_tracks(data: dict[str, Any], probs_seq: np.ndarray, esc_head: np.ndarray | None = None):
        tracks = []
        for i, dlg in enumerate(data["dialogs"]):
            idxs = np.where(data["mask"][i] & (data["speakers"][i] == 0))[0]
            s_seq = probs_seq[i, idxs][:, neg_idx].sum(axis=1)
            y_turn_esc = data["esc_seq"][i, idxs]
            tau = dlg["escalation_onset_customer_turn"]
            head_seq = esc_head[i, idxs] if esc_head is not None else None
            tracks.append((s_seq, y_turn_esc, tau, head_seq, dlg["turns"]))
        return tracks

    # Estimate baseline calm mean mu_0 and emission Gaussian params on TRAIN customer tracks
    tr_tracks = extract_customer_tracks(tr_data, tr_data["aco_probs"])
    calm_vals = np.concatenate([s[y < 0.5] for s, y, _, _, _ in tr_tracks if (y < 0.5).any()])
    esc_vals = np.concatenate([s[y > 0.5] for s, y, _, _, _ in tr_tracks if (y > 0.5).any()])
    mu0 = float(np.mean(calm_vals))
    sd0 = max(0.08, float(np.std(calm_vals)))
    mu1 = float(np.mean(esc_vals)) if len(esc_vals) > 0 else 0.72
    sd1 = max(0.08, float(np.std(esc_vals))) if len(esc_vals) > 0 else 0.20

    te_tracks_ctx = extract_customer_tracks(te_data, te_ctx_probs, te_ctx_esc_head)
    te_tracks_aco = extract_customer_tracks(te_data, te_data["aco_probs"], None)

    def run_detector(tracks, kind: str) -> dict[str, Any]:
        y_all, sc_all = [], []
        delays = []
        tp_ev, fp_ev, fn_ev = 0, 0, 0
        non_esc_fa = 0
        non_esc_cnt = 0

        for s_seq, y_esc, tau, head_seq, _ in tracks:
            T = len(s_seq)
            scores = np.zeros(T, dtype=np.float64)
            if kind == "D1_Threshold":
                scores = s_seq
                thresh = 0.55
            elif kind == "D2_SlidingWindow":
                for t in range(T):
                    scores[t] = float(np.mean(s_seq[max(0, t - 2) : t + 1]))
                thresh = 0.50
            elif kind == "D3_EWMA":
                z = mu0
                for t in range(T):
                    z = 0.45 * s_seq[t] + 0.55 * z
                    scores[t] = z
                thresh = 0.50
            elif kind == "D4_CUSUM":
                g = 0.0
                for t in range(T):
                    g = max(0.0, g + s_seq[t] - mu0 - 0.10)
                    scores[t] = g
                thresh = 0.55
            elif kind == "D5_HMM_Filter":
                # 2-state causal HMM forward filter: state 0=calm, state 1=escalated
                p0, p1 = 0.90, 0.10
                for t in range(T):
                    p0_pred = 0.88 * p0 + 0.05 * p1
                    p1_pred = 0.12 * p0 + 0.95 * p1
                    l0 = np.exp(-0.5 * ((s_seq[t] - mu0) / sd0) ** 2) / sd0
                    l1 = np.exp(-0.5 * ((s_seq[t] - mu1) / sd1) ** 2) / sd1
                    u0, u1 = p0_pred * l0, p1_pred * l1
                    denom = max(1e-12, u0 + u1)
                    p0, p1 = u0 / denom, u1 / denom
                    scores[t] = p1
                thresh = 0.55
            elif kind == "D6_TransformerHead":
                scores = head_seq
                thresh = 0.45

            y_all.append(y_esc)
            sc_all.append(scores)

            alarm_t = None
            for t in range(T):
                if scores[t] >= thresh:
                    alarm_t = t
                    break
            if tau is not None:
                if alarm_t is not None and alarm_t <= tau + 3 and alarm_t >= max(0, tau - 2):
                    tp_ev += 1
                    delays.append(alarm_t - tau)
                elif alarm_t is not None and alarm_t > tau + 3:
                    tp_ev += 1
                    delays.append(alarm_t - tau)
                else:
                    fn_ev += 1
            else:
                non_esc_cnt += 1
                if alarm_t is not None:
                    fp_ev += 1
                    non_esc_fa += 1

        yf = np.concatenate(y_all)
        sf = np.concatenate(sc_all)
        rank = binary_ranking_metrics(yf, sf)
        prec = tp_ev / max(1, tp_ev + fp_ev)
        rec = tp_ev / max(1, tp_ev + fn_ev)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        return {
            **rank,
            "event_precision": round(float(prec), 4),
            "event_recall": round(float(rec), 4),
            "event_f1": round(float(f1), 4),
            "median_delay_turns": round(float(np.median(delays)) if delays else 0.0, 2),
            "false_alarm_rate_per_track": round(non_esc_fa / max(1, non_esc_cnt), 4),
        }

    detectors = {}
    detectors["D1_Threshold (Audio-Only CNN)"] = run_detector(te_tracks_aco, "D1_Threshold")
    detectors["D4_CUSUM (Audio-Only CNN)"] = run_detector(te_tracks_aco, "D4_CUSUM")
    for k_name in ["D1_Threshold", "D2_SlidingWindow", "D3_EWMA", "D4_CUSUM", "D5_HMM_Filter", "D6_TransformerHead"]:
        detectors[f"{k_name} (Context)"] = run_detector(te_tracks_ctx, k_name)

    # Sensitivity grid across operational definition (W, rho, k, m)
    sensitivity = []
    for w_p, rho_p, k_p, m_p in [(2, 0.5, 2, 2), (3, 1.0 / 3.0, 3, 2), (3, 0.34, 4, 3)]:
        pos_dlgs = 0
        for dlg in te_data["dialogs"]:
            c_emos = [t["emotion"] for t in dlg["turns"] if t["speaker"] == "CUSTOMER"]
            c_ints = [t["intensity"] for t in dlg["turns"] if t["speaker"] == "CUSTOMER"]
            if compute_escalation_onset(c_emos, c_ints, w=w_p, rho=rho_p, k=k_p, m=m_p) is not None:
                pos_dlgs += 1
        sensitivity.append({
            "params": f"W={w_p}, rho={rho_p:.2f}, k={k_p}, m={m_p}",
            "positive_test_dialogs": pos_dlgs,
            "prevalence": round(pos_dlgs / max(1, len(te_data["dialogs"])), 4),
        })

    return {
        "detectors": detectors,
        "hmm_params": {"mu0": round(mu0, 4), "sd0": round(sd0, 4), "mu1": round(mu1, 4), "sd1": round(sd1, 4)},
        "sensitivity_grid": sensitivity,
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_context_and_export(
    crema: dict[str, np.ndarray],
    ravdess: dict[str, np.ndarray],
    crema_dialogs: list[dict[str, Any]],
    ravdess_dialogs: list[dict[str, Any]],
    utt_bundle: dict[str, Any],
    models_dir: Path,
) -> dict[str, Any]:
    print("Running E6/E7/E8: Stage C Conversational Context Models (HMM, BiLSTM, Causal/Bidir Transformer)...", flush=True)
    torch.set_num_threads(6)
    models_dir.mkdir(parents=True, exist_ok=True)
    temp = utt_bundle["temperature"]
    emb_c = utt_bundle["crema_oof_emb"]
    lg_c = utt_bundle["crema_oof_logits"]

    tr_dlgs = [d for d in crema_dialogs if d["split"] == "train"]
    va_dlgs = [d for d in crema_dialogs if d["split"] == "val"]
    te_dlgs = [d for d in crema_dialogs if d["split"] == "test"]

    tr_data = build_dialog_tensors(tr_dlgs, emb_c, lg_c, temp)
    va_data = build_dialog_tensors(va_dlgs, emb_c, lg_c, temp)
    te_data = build_dialog_tensors(te_dlgs, emb_c, lg_c, temp)

    rav_data = build_dialog_tensors(ravdess_dialogs, utt_bundle["ravdess_emb"], utt_bundle["ravdess_logits"], temp)

    # 1. No-Context ResNet-SER baseline on CREMA-Dialog test
    y_te_seq = te_data["y_seq"]
    m_te = te_data["mask"]
    agr_te = te_data["agreements"]
    no_ctx_metrics = sequence_metrics(y_te_seq, te_data["aco_probs"], m_te, agr_te)

    # 2. HMM Smoothing non-neural temporal control
    log_prior, log_trans = fit_hmm_smoother(tr_data)
    hmm_probs = apply_hmm_forward(te_data["aco_probs"], m_te, te_data["speakers"], log_prior, log_trans)
    hmm_metrics = sequence_metrics(y_te_seq, hmm_probs, m_te, agr_te)

    # 3. Neural sequence models (3 seeds for Causal & Bidirectional Transformer)
    ctx_results = {
        "No-Context (ResNet-SER)": no_ctx_metrics,
        "HMM-Forward (Control)": hmm_metrics,
    }

    best_causal_tf = None
    best_causal_probs = None
    best_causal_esc = None
    best_causal_attn = None

    for name, builder, seeds in [
        ("ResNet + Causal LSTM", lambda: ContextLstm(130, 96, 6, bidirectional=False), [42]),
        ("ResNet + BiLSTM", lambda: ContextLstm(130, 96, 6, bidirectional=True), [42, 43]),
        ("ResNet + Causal Transformer", lambda: ContextTransformer(130, 128, 4, 2, 6, causal=True), [42, 43, 44]),
        ("ResNet + Bidir Transformer", lambda: ContextTransformer(130, 128, 4, 2, 6, causal=False), [42, 43, 44]),
    ]:
        seed_uars = []
        last_met = None
        for sd in seeds:
            m = train_context_model(builder(), tr_data, va_data, epochs=32, lr=1e-3, seed=sd)
            with torch.no_grad():
                out = m(torch.from_numpy(te_data["x_seq"]))
                lg = out[0].numpy()
                esc = out[1].squeeze(-1).numpy()
                probs = softmax_np(lg.reshape(-1, 6)).reshape(lg.shape)
            met = sequence_metrics(y_te_seq, probs, m_te, agr_te)
            seed_uars.append(met["uar"])
            last_met = met
            if name == "ResNet + Causal Transformer" and sd == 42:
                best_causal_tf = m
                best_causal_probs = probs
                best_causal_esc = esc
                best_causal_attn = out[2][0].numpy()  # [T, T] sample attention map
        last_met["uar_mean"] = round(float(np.mean(seed_uars)), 4)
        last_met["uar_std"] = round(float(np.std(seed_uars)), 4)
        ctx_results[name] = last_met
        print(f"  {name}: UAR={last_met['uar_mean']:.4f}±{last_met['uar_std']:.4f}, SeqAcc={last_met['turn_accuracy']:.4f}, JSD={last_met['conversation_jsd']:.4f}")

    # Cluster bootstrap significance test (resampling dialogs) for Hypothesis H2
    dialog_ids = np.repeat(np.arange(len(te_dlgs))[:, None], m_te.shape[1], axis=1)[m_te]
    y_flat = y_te_seq[m_te]
    pred_no_ctx = np.argmax(te_data["aco_probs"][m_te], axis=1)
    pred_hmm = np.argmax(hmm_probs[m_te], axis=1)
    pred_tf = np.argmax(best_causal_probs[m_te], axis=1)

    from sklearn.metrics import recall_score

    uar_fn = lambda yt, yp: float(recall_score(yt, yp, average="macro", zero_division=0))
    boot_vs_nocontext = cluster_bootstrap_diff(y_flat, pred_no_ctx, pred_tf, dialog_ids, uar_fn)
    boot_vs_hmm = cluster_bootstrap_diff(y_flat, pred_hmm, pred_tf, dialog_ids, uar_fn)

    # E8: Context window ablation (K in 1, 3, 5, 8, full)
    window_ablation = {}
    for k_win in [1, 3, 5, 8, None]:
        m_w = train_context_model(
            ContextTransformer(130, 128, 4, 2, 6, causal=True),
            tr_data, va_data, epochs=24, lr=1e-3, seed=42, context_limit=k_win
        )
        with torch.no_grad():
            x_in = torch.from_numpy(te_data["x_seq"][:, -k_win:, :] if k_win else te_data["x_seq"])
            y_sub = y_te_seq[:, -k_win:] if k_win else y_te_seq
            m_sub = m_te[:, -k_win:] if k_win else m_te
            a_sub = agr_te[:, -k_win:] if k_win else agr_te
            lg_w = m_w(x_in)[0].numpy()
            pr_w = softmax_np(lg_w.reshape(-1, 6)).reshape(lg_w.shape)
        met_w = sequence_metrics(y_sub, pr_w, m_sub, a_sub)
        window_ablation[f"K={k_win if k_win else 'Full'}"] = {"uar": met_w["uar"], "macro_f1": met_w["macro_f1"]}

    # Zero-shot RAVDESS-Dialog evaluation
    with torch.no_grad():
        rav_out = best_causal_tf(torch.from_numpy(rav_data["x_seq"]))
        rav_ctx_probs = softmax_np(rav_out[0].numpy().reshape(-1, 6)).reshape(rav_out[0].shape)
    rav_noc_m = sequence_metrics(rav_data["y_seq"], rav_data["aco_probs"], rav_data["mask"], rav_data["agreements"])
    rav_ctx_m = sequence_metrics(rav_data["y_seq"], rav_ctx_probs, rav_data["mask"], rav_data["agreements"])

    # E11: Escalation detectors
    esc_results = evaluate_escalation_detectors(tr_data, te_data, best_causal_probs, best_causal_esc)
    print(f"  Escalation D5 HMM AUPRC={esc_results['detectors']['D5_HMM_Filter (Context)']['auprc']:.4f}, "
          f"D6 Transformer AUPRC={esc_results['detectors']['D6_TransformerHead (Context)']['auprc']:.4f}")

    # ------------------------------------------------------------------
    # Export ONNX models + verify ORT numerical parity + save parity_fixtures.json
    # ------------------------------------------------------------------
    print("Exporting ONNX models (opset 17) & parity fixtures for Java...")
    resnet = utt_bundle["best_resnet"]
    resnet.eval()
    best_causal_tf.eval()

    resnet_onnx = models_dir / "resnet_ser.onnx"
    tf_onnx = models_dir / "context_transformer.onnx"

    dummy_spec = torch.randn(2, 1, 64, 128)
    torch.onnx.export(
        resnet,
        (dummy_spec,),
        str(resnet_onnx),
        input_names=["log_mel"],
        output_names=["logits", "embedding"],
        dynamic_axes={"log_mel": {0: "batch"}, "logits": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    dummy_seq = torch.randn(1, 12, 130)
    torch.onnx.export(
        best_causal_tf,
        (dummy_seq,),
        str(tf_onnx),
        input_names=["seq_features"],
        output_names=["context_logits", "escalation_prob", "attention_weights"],
        dynamic_axes={
            "seq_features": {0: "batch", 1: "seq_len"},
            "context_logits": {0: "batch", 1: "seq_len"},
            "escalation_prob": {0: "batch", 1: "seq_len"},
            "attention_weights": {0: "batch", 1: "seq_len"},
        },
        opset_version=17,
        dynamo=False,
    )

    # Verify PyTorch <-> ONNX Runtime parity
    sess_res = ort.InferenceSession(str(resnet_onnx), providers=["CPUExecutionProvider"])
    sess_tf = ort.InferenceSession(str(tf_onnx), providers=["CPUExecutionProvider"])

    test_spec = utt_bundle["x_lm_te"][:10]
    with torch.no_grad():
        pt_lg, pt_em = resnet(torch.from_numpy(test_spec))
    ort_lg, ort_em = sess_res.run(None, {"log_mel": test_spec})
    max_res_diff = float(np.max(np.abs(pt_lg.numpy() - ort_lg)))

    seq_sample = te_data["x_seq"][:1, :10, :]
    with torch.no_grad():
        pt_clg, pt_esc, _ = best_causal_tf(torch.from_numpy(seq_sample))
    ort_clg, ort_esc, _ = sess_tf.run(None, {"seq_features": seq_sample})
    max_tf_diff = float(np.max(np.abs(pt_clg.numpy() - ort_clg)))
    print(f"  ONNX parity check: ResNet max_abs_diff={max_res_diff:.2e}, Transformer max_abs_diff={max_tf_diff:.2e}")

    # Build 10-utterance parity fixture (including preprocessed raw 16kHz waveforms for end-to-end Java audio parity!)
    te_indices = np.where(crema["splits"] == "test")[0][:10]
    # Also load raw paths from crema_d AudioWAV matching utterance_ids
    raw_dir = Path(__file__).resolve().parents[3] / "data" / "raw" / "crema_d" / "AudioWAV"
    utt_fixtures = []
    for pos_i, ds_idx in enumerate(te_indices):
        uid = str(crema["utterance_ids"][ds_idx])
        wav_p = raw_dir / f"{uid}.wav"
        y16, _ = load_and_validate_wav(wav_p)
        fixed_y, _, _ = preprocess_signal(y16)
        spec_in = test_spec[pos_i : pos_i + 1]
        lg_i, em_i = sess_res.run(None, {"log_mel": spec_in})
        pr_i = softmax_np(lg_i, temp)[0]
        utt_fixtures.append({
            "utterance_id": uid,
            "true_emotion": EMOTIONS_6[int(crema["labels6"][ds_idx])],
            "pcm_sample_prefix": fixed_y[:16].round(6).tolist(),
            "log_mel_mean_val": round(float(crema["log_mel"][ds_idx].mean()), 6),
            "log_mel_norm": spec_in[0, 0].round(6).tolist(),
            "expected_logits": lg_i[0].round(6).tolist(),
            "expected_probs": pr_i.round(6).tolist(),
            "expected_embedding": em_i[0].round(6).tolist(),
        })

    parity_payload = {
        "resnet_max_ort_diff": max_res_diff,
        "transformer_max_ort_diff": max_tf_diff,
        "utterances": utt_fixtures,
        "sequence_test": {
            "seq_features": seq_sample[0].round(6).tolist(),
            "expected_context_logits": ort_clg[0].round(6).tolist(),
            "expected_escalation_prob": ort_esc[0, :, 0].round(6).tolist(),
        },
    }
    (models_dir / "parity_fixtures.json").write_text(json.dumps(parity_payload))

    meta_payload = {
        "emotions": EMOTIONS_6,
        "temperature": round(float(temp), 6),
        "log_mel_mean": round(float(utt_bundle["norm_stats"]["log_mel_mean"]), 6),
        "log_mel_std": round(float(utt_bundle["norm_stats"]["log_mel_std"]), 6),
        "hmm_escalation_params": esc_results["hmm_params"],
        "cusum_params": {"mu0": esc_results["hmm_params"]["mu0"], "kappa": 0.10, "threshold": 0.55},
        "hmm_smoothing_log_prior": log_prior.round(6).tolist(),
        "hmm_smoothing_log_trans": log_trans.round(6).tolist(),
    }
    (models_dir / "model_metadata.json").write_text(json.dumps(meta_payload, indent=2))

    registry_payload = {
        "models": [
            {
                "name": "resnet_ser",
                "version": "1.0.0",
                "file": "resnet_ser.onnx",
                "sha256": sha256_file(resnet_onnx),
                "opset": 17,
                "dataset": "CREMA-D (speaker-independent 64/13/14)",
                "classes": EMOTIONS_6,
            },
            {
                "name": "context_transformer",
                "version": "1.0.0",
                "file": "context_transformer.onnx",
                "sha256": sha256_file(tf_onnx),
                "opset": 17,
                "dataset": "CREMA-Dialog (cross-fitted ResNet embeddings)",
                "classes": EMOTIONS_6,
            },
        ]
    }
    (models_dir / "registry.json").write_text(json.dumps(registry_payload, indent=2))

    return {
        "e6_e7_context": ctx_results,
        "h2_bootstrap_vs_nocontext": boot_vs_nocontext,
        "h2_bootstrap_vs_hmm": boot_vs_hmm,
        "e8_context_windows": window_ablation,
        "ravdess_dialog_zeroshot": {"no_context": rav_noc_m, "causal_transformer": rav_ctx_m},
        "e11_escalation": esc_results,
        "sample_attention": best_causal_attn.round(4).tolist(),
        "onnx_parity": {"resnet_max_diff": max_res_diff, "transformer_max_diff": max_tf_diff},
    }
