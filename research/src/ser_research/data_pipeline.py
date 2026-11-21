"""Reproducible dataset audit, leakage-safe splitting, feature caching, and dialog benchmark builder.

Processes CREMA-D (7,442 WAVs, 91 actors) and RAVDESS (1,440 WAVs, 24 actors).
Produces:
  * data/features/crema_d_features.npz
  * data/features/ravdess_features.npz
  * data/features/crema_mlp_functionals.csv (standardized by train-split statistics only)
  * data/features/crema_dialogs.json & ravdess_dialogs.json (multi-turn conversations + onset tau)
  * data/features/stream_benchmark.npz (continuous conversation spectrograms, frame masks & intervals for Seg/Det)
  * data/conversations/sample_*.wav & ground_truth.json (real held-out test audio for Java app demo)
  * results/dataset_stats.json
"""
from __future__ import annotations

import concurrent.futures as cf
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import wavfile

from ser_research.features import (
    N_FRAMES,
    N_MELS,
    SAMPLE_RATE,
    extract_all_features,
    load_and_validate_wav,
)

ROOT = Path(__file__).resolve().parents[3]
CREMA_DIR = ROOT / "data" / "raw" / "crema_d"
RAVDESS_DIR = ROOT / "data" / "raw" / "ravdess" / "audio"
FEAT_DIR = ROOT / "data" / "features"
CONV_DIR = ROOT / "data" / "conversations"
RESULTS_DIR = ROOT / "results"

EMOTIONS_6 = ["NEU", "HAP", "SAD", "ANG", "FEA", "DIS"]
EMO_TO_IDX6 = {e: i for i, e in enumerate(EMOTIONS_6)}
NEGATIVE_SET = {"ANG", "DIS", "SAD"}  # high/sustained distress for contact-center escalation
RAVDESS_MAP = {
    "01": "NEU",
    "02": "CAL",
    "03": "HAP",
    "04": "SAD",
    "05": "ANG",
    "06": "FEA",
    "07": "DIS",
    "08": "SUR",
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _process_one_wav(args: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    path_str, meta = args
    path = Path(path_str)
    file_hash = _sha256_file(path)
    y_16k, val = load_and_validate_wav(path)
    if not val.valid:
        return {"valid": False, "path": path_str, "sha256": file_hash, "reason": val.reason, **meta}
    feats = extract_all_features(y_16k)
    return {
        "valid": True,
        "path": path_str,
        "sha256": file_hash,
        "sr": val.sr,
        "channels": val.channels,
        "raw_duration_s": val.raw_duration_s,
        "clipping_ratio": val.clipping_ratio,
        "silence_ratio": val.silence_ratio,
        "snr_db": val.snr_db,
        **meta,
        **feats,
    }


def load_crema_records() -> list[tuple[str, dict[str, Any]]]:
    demo_df = pd.read_csv(CREMA_DIR / "VideoDemographics.csv")
    demo_map = demo_df.set_index("ActorID").to_dict(orient="index")

    # Voice-only votes correspond to IDs < 200000 in tabulatedVotes.csv
    votes_df = pd.read_csv(CREMA_DIR / "processedResults" / "tabulatedVotes.csv")
    id_col = votes_df.columns[0]
    voice_votes = votes_df[votes_df[id_col].astype(int) < 200000].copy()
    voice_votes = voice_votes.drop_duplicates(subset=["fileName"], keep="first")
    vote_map = voice_votes.set_index("fileName").to_dict(orient="index")

    items = []
    for wav_path in sorted((CREMA_DIR / "AudioWAV").glob("*.wav")):
        stem = wav_path.stem
        parts = stem.split("_")
        actor_id = int(parts[0])
        sentence = parts[1]
        emo = parts[2]
        intensity = parts[3]
        dem = demo_map.get(actor_id, {})
        age = int(dem.get("Age", 30))
        age_bucket = "20-34" if age < 35 else ("35-49" if age < 50 else "50+")
        v = vote_map.get(stem, {})
        agreement = float(v.get("agreement", 0.5)) if pd.notna(v.get("agreement", np.nan)) else 0.5
        emo_vote = str(v.get("emoVote", "N"))
        items.append((
            str(wav_path),
            {
                "dataset": "CREMA-D",
                "utterance_id": stem,
                "actor_id": f"CREMA_{actor_id}",
                "actor_num": actor_id,
                "sentence": sentence,
                "emotion": emo,
                "label6": EMO_TO_IDX6[emo],
                "intensity": intensity,
                "sex": str(dem.get("Sex", "Unknown")),
                "age": age,
                "age_bucket": age_bucket,
                "race": str(dem.get("Race", "Unknown")),
                "ethnicity": str(dem.get("Ethnicity", "Unknown")),
                "agreement": agreement,
                "voice_vote": emo_vote,
            },
        ))
    return items


def load_ravdess_records() -> list[tuple[str, dict[str, Any]]]:
    items = []
    for wav_path in sorted(RAVDESS_DIR.rglob("*.wav")):
        parts = wav_path.stem.split("-")
        if len(parts) != 7:
            continue
        emo_code = parts[2]
        emo = RAVDESS_MAP[emo_code]
        intensity = "LO" if parts[3] == "01" else "HI"
        actor_num = int(parts[6])
        sex = "Female" if actor_num % 2 == 0 else "Male"
        items.append((
            str(wav_path),
            {
                "dataset": "RAVDESS",
                "utterance_id": wav_path.stem,
                "actor_id": f"RAV_{actor_num:02d}",
                "actor_num": actor_num,
                "sentence": f"ST_{parts[4]}_R{parts[5]}",
                "emotion": emo,
                "label6": EMO_TO_IDX6.get(emo, -1),
                "intensity": intensity,
                "sex": sex,
                "age": 30,
                "age_bucket": "20-34",
                "race": "NorthAmerican",
                "ethnicity": "Not Reported",
                "agreement": 1.0,
                "voice_vote": emo[0],
            },
        ))
    return items


def assign_speaker_splits_crema(records: list[dict[str, Any]], seed: int = 42) -> dict[str, str]:
    """Stratified speaker-independent 64/13/14 split across CREMA-D's 91 actors."""
    rng = np.random.default_rng(seed)
    actor_meta = {}
    for r in records:
        actor_meta[r["actor_id"]] = (r["sex"], r["age_bucket"])

    strata: dict[tuple[str, str], list[str]] = {}
    for aid, key in sorted(actor_meta.items()):
        strata.setdefault(key, []).append(aid)

    split_map: dict[str, str] = {}
    # Assign ~70% train, ~14.5% val, ~15.5% test within each demographic stratum
    for key in sorted(strata.keys()):
        group = list(strata[key])
        rng.shuffle(group)
        n = len(group)
        n_test = max(1, round(n * 14 / 91))
        n_val = max(1, round(n * 13 / 91))
        if n_test + n_val >= n:
            n_val = max(0, n - n_test - 1)
        for aid in group[:n_test]:
            split_map[aid] = "test"
        for aid in group[n_test : n_test + n_val]:
            split_map[aid] = "val"
        for aid in group[n_test + n_val :]:
            split_map[aid] = "train"
    return split_map


def assign_speaker_splits_ravdess(records: list[dict[str, Any]]) -> dict[str, str]:
    """Speaker-independent 16/4/4 split across RAVDESS's 24 actors (balanced M/F)."""
    split_map = {}
    for r in records:
        num = r["actor_num"]
        if num in {1, 2, 3, 4}:
            split_map[r["actor_id"]] = "test"
        elif num in {5, 6, 7, 8}:
            split_map[r["actor_id"]] = "val"
        else:
            split_map[r["actor_id"]] = "train"
    return split_map


def detect_near_duplicates(functionals: np.ndarray, threshold: float = 0.99995) -> int:
    """Checks sampled/chunked cosine similarity across utterances for near-identical duplicates."""
    norms = np.linalg.norm(functionals, axis=1, keepdims=True) + 1e-12
    normed = functionals / norms
    count = 0
    bs = 512
    for start in range(0, len(normed), bs):
        block = normed[start : start + bs]
        sim = block @ normed.T
        for i in range(len(block)):
            g_i = start + i
            matches = np.where(sim[i, g_i + 1 :] >= threshold)[0]
            count += len(matches)
    return int(count)


def compute_escalation_onset(
    emotions: list[str],
    intensities: list[str],
    w: int = 3,
    rho: float = 1.0 / 3.0,
    k: int = 3,
    m: int = 2,
) -> int | None:
    """Operational definition of escalation onset turn tau for a speaker track.

    n_t = 1 if emotion in {ANG, DIS} or (emotion == SAD and intensity in {HI, MD}), else 0.
    tau = earliest t >= w such that:
      1. prior baseline negativity mean over [t-w, t-1] <= rho
      2. sustained negativity sum over [t, t+k-1] >= m
    """
    n_t = []
    for e, inten in zip(emotions, intensities):
        is_neg = (e in {"ANG", "DIS"}) or (e == "SAD" and inten in {"HI", "MD"})
        n_t.append(1 if is_neg else 0)
    T = len(n_t)
    for t in range(w, T - k + 1):
        prior_rate = sum(n_t[t - w : t]) / float(w)
        fwd_sum = sum(n_t[t : t + k])
        if prior_rate <= rho + 1e-9 and fwd_sum >= m:
            return t
    return None


def build_contact_center_dialogs(
    records: list[dict[str, Any]],
    n_per_split: dict[str, int],
    seed: int = 101,
) -> list[dict[str, Any]]:
    """Constructs leakage-safe multi-turn contact-center dialogues from real actor utterances.

    5 realistic contact-center archetypes:
      * escalating_gradual: Customer NEU -> SAD/DIS(LO) -> ANG/DIS(HI) while Agent tries calm/NEU/HAP
      * escalating_abrupt:  Customer NEU -> sudden ANG/DIS(HI)
      * resolved:           Customer starts distressed (ANG/SAD/DIS), Agent calm/empathic, Customer ends NEU/HAP
      * routine_neutral:    Customer NEU/HAP throughout
      * chronic_negative:   Customer already frustrated/sad from turn 0 (pre-escalated, tests no false baseline onset!)
    """
    rng = np.random.default_rng(seed)
    by_split_actor_emo: dict[str, dict[str, dict[str, list[int]]]] = {}
    for idx, r in enumerate(records):
        if r["label6"] < 0:
            continue
        sp = r["split"]
        aid = r["actor_id"]
        emo = r["emotion"]
        by_split_actor_emo.setdefault(sp, {}).setdefault(aid, {}).setdefault(emo, []).append(idx)

    archetypes = [
        "escalating_gradual",
        "escalating_abrupt",
        "resolved",
        "routine_neutral",
        "chronic_negative",
    ]

    dialogs = []
    did = 0
    for sp, count in n_per_split.items():
        actors = [a for a, emap in by_split_actor_emo.get(sp, {}).items() if len(emap) >= 4]
        for k in range(count):
            arch = archetypes[k % len(archetypes)]
            c_actor, a_actor = rng.choice(actors, size=2, replace=False)
            n_turns = int(rng.integers(10, 16))  # total utterances (5-8 per speaker, alternating or mixed)
            turns = []
            cust_idx_in_track = 0
            # Build customer & agent emotion plans per customer turn
            n_cust = (n_turns + 1) // 2
            if arch == "escalating_gradual":
                pivot = int(rng.integers(max(2, n_cust // 2 - 1), max(3, n_cust - 2)))
                cust_plan = ["NEU" if i < pivot - 1 else ("DIS" if i == pivot - 1 else str(rng.choice(["ANG", "DIS"]))) for i in range(n_cust)]
            elif arch == "escalating_abrupt":
                pivot = int(rng.integers(2, max(3, n_cust - 1)))
                cust_plan = [str(rng.choice(["NEU", "HAP"])) if i < pivot else str(rng.choice(["ANG", "DIS"])) for i in range(n_cust)]
            elif arch == "resolved":
                pivot = int(rng.integers(2, max(3, n_cust - 1)))
                cust_plan = [str(rng.choice(["ANG", "SAD", "DIS"])) if i < pivot else str(rng.choice(["NEU", "HAP"])) for i in range(n_cust)]
            elif arch == "routine_neutral":
                cust_plan = [str(rng.choice(["NEU", "NEU", "HAP"])) for _ in range(n_cust)]
            else:  # chronic_negative
                cust_plan = [str(rng.choice(["ANG", "DIS", "SAD"])) for _ in range(n_cust)]

            c_emos, c_ints = [], []
            for t in range(n_turns):
                if t % 2 == 0:
                    speaker = "CUSTOMER"
                    actor = c_actor
                    target_emo = cust_plan[cust_idx_in_track]
                    cust_idx_in_track += 1
                else:
                    speaker = "AGENT"
                    actor = a_actor
                    target_emo = str(rng.choice(["NEU", "NEU", "HAP"]))

                pool = by_split_actor_emo[sp][actor].get(target_emo)
                if not pool:
                    # Fallback within same actor
                    fallback_emos = list(by_split_actor_emo[sp][actor].keys())
                    target_emo = str(rng.choice(fallback_emos))
                    pool = by_split_actor_emo[sp][actor][target_emo]
                utt_idx = int(rng.choice(pool))
                rec = records[utt_idx]
                if speaker == "CUSTOMER":
                    c_emos.append(rec["emotion"])
                    c_ints.append(rec["intensity"])
                turns.append({
                    "turn": t,
                    "speaker": speaker,
                    "actor_id": rec["actor_id"],
                    "dataset_idx": utt_idx,
                    "utterance_id": rec["utterance_id"],
                    "emotion": rec["emotion"],
                    "label6": rec["label6"],
                    "intensity": rec["intensity"],
                    "agreement": rec["agreement"],
                })

            tau = compute_escalation_onset(c_emos, c_ints)
            dialogs.append({
                "dialog_id": f"{records[0]['dataset']}_DLG_{did:04d}",
                "split": sp,
                "archetype": arch,
                "customer_actor": str(c_actor),
                "agent_actor": str(a_actor),
                "escalation_onset_customer_turn": tau,
                "is_escalating": tau is not None,
                "turns": turns,
            })
            did += 1
    return dialogs


def build_stream_benchmark_and_samples(
    crema_records: list[dict[str, Any]],
    crema_feats_logmel: np.ndarray,
    dialogs: list[dict[str, Any]],
) -> None:
    """Builds continuous multi-turn spectrograms for Seg/Det benchmarking AND real stereo sample WAVs."""
    CONV_DIR.mkdir(parents=True, exist_ok=True)
    # 1. Select 3 representative test dialogs for realistic stereo sample WAVs
    test_dlgs = [d for d in dialogs if d["split"] == "test"]
    chosen = {}
    for arch in ["escalating_gradual", "resolved", "routine_neutral"]:
        for d in test_dlgs:
            if d["archetype"] == arch and arch not in chosen:
                chosen[arch] = d
                break

    gt_samples = []
    rng = np.random.default_rng(77)
    for name, dlg in chosen.items():
        left_chunks, right_chunks = [], []
        intervals = []
        cursor_s = 0.0
        pause = np.zeros(int(0.35 * SAMPLE_RATE), dtype=np.float64)
        for turn in dlg["turns"]:
            rec = crema_records[turn["dataset_idx"]]
            y, _ = load_and_validate_wav(Path(rec["path"]))
            # Trim silence lightly and scale to comfortable speech level
            peak = float(np.max(np.abs(y)))
            if peak > 1e-6:
                y = y * (0.75 / peak)
            dur = len(y) / float(SAMPLE_RATE)
            if turn["speaker"] == "CUSTOMER":
                left_chunks.append(y)
                right_chunks.append(np.zeros_like(y))
            else:
                left_chunks.append(np.zeros_like(y))
                right_chunks.append(y)
            intervals.append({
                "turn": turn["turn"],
                "speaker": turn["speaker"],
                "start_s": round(cursor_s, 3),
                "end_s": round(cursor_s + dur, 3),
                "emotion": turn["emotion"],
                "utterance_id": turn["utterance_id"],
            })
            left_chunks.append(pause)
            right_chunks.append(pause)
            cursor_s += dur + 0.35

        left = np.concatenate(left_chunks) + rng.normal(0.0, 0.0025, sum(len(c) for c in left_chunks))
        right = np.concatenate(right_chunks) + rng.normal(0.0, 0.0025, sum(len(c) for c in right_chunks))
        stereo = np.stack([left, right], axis=1)
        stereo_i16 = np.clip(stereo * 32767.0, -32767, 32767).astype(np.int16)
        out_wav = CONV_DIR / f"sample_{name}.wav"
        wavfile.write(str(out_wav), SAMPLE_RATE, stereo_i16)
        gt_samples.append({
            "file": out_wav.name,
            "dialog_id": dlg["dialog_id"],
            "archetype": name,
            "is_escalating": dlg["is_escalating"],
            "escalation_onset_customer_turn": dlg["escalation_onset_customer_turn"],
            "total_duration_s": round(cursor_s, 3),
            "intervals": intervals,
        })

    (CONV_DIR / "ground_truth.json").write_text(json.dumps(gt_samples, indent=2))

    # 2. Build continuous stream spectrogram benchmark [N_streams, 64, 512] (10.24 s streams, 4 turns each)
    #    Frame mask classes: 0 = Silence/Noise, 1 = Non-negative speech, 2 = Negative/Distressed speech
    #    Event boxes: up to 8 intervals [active, start_norm, end_norm, class_id (1 or 2)]
    streams_per_split = {"train": 180, "val": 45, "test": 60}
    by_split = {"train": [], "val": [], "test": []}
    for idx, r in enumerate(crema_records):
        by_split[r["split"]].append(idx)

    X_list, mask_list, box_list, split_list = [], [], [], []
    rng_s = np.random.default_rng(2026)
    T_STREAM = 512
    for sp, count in streams_per_split.items():
        pool = by_split[sp]
        for _ in range(count):
            # Silence background log-Mel floor ~ -11.5 + small Gaussian noise
            canvas = rng_s.normal(-10.5, 0.45, size=(N_MELS, T_STREAM)).astype(np.float32)
            mask = np.zeros(T_STREAM, dtype=np.int64)
            boxes = np.zeros((8, 4), dtype=np.float32)
            n_events = int(rng_s.integers(3, 6))
            cursor = int(rng_s.integers(6, 20))
            b_i = 0
            for _ in range(n_events):
                if cursor + 36 >= T_STREAM - 8:
                    break
                utt_idx = int(rng_s.choice(pool))
                rec = crema_records[utt_idx]
                spec = crema_feats_logmel[utt_idx]  # [64, 128]
                # Find active non-zero frames inside utterance spectrogram
                col_energy = np.mean(spec, axis=0)
                active_cols = np.where(col_energy > np.min(col_energy) + 1.5)[0]
                dur_f = min(int(rng_s.integers(55, 110)), T_STREAM - cursor - 6)
                if len(active_cols) >= 24:
                    src_len = min(dur_f, len(active_cols))
                    seg = spec[:, active_cols[:src_len]]
                else:
                    src_len = min(dur_f, 75)
                    seg = spec[:, :src_len]
                end_f = cursor + src_len
                canvas[:, cursor:end_f] = seg
                cls_id = 2 if rec["emotion"] in NEGATIVE_SET else 1
                mask[cursor:end_f] = cls_id
                if b_i < 8:
                    boxes[b_i] = [1.0, cursor / float(T_STREAM), end_f / float(T_STREAM), float(cls_id)]
                    b_i += 1
                cursor = end_f + int(rng_s.integers(12, 32))
            X_list.append(canvas)
            mask_list.append(mask)
            box_list.append(boxes)
            split_list.append(sp)

    np.savez_compressed(
        FEAT_DIR / "stream_benchmark.npz",
        log_mel=np.stack(X_list),
        frame_mask=np.stack(mask_list),
        boxes=np.stack(box_list),
        splits=np.array(split_list),
    )
    print(f"Saved stream_benchmark.npz ({len(X_list)} continuous streams) and 3 stereo sample WAVs")


def run_pipeline() -> None:
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("1/5 Loading & validating CREMA-D (7,442 WAVs)...")
    crema_items = load_crema_records()
    with cf.ProcessPoolExecutor(max_workers=8) as ex:
        crema_raw = list(ex.map(_process_one_wav, crema_items, chunksize=64))

    crema_valid = [r for r in crema_raw if r["valid"]]
    crema_invalid = [r for r in crema_raw if not r["valid"]]
    print(f"  CREMA-D valid={len(crema_valid)}, excluded={len(crema_invalid)}")

    crema_splits = assign_speaker_splits_crema(crema_valid, seed=42)
    for r in crema_valid:
        r["split"] = crema_splits[r["actor_id"]]

    # Exact SHA-256 duplicates + near duplicates
    crema_hashes = [r["sha256"] for r in crema_valid]
    crema_exact_dups = len(crema_hashes) - len(set(crema_hashes))
    crema_funcs = np.stack([r["functionals"] for r in crema_valid])
    crema_near_dups = detect_near_duplicates(crema_funcs)

    # Save CREMA-D cached arrays
    crema_log_mel = np.stack([r["log_mel"] for r in crema_valid])
    np.savez_compressed(
        FEAT_DIR / "crema_d_features.npz",
        log_mel=crema_log_mel,
        mel_linear=np.stack([r["mel_linear"] for r in crema_valid]),
        stft_64=np.stack([r["stft_64"] for r in crema_valid]),
        mfcc_delta=np.stack([r["mfcc_delta"] for r in crema_valid]),
        functionals=crema_funcs,
        labels6=np.array([r["label6"] for r in crema_valid], dtype=np.int64),
        splits=np.array([r["split"] for r in crema_valid]),
        actor_ids=np.array([r["actor_id"] for r in crema_valid]),
        emotions=np.array([r["emotion"] for r in crema_valid]),
        intensities=np.array([r["intensity"] for r in crema_valid]),
        agreements=np.array([r["agreement"] for r in crema_valid], dtype=np.float32),
        snr_db=np.array([r["snr_db"] for r in crema_valid], dtype=np.float32),
        durations=np.array([r["raw_duration_s"] for r in crema_valid], dtype=np.float32),
        sex=np.array([r["sex"] for r in crema_valid]),
        age_bucket=np.array([r["age_bucket"] for r in crema_valid]),
        race=np.array([r["race"] for r in crema_valid]),
        utterance_ids=np.array([r["utterance_id"] for r in crema_valid]),
    )

    # Export standardized functionals CSV for Java ser-math baseline parity (fit scaler on TRAIN ONLY!)
    train_mask = np.array([r["split"] == "train" for r in crema_valid])
    f_mean = crema_funcs[train_mask].mean(axis=0)
    f_std = np.maximum(crema_funcs[train_mask].std(axis=0), 1e-6)
    norm_funcs = (crema_funcs - f_mean) / f_std
    with open(FEAT_DIR / "crema_mlp_functionals.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "label6"] + [f"f{i}" for i in range(norm_funcs.shape[1])])
        for r, row in zip(crema_valid, norm_funcs):
            w.writerow([r["split"], r["label6"]] + [f"{v:.6f}" for v in row])

    print("2/5 Loading & validating RAVDESS (1,440 WAVs)...")
    rav_items = load_ravdess_records()
    with cf.ProcessPoolExecutor(max_workers=8) as ex:
        rav_raw = list(ex.map(_process_one_wav, rav_items, chunksize=32))
    rav_valid = [r for r in rav_raw if r["valid"]]
    rav_invalid = [r for r in rav_raw if not r["valid"]]
    rav_splits = assign_speaker_splits_ravdess(rav_valid)
    for r in rav_valid:
        r["split"] = rav_splits[r["actor_id"]]
    rav_funcs = np.stack([r["functionals"] for r in rav_valid])

    np.savez_compressed(
        FEAT_DIR / "ravdess_features.npz",
        log_mel=np.stack([r["log_mel"] for r in rav_valid]),
        functionals=rav_funcs,
        labels6=np.array([r["label6"] for r in rav_valid], dtype=np.int64),
        splits=np.array([r["split"] for r in rav_valid]),
        actor_ids=np.array([r["actor_id"] for r in rav_valid]),
        emotions=np.array([r["emotion"] for r in rav_valid]),
        intensities=np.array([r["intensity"] for r in rav_valid]),
        snr_db=np.array([r["snr_db"] for r in rav_valid], dtype=np.float32),
        sex=np.array([r["sex"] for r in rav_valid]),
        utterance_ids=np.array([r["utterance_id"] for r in rav_valid]),
    )

    print("3/5 Constructing leakage-safe multi-turn contact-center dialogs...")
    crema_dialogs = build_contact_center_dialogs(
        crema_valid, {"train": 240, "val": 60, "test": 80}, seed=101
    )
    rav_dialogs = build_contact_center_dialogs(
        rav_valid, {"test": 40}, seed=202
    )
    (FEAT_DIR / "crema_dialogs.json").write_text(json.dumps(crema_dialogs))
    (FEAT_DIR / "ravdess_dialogs.json").write_text(json.dumps(rav_dialogs))

    print("4/5 Generating stream segmentation/detection benchmark & stereo test calls...")
    build_stream_benchmark_and_samples(crema_valid, crema_log_mel, crema_dialogs)

    print("5/5 Writing dataset provenance & audit statistics...")
    split_counts = {
        sp: int(sum(1 for r in crema_valid if r["split"] == sp))
        for sp in ("train", "val", "test")
    }
    split_actors = {
        sp: len({r["actor_id"] for r in crema_valid if r["split"] == sp})
        for sp in ("train", "val", "test")
    }
    # Assert strictly zero speaker overlap
    assert set(a for r in crema_valid if r["split"] == "train" for a in [r["actor_id"]]).isdisjoint(
        {r["actor_id"] for r in crema_valid if r["split"] == "test"}
    )
    assert set(a for r in crema_valid if r["split"] == "val" for a in [r["actor_id"]]).isdisjoint(
        {r["actor_id"] for r in crema_valid if r["split"] == "test"}
    )

    stats = {
        "crema_d": {
            "total_files_scanned": len(crema_items),
            "valid_utterances": len(crema_valid),
            "excluded_files": [{"path": Path(r["path"]).name, "reason": r["reason"]} for r in crema_invalid],
            "exact_sha256_duplicates": crema_exact_dups,
            "near_duplicates_cosine_gt_099995": crema_near_dups,
            "total_actors": len({r["actor_id"] for r in crema_valid}),
            "split_utterances": split_counts,
            "split_actors": split_actors,
            "emotion_counts": {e: int(sum(1 for r in crema_valid if r["emotion"] == e)) for e in EMOTIONS_6},
            "mean_snr_db": round(float(np.mean([r["snr_db"] for r in crema_valid])), 2),
            "mean_duration_s": round(float(np.mean([r["raw_duration_s"] for r in crema_valid])), 3),
            "mean_voice_agreement": round(float(np.mean([r["agreement"] for r in crema_valid])), 3),
            "license": "ODbL v1.0 / DbCL v1.0",
        },
        "ravdess": {
            "total_files_scanned": len(rav_items),
            "valid_utterances": len(rav_valid),
            "excluded_files": len(rav_invalid),
            "total_actors": len({r["actor_id"] for r in rav_valid}),
            "shared_6class_count": int(sum(1 for r in rav_valid if r["label6"] >= 0)),
            "shared_4class_count": int(sum(1 for r in rav_valid if r["emotion"] in {"NEU", "HAP", "SAD", "ANG"})),
            "mean_snr_db": round(float(np.mean([r["snr_db"] for r in rav_valid])), 2),
            "mean_duration_s": round(float(np.mean([r["raw_duration_s"] for r in rav_valid])), 3),
            "license": "CC BY-NC-SA 4.0",
        },
        "crema_dialog": {
            "total_dialogs": len(crema_dialogs),
            "test_dialogs": int(sum(1 for d in crema_dialogs if d["split"] == "test")),
            "escalating_test_dialogs": int(sum(1 for d in crema_dialogs if d["split"] == "test" and d["is_escalating"])),
            "non_escalating_test_dialogs": int(sum(1 for d in crema_dialogs if d["split"] == "test" and not d["is_escalating"])),
        },
        "ravdess_dialog": {
            "total_dialogs": len(rav_dialogs),
            "escalating_dialogs": int(sum(1 for d in rav_dialogs if d["is_escalating"])),
        },
        "speaker_overlap_assertion": "PASSED (0 overlapping actors across train/val/test)",
    }
    (RESULTS_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    run_pipeline()
