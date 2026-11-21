"""Project 20 Item 8 Benchmark: Classification, Semantic Segmentation, and Temporal Event Detection.

Compares on continuous multi-turn conversation spectrograms (stream_benchmark.npz):
  * Semantic Segmentation: SpecUNet1D (Frame Acc, mIoU, Macro-F1, Latency)
  * Two-Stage Detectors: R-CNN-1D, Fast-RCNN-1D, Faster-RCNN-1D (RPN + RoI-Align)
  * Single-Stage Detectors: YOLO-1D, RetinaNet-1D (Multi-scale FPN + Focal Loss)
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ser_research.models import (
    FastRCNN1D,
    FasterRCNN1D,
    RCNN1D,
    RetinaNet1D,
    SpecUNet1D,
    YOLO1D,
)

T_STREAM = 512


def interval_iou(a_s: float, a_e: float, b_s: float, b_e: float) -> float:
    inter = max(0.0, min(a_e, b_e) - max(a_s, b_s))
    union = max(1e-6, (a_e - a_s) + (b_e - b_s) - inter)
    return inter / union


def nms_1d(boxes: list[tuple[float, float, int, float]], iou_thresh: float = 0.45) -> list[tuple[float, float, int, float]]:
    boxes = sorted(boxes, key=lambda b: b[3], reverse=True)
    kept = []
    for b in boxes:
        if all(interval_iou(b[0], b[1], k[0], k[1]) < iou_thresh for k in kept):
            kept.append(b)
    return kept


def generate_energy_proposals(log_mel_stream: np.ndarray, max_props: int = 8) -> np.ndarray:
    """Classical energy-contour region proposals (used by R-CNN and Fast R-CNN)."""
    energy = log_mel_stream.mean(axis=0)  # [512]
    thresh = float(np.percentile(energy, 35) + 0.35 * (np.max(energy) - np.percentile(energy, 35)))
    active = energy > thresh
    props = []
    in_seg = False
    start = 0
    for t in range(len(active)):
        if active[t] and not in_seg:
            in_seg = True
            start = t
        elif not active[t] and in_seg:
            in_seg = False
            if t - start >= 18:
                props.append([start / float(T_STREAM), t / float(T_STREAM)])
    if in_seg and len(active) - start >= 18:
        props.append([start / float(T_STREAM), 1.0])
    if not props:
        props = [[0.05, 0.25], [0.35, 0.55], [0.65, 0.85]]
    return np.array(props[:max_props], dtype=np.float32)


def evaluate_detector(
    gt_boxes_list: list[np.ndarray],
    pred_boxes_list: list[list[tuple[float, float, int, float]]],
    latencies_ms: list[float],
) -> dict[str, Any]:
    """Computes Event F1@0.5 IoU, mean matched IoU, class-aware mAP@0.5, and p50 latency."""
    tp = 0
    fp = 0
    fn = 0
    matched_ious = []
    per_cls_scores: dict[int, list[tuple[float, int]]] = {1: [], 2: []}
    per_cls_gt = {1: 0, 2: 0}

    for gt_arr, preds in zip(gt_boxes_list, pred_boxes_list):
        valid_gt = [g for g in gt_arr if g[0] > 0.5]
        for g in valid_gt:
            per_cls_gt[int(g[3])] += 1
        used = set()
        for ps, pe, pcls, pscore in sorted(preds, key=lambda x: x[3], reverse=True):
            best_iou = 0.0
            best_j = -1
            for j, g in enumerate(valid_gt):
                if j in used:
                    continue
                iou = interval_iou(ps, pe, float(g[1]), float(g[2]))
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= 0.5 and best_j >= 0 and int(valid_gt[best_j][3]) == pcls:
                tp += 1
                used.add(best_j)
                matched_ious.append(best_iou)
                per_cls_scores[pcls].append((pscore, 1))
            else:
                fp += 1
                if pcls in per_cls_scores:
                    per_cls_scores[pcls].append((pscore, 0))
        fn += len(valid_gt) - len(used)

    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)

    aps = []
    for cls in (1, 2):
        n_gt = max(1, per_cls_gt[cls])
        items = sorted(per_cls_scores[cls], key=lambda x: x[0], reverse=True)
        tp_c, fp_c = 0, 0
        precs, recs = [1.0], [0.0]
        for _, is_tp in items:
            if is_tp:
                tp_c += 1
            else:
                fp_c += 1
            precs.append(tp_c / max(1, tp_c + fp_c))
            recs.append(tp_c / n_gt)
        # 11-point / area AP
        ap = 0.0
        for r_t in np.linspace(0.0, 1.0, 11):
            p_at = [p for p, r in zip(precs, recs) if r >= r_t]
            ap += max(p_at) if p_at else 0.0
        aps.append(ap / 11.0)

    return {
        "mAP@0.5": round(float(np.mean(aps)), 4),
        "event_precision": round(float(prec), 4),
        "event_recall": round(float(rec), 4),
        "event_f1": round(float(f1), 4),
        "mean_iou": round(float(np.mean(matched_ious)) if matched_ious else 0.0, 4),
        "latency_ms_per_stream": round(float(np.median(latencies_ms)), 3),
    }


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.exp(-bce)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * ((1.0 - pt) ** gamma) * bce).mean()


def run_segdet_experiments(stream_npz: dict[str, np.ndarray]) -> dict[str, Any]:
    print("Running E8-SegDet: Semantic Segmentation (SpecUNet) & Detection (R-CNN, Fast/Faster R-CNN, YOLO, RetinaNet)...", flush=True)
    torch.set_num_threads(6)
    splits = stream_npz["splits"]
    tr = splits == "train"
    te = splits == "test"

    x_all = stream_npz["log_mel"]
    mu = float(x_all[tr].mean())
    sd = max(1e-6, float(x_all[tr].std()))
    x_norm = ((x_all - mu) / sd).astype(np.float32)

    x_tr = torch.from_numpy(x_norm[tr, None, :, :])
    x_te = torch.from_numpy(x_norm[te, None, :, :])
    mask_tr = torch.from_numpy(stream_npz["frame_mask"][tr]).long()
    mask_te = stream_npz["frame_mask"][te]
    boxes_tr = stream_npz["boxes"][tr]
    boxes_te = list(stream_npz["boxes"][te])

    results: dict[str, Any] = {}

    # 1. Semantic Segmentation: SpecUNet1D
    unet = SpecUNet1D(num_classes=3)
    opt = torch.optim.AdamW(unet.parameters(), lr=1.5e-3, weight_decay=1e-4)
    dl = DataLoader(TensorDataset(x_tr, mask_tr), batch_size=16, shuffle=True)
    for _ in range(12):
        unet.train()
        for xb, mb in dl:
            opt.zero_grad()
            loss = F.cross_entropy(unet(xb), mb)
            loss.backward()
            opt.step()
    unet.eval()
    lats = []
    with torch.no_grad():
        preds_u = []
        for i in range(len(x_te)):
            t0 = time.perf_counter()
            lg = unet(x_te[i : i + 1])
            lats.append((time.perf_counter() - t0) * 1000.0)
            preds_u.append(lg.argmax(1)[0].numpy())
    pred_mask = np.stack(preds_u)
    frame_acc = float((pred_mask == mask_te).mean())
    ious = []
    for cls in (0, 1, 2):
        inter = int(((pred_mask == cls) & (mask_te == cls)).sum())
        uni = int(((pred_mask == cls) | (mask_te == cls)).sum())
        ious.append(inter / max(1, uni))
    # Also derive detected boxes from UNet foreground runs for apples-to-apples comparison
    unet_boxes = []
    for pm in pred_mask:
        b_list = []
        cur_c, start = int(pm[0]), 0
        for t in range(1, len(pm) + 1):
            c_now = int(pm[t]) if t < len(pm) else 0
            if c_now != cur_c:
                if cur_c in (1, 2) and (t - start) >= 16:
                    b_list.append((start / float(T_STREAM), t / float(T_STREAM), cur_c, 0.85))
                cur_c, start = c_now, t
        unet_boxes.append(b_list)
    unet_det_eval = evaluate_detector(boxes_te, unet_boxes, lats)
    results["SpecUNet-1D (Seg)"] = {
        "task": "Semantic Segmentation",
        "frame_accuracy": round(frame_acc, 4),
        "mIoU": round(float(np.mean(ious)), 4),
        **unet_det_eval,
    }
    print(f"  SpecUNet-1D: FrameAcc={frame_acc:.4f}, mIoU={np.mean(ious):.4f}, EventF1={unet_det_eval['event_f1']:.4f}")

    # Shared external proposals for R-CNN and Fast R-CNN
    props_tr = [generate_energy_proposals(x_norm[tr][i]) for i in range(int(tr.sum()))]
    props_te = [generate_energy_proposals(x_norm[te][i]) for i in range(int(te.sum()))]

    def match_proposals(props: np.ndarray, gt_arr: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        valid_gt = [g for g in gt_arr if g[0] > 0.5]
        cls_t = torch.zeros(len(props), dtype=torch.long)
        reg_t = torch.zeros((len(props), 2), dtype=torch.float32)
        for pi, (ps, pe) in enumerate(props):
            best_i, best_g = 0.0, None
            for g in valid_gt:
                iou = interval_iou(ps, pe, float(g[1]), float(g[2]))
                if iou > best_i:
                    best_i, best_g = iou, g
            if best_i >= 0.35 and best_g is not None:
                cls_t[pi] = int(best_g[3])
                pc, pw = 0.5 * (ps + pe), max(1e-3, pe - ps)
                gc, gw = 0.5 * (float(best_g[1]) + float(best_g[2])), max(1e-3, float(best_g[2]) - float(best_g[1]))
                reg_t[pi] = torch.tensor([(gc - pc) / pw, np.log(gw / pw)], dtype=torch.float32)
        return cls_t, reg_t

    def crop_resize(spec: torch.Tensor, props: np.ndarray) -> torch.Tensor:
        crops = []
        for ps, pe in props:
            s_i = max(0, int(ps * T_STREAM))
            e_i = min(T_STREAM, max(s_i + 4, int(pe * T_STREAM)))
            sub = spec[:, :, :, s_i:e_i]
            crops.append(F.interpolate(sub, size=(64, 48), mode="bilinear", align_corners=False)[0])
        return torch.stack(crops, dim=0)

    # 2. R-CNN-1D
    rcnn = RCNN1D(num_classes=3)
    opt_r = torch.optim.AdamW(rcnn.parameters(), lr=1.5e-3)
    for _ in range(8):
        rcnn.train()
        for i in range(len(x_tr)):
            crops = crop_resize(x_tr[i : i + 1], props_tr[i])
            ct, rt = match_proposals(props_tr[i], boxes_tr[i])
            opt_r.zero_grad()
            cl, rg = rcnn.forward_crops(crops)
            fg = ct > 0
            loss = F.cross_entropy(cl, ct) + (F.smooth_l1_loss(rg[fg], rt[fg]) if fg.any() else 0.0)
            loss.backward()
            opt_r.step()
    rcnn.eval()
    rcnn_preds, rcnn_lats = [], []
    with torch.no_grad():
        for i in range(len(x_te)):
            t0 = time.perf_counter()
            crops = crop_resize(x_te[i : i + 1], props_te[i])
            cl, rg = rcnn.forward_crops(crops)
            rcnn_lats.append((time.perf_counter() - t0) * 1000.0)
            pr = torch.softmax(cl, dim=1)
            b_list = []
            for pi, (ps, pe) in enumerate(props_te[i]):
                c_id = int(pr[pi, 1:].argmax().item()) + 1
                sc = float(pr[pi, c_id].item())
                pc, pw = 0.5 * (ps + pe), pe - ps
                gc = pc + float(rg[pi, 0].item()) * pw
                gw = pw * float(np.exp(np.clip(rg[pi, 1].item(), -1.0, 1.0)))
                if sc >= 0.35:
                    b_list.append((max(0.0, gc - 0.5 * gw), min(1.0, gc + 0.5 * gw), c_id, sc))
            rcnn_preds.append(nms_1d(b_list))
    results["R-CNN-1D (Two-Stage Slow)"] = {"task": "Proposal CNN", **evaluate_detector(boxes_te, rcnn_preds, rcnn_lats)}

    # 3. Fast R-CNN-1D
    fastrcnn = FastRCNN1D(num_classes=3)
    opt_fr = torch.optim.AdamW(fastrcnn.parameters(), lr=1.5e-3)
    for _ in range(12):
        fastrcnn.train()
        for i in range(0, len(x_tr), 16):
            xb = x_tr[i : i + 16]
            feats = fastrcnn.backbone(xb)
            opt_fr.zero_grad()
            tot_l = torch.tensor(0.0)
            for bi in range(len(xb)):
                pr_np = props_tr[i + bi]
                ct, rt = match_proposals(pr_np, boxes_tr[i + bi])
                cl, rg = fastrcnn.forward_proposals(feats[bi], torch.from_numpy(pr_np))
                fg = ct > 0
                tot_l = tot_l + F.cross_entropy(cl, ct) + (F.smooth_l1_loss(rg[fg], rt[fg]) if fg.any() else 0.0)
            (tot_l / len(xb)).backward()
            opt_fr.step()
    fastrcnn.eval()
    fr_preds, fr_lats = [], []
    with torch.no_grad():
        for i in range(len(x_te)):
            t0 = time.perf_counter()
            feat = fastrcnn.backbone(x_te[i : i + 1])[0]
            cl, rg = fastrcnn.forward_proposals(feat, torch.from_numpy(props_te[i]))
            fr_lats.append((time.perf_counter() - t0) * 1000.0)
            pr = torch.softmax(cl, dim=1)
            b_list = []
            for pi, (ps, pe) in enumerate(props_te[i]):
                c_id = int(pr[pi, 1:].argmax().item()) + 1
                sc = float(pr[pi, c_id].item())
                pc, pw = 0.5 * (ps + pe), pe - ps
                gc = pc + float(rg[pi, 0].item()) * pw
                gw = pw * float(np.exp(np.clip(rg[pi, 1].item(), -1.0, 1.0)))
                if sc >= 0.35:
                    b_list.append((max(0.0, gc - 0.5 * gw), min(1.0, gc + 0.5 * gw), c_id, sc))
            fr_preds.append(nms_1d(b_list))
    results["Fast-RCNN-1D (RoI-Align)"] = {"task": "Shared Backbone + RoI", **evaluate_detector(boxes_te, fr_preds, fr_lats)}

    # 4. Faster R-CNN-1D (RPN + RoI-Align)
    faster = FasterRCNN1D(num_classes=3)
    opt_fa = torch.optim.AdamW(faster.parameters(), lr=1.5e-3)
    anc_np = faster.anchors(64, torch.device("cpu")).numpy()
    for _ in range(12):
        faster.train()
        for i in range(0, len(x_tr), 16):
            xb = x_tr[i : i + 16]
            opt_fa.zero_grad()
            feat, rpn_obj, rpn_reg = faster.forward_rpn(xb)
            tot_l = torch.tensor(0.0)
            for bi in range(len(xb)):
                valid_gt = [g for g in boxes_tr[i + bi] if g[0] > 0.5]
                obj_tgt = torch.zeros(len(anc_np))
                reg_tgt = torch.zeros((len(anc_np), 2))
                for ai, (ac, aw) in enumerate(anc_np):
                    for g in valid_gt:
                        iou = interval_iou(ac - 0.5 * aw, ac + 0.5 * aw, float(g[1]), float(g[2]))
                        if iou >= 0.45:
                            obj_tgt[ai] = 1.0
                            gc, gw = 0.5 * (float(g[1]) + float(g[2])), float(g[2]) - float(g[1])
                            reg_tgt[ai] = torch.tensor([(gc - ac) / aw, np.log(max(1e-3, gw) / aw)])
                            break
                pos = obj_tgt > 0.5
                rpn_l = F.binary_cross_entropy_with_logits(rpn_obj[bi], obj_tgt)
                if pos.any():
                    rpn_l = rpn_l + F.smooth_l1_loss(rpn_reg[bi][pos], reg_tgt[pos])
                # Stage 2 RoI classification on top-8 RPN boxes
                top_idx = torch.topk(rpn_obj[bi].detach(), 8).indices
                prop_boxes = []
                for idx in top_idx:
                    ac, aw = float(anc_np[idx, 0]), float(anc_np[idx, 1])
                    dc, dw = float(rpn_reg[bi, idx, 0].detach()), float(rpn_reg[bi, idx, 1].detach())
                    pc = ac + dc * aw
                    pw = aw * float(np.exp(np.clip(dw, -1.0, 1.0)))
                    prop_boxes.append([max(0.0, pc - 0.5 * pw), min(1.0, pc + 0.5 * pw)])
                prop_np = np.array(prop_boxes, dtype=np.float32)
                ct, rt = match_proposals(prop_np, boxes_tr[i + bi])
                roi_feat = faster.roi_head(
                    torch.stack([
                        F.interpolate(feat[bi : bi + 1, :, max(0, int(p[0] * 64)) : min(64, max(int(p[0] * 64) + 1, int(p[1] * 64)))],
                                      size=8, mode="linear", align_corners=False)[0].reshape(-1)
                        for p in prop_np
                    ])
                )
                cl = faster.cls_head(roi_feat)
                rg = faster.reg_head(roi_feat)
                fg = ct > 0
                tot_l = tot_l + rpn_l + F.cross_entropy(cl, ct) + (F.smooth_l1_loss(rg[fg], rt[fg]) if fg.any() else 0.0)
            (tot_l / len(xb)).backward()
            opt_fa.step()
    faster.eval()
    fa_preds, fa_lats = [], []
    with torch.no_grad():
        for i in range(len(x_te)):
            t0 = time.perf_counter()
            feat, rpn_obj, rpn_reg = faster.forward_rpn(x_te[i : i + 1])
            top_idx = torch.topk(rpn_obj[0], 8).indices
            prop_boxes = []
            for idx in top_idx:
                ac, aw = float(anc_np[idx, 0]), float(anc_np[idx, 1])
                pc = ac + float(rpn_reg[0, idx, 0]) * aw
                pw = aw * float(np.exp(np.clip(float(rpn_reg[0, idx, 1]), -1.0, 1.0)))
                prop_boxes.append([max(0.0, pc - 0.5 * pw), min(1.0, pc + 0.5 * pw)])
            prop_np = np.array(prop_boxes, dtype=np.float32)
            roi_feat = faster.roi_head(
                torch.stack([
                    F.interpolate(feat[0:1, :, max(0, int(p[0] * 64)) : min(64, max(int(p[0] * 64) + 1, int(p[1] * 64)))],
                                  size=8, mode="linear", align_corners=False)[0].reshape(-1)
                    for p in prop_np
                ])
            )
            cl = faster.cls_head(roi_feat)
            rg = faster.reg_head(roi_feat)
            fa_lats.append((time.perf_counter() - t0) * 1000.0)
            pr = torch.softmax(cl, dim=1)
            b_list = []
            for pi, (ps, pe) in enumerate(prop_np):
                c_id = int(pr[pi, 1:].argmax().item()) + 1
                sc = float(pr[pi, c_id].item())
                pc, pw = 0.5 * (ps + pe), pe - ps
                gc = pc + float(rg[pi, 0].item()) * pw
                gw = pw * float(np.exp(np.clip(rg[pi, 1].item(), -1.0, 1.0)))
                if sc >= 0.38:
                    b_list.append((max(0.0, gc - 0.5 * gw), min(1.0, gc + 0.5 * gw), c_id, sc))
            fa_preds.append(nms_1d(b_list))
    results["Faster-RCNN-1D (RPN+RoI)"] = {"task": "End-to-End 2-Stage", **evaluate_detector(boxes_te, fa_preds, fa_lats)}

    # 5. YOLO-1D (Single-pass 16-cell grid)
    yolo = YOLO1D(num_classes=2)
    opt_y = torch.optim.AdamW(yolo.parameters(), lr=1.5e-3)
    S = 16
    for _ in range(14):
        yolo.train()
        for i in range(0, len(x_tr), 16):
            xb = x_tr[i : i + 16]
            opt_y.zero_grad()
            out = yolo(xb)  # [B, 16, 5]
            obj_tgt = torch.zeros((len(xb), S))
            box_tgt = torch.zeros((len(xb), S, 2))
            cls_tgt = torch.zeros((len(xb), S), dtype=torch.long)
            for bi in range(len(xb)):
                for g in boxes_tr[i + bi]:
                    if g[0] < 0.5:
                        continue
                    gc = 0.5 * (float(g[1]) + float(g[2]))
                    gw = max(0.02, float(g[2]) - float(g[1]))
                    cell = min(S - 1, max(0, int(gc * S)))
                    obj_tgt[bi, cell] = 1.0
                    box_tgt[bi, cell] = torch.tensor([gc * S - cell, np.log(gw * S)])
                    cls_tgt[bi, cell] = int(g[3]) - 1
            obj_l = F.binary_cross_entropy_with_logits(out[:, :, 0], obj_tgt)
            pos = obj_tgt > 0.5
            box_l = F.smooth_l1_loss(out[:, :, 1:3][pos], box_tgt[pos]) if pos.any() else 0.0
            cls_l = F.cross_entropy(out[:, :, 3:][pos], cls_tgt[pos]) if pos.any() else 0.0
            (obj_l + 1.5 * box_l + cls_l).backward()
            opt_y.step()
    yolo.eval()
    y_preds, y_lats = [], []
    with torch.no_grad():
        for i in range(len(x_te)):
            t0 = time.perf_counter()
            out = yolo(x_te[i : i + 1])[0]
            y_lats.append((time.perf_counter() - t0) * 1000.0)
            obj_p = torch.sigmoid(out[:, 0])
            cls_p = torch.softmax(out[:, 3:], dim=1)
            b_list = []
            for cell in range(S):
                sc = float(obj_p[cell].item())
                if sc >= 0.35:
                    cx = (cell + float(torch.sigmoid(out[cell, 1]).item())) / S
                    w = float(np.exp(np.clip(float(out[cell, 2].item()), -1.0, 2.0))) / S
                    c_id = int(cls_p[cell].argmax().item()) + 1
                    b_list.append((max(0.0, cx - 0.5 * w), min(1.0, cx + 0.5 * w), c_id, sc * float(cls_p[cell].max().item())))
            y_preds.append(nms_1d(b_list))
    results["YOLO-1D (Single-Stage)"] = {"task": "Grid Single-Shot", **evaluate_detector(boxes_te, y_preds, y_lats)}

    # 6. RetinaNet-1D (FPN + Focal Loss)
    ret = RetinaNet1D(num_classes=2)
    opt_ret = torch.optim.AdamW(ret.parameters(), lr=1.5e-3)
    for _ in range(12):
        ret.train()
        for i in range(0, len(x_tr), 16):
            xb = x_tr[i : i + 16]
            opt_ret.zero_grad()
            cls_out, box_out, anc = ret(xb)
            anc_np_r = anc.numpy()
            cls_tgt = torch.zeros_like(cls_out)
            box_tgt = torch.zeros_like(box_out)
            pos_mask = torch.zeros((len(xb), len(anc_np_r)), dtype=torch.bool)
            for bi in range(len(xb)):
                valid_gt = [g for g in boxes_tr[i + bi] if g[0] > 0.5]
                for ai, (ac, aw) in enumerate(anc_np_r):
                    for g in valid_gt:
                        iou = interval_iou(ac - 0.5 * aw, ac + 0.5 * aw, float(g[1]), float(g[2]))
                        if iou >= 0.45:
                            pos_mask[bi, ai] = True
                            cls_tgt[bi, ai, int(g[3]) - 1] = 1.0
                            gc, gw = 0.5 * (float(g[1]) + float(g[2])), float(g[2]) - float(g[1])
                            box_tgt[bi, ai] = torch.tensor([(gc - ac) / aw, np.log(max(1e-3, gw) / aw)])
                            break
            fl = focal_loss(cls_out, cls_tgt, gamma=2.0, alpha=0.25)
            bl = F.smooth_l1_loss(box_out[pos_mask], box_tgt[pos_mask]) if pos_mask.any() else 0.0
            (fl * 4.0 + bl).backward()
            opt_ret.step()
    ret.eval()
    ret_preds, ret_lats = [], []
    with torch.no_grad():
        for i in range(len(x_te)):
            t0 = time.perf_counter()
            cls_out, box_out, anc = ret(x_te[i : i + 1])
            ret_lats.append((time.perf_counter() - t0) * 1000.0)
            pr = torch.sigmoid(cls_out[0])
            b_list = []
            for ai in range(len(anc)):
                c_idx = int(pr[ai].argmax().item())
                sc = float(pr[ai, c_idx].item())
                if sc >= 0.30:
                    ac, aw = float(anc[ai, 0]), float(anc[ai, 1])
                    gc = ac + float(box_out[0, ai, 0]) * aw
                    gw = aw * float(np.exp(np.clip(float(box_out[0, ai, 1]), -1.0, 1.0)))
                    b_list.append((max(0.0, gc - 0.5 * gw), min(1.0, gc + 0.5 * gw), c_idx + 1, sc))
            ret_preds.append(nms_1d(b_list))
    results["RetinaNet-1D (FPN+Focal)"] = {"task": "FPN Focal Single-Shot", **evaluate_detector(boxes_te, ret_preds, ret_lats)}

    for k, v in results.items():
        if "mAP@0.5" in v:
            print(f"  {k}: mAP@0.5={v['mAP@0.5']:.4f}, EventF1={v['event_f1']:.4f}, Lat={v['latency_ms_per_stream']:.2f}ms")
    return results
