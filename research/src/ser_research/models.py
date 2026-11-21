"""PyTorch architectures for SER, acoustic segmentation/detection, conversational context, and generative augmentation."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MlpBaseline(nn.Module):
    def __init__(self, in_dim: int = 64, num_classes: int = 6) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.head = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.net(x)
        return self.head(emb), emb


class CnnBasic(nn.Module):
    def __init__(self, in_ch: int = 1, num_classes: int = 6) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.features(x)
        pooled = h.mean(dim=(2, 3))
        emb = self.proj(pooled)
        return self.head(emb), emb


class CnnDeep(nn.Module):
    def __init__(self, in_ch: int = 1, num_classes: int = 6) -> None:
        super().__init__()

        def block(ci: int, co: int, first_stride: int = 1) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, stride=first_stride, padding=1, bias=False),
                nn.BatchNorm2d(co),
                nn.ReLU(),
                nn.Conv2d(co, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_ch, 16, first_stride=2),
            block(16, 32),
            block(32, 64),
            nn.Conv2d(64, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )
        self.proj = nn.Sequential(nn.Linear(96, 128), nn.ReLU(), nn.Dropout(0.3))
        self.head = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.features(x)
        emb = self.proj(h.mean(dim=(2, 3)))
        return self.head(emb), emb


class ResBlock2d(nn.Module):
    def __init__(self, ci: int, co: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(ci, co, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(co)
        self.conv2 = nn.Conv2d(co, co, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(co)
        self.skip = (
            nn.Sequential(nn.Conv2d(ci, co, 1, bias=False), nn.BatchNorm2d(co))
            if ci != co
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x))


class ResNetSer(nn.Module):
    """Primary ONNX-exportable 2D Residual CNN with statistics pooling & 128-d embedding."""

    def __init__(self, in_ch: int = 1, num_classes: int = 6, emb_dim: int = 128) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.stage1 = nn.Sequential(ResBlock2d(24, 24), nn.MaxPool2d(2))
        self.stage2 = nn.Sequential(ResBlock2d(24, 48), nn.MaxPool2d(2))
        self.stage3 = ResBlock2d(48, 96)
        self.stage4 = ResBlock2d(96, 96)
        self.emb_proj = nn.Sequential(
            nn.Linear(192, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
        )
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(emb_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        return self.stage4(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.forward_features(x)
        avg_p = h.mean(dim=(2, 3))
        max_p = torch.amax(h, dim=(2, 3))
        stats = torch.cat([avg_p, max_p], dim=1)
        emb = self.emb_proj(stats)
        logits = self.head(self.drop(emb))
        return logits, emb


class InceptionBlock(nn.Module):
    def __init__(self, ci: int, c1: int, c3: int, c5: int, cp: int) -> None:
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(ci, c1, 1, bias=False), nn.BatchNorm2d(c1), nn.ReLU())
        self.b3 = nn.Sequential(
            nn.Conv2d(ci, c3, 1, bias=False), nn.ReLU(),
            nn.Conv2d(c3, c3, 3, padding=1, bias=False), nn.BatchNorm2d(c3), nn.ReLU(),
        )
        self.b5 = nn.Sequential(
            nn.Conv2d(ci, c5, 1, bias=False), nn.ReLU(),
            nn.Conv2d(c5, c5, 5, padding=2, bias=False), nn.BatchNorm2d(c5), nn.ReLU(),
        )
        self.bp = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(ci, cp, 1, bias=False), nn.BatchNorm2d(cp), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.b1(x), self.b3(x), self.b5(x), self.bp(x)], dim=1)


class GoogLeNetSer(nn.Module):
    def __init__(self, in_ch: int = 1, num_classes: int = 6) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.inc1 = InceptionBlock(24, 12, 16, 12, 8)   # out=48
        self.pool1 = nn.MaxPool2d(2)
        self.inc2 = InceptionBlock(48, 24, 32, 24, 16)  # out=96
        self.pool2 = nn.MaxPool2d(2)
        self.proj = nn.Sequential(nn.Linear(96, 128), nn.ReLU(), nn.Dropout(0.3))
        self.head = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.pool2(self.inc2(self.pool1(self.inc1(self.stem(x)))))
        emb = self.proj(h.mean(dim=(2, 3)))
        return self.head(emb), emb


class Conv3DSer(nn.Module):
    """3D CNN operating on [B, 1, 3, 64, 128] stacked (Log-Mel, Time-Delta, Freq-Delta)."""

    def __init__(self, num_classes: int = 6) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 12, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(12),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(12, 24, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(24),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(24, 48, kernel_size=(3, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(48),
            nn.ReLU(),
        )
        self.proj = nn.Sequential(nn.Linear(48, 128), nn.ReLU(), nn.Dropout(0.3))
        self.head = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)  # [B, 64, 1, H', W']
        emb = self.proj(h.mean(dim=(2, 3, 4)))
        return self.head(emb), emb


# ====================================================================
# Stage B2: Stream Semantic Segmentation & Temporal Event Detection
# ====================================================================

class StreamTemporalEncoder(nn.Module):
    """Collapses 64 frequency bins into a 1D temporal feature map [B, 128, 64] (stride=8)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=(2, 2), padding=1, bias=False),  # [32, 256]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=(2, 2), padding=1, bias=False),  # [16, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=(2, 2), padding=1, bias=False),  # [8, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv2d(x)  # [B, 128, 8, 64]
        return h.mean(dim=2)  # [B, 128, 64]


class SpecUNet1D(nn.Module):
    """Semantic segmentation network predicting frame-level labels [B, 3, 512]."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.freq_proj = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(64, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )  # -> [B, 32, 1, 512]
        self.enc1 = nn.Sequential(nn.Conv1d(32, 48, 5, padding=2), nn.BatchNorm1d(48), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv1d(48, 96, 5, stride=2, padding=2), nn.BatchNorm1d(96), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv1d(96, 128, 5, stride=2, padding=2), nn.BatchNorm1d(128), nn.ReLU())
        self.up2 = nn.ConvTranspose1d(128, 96, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.Sequential(nn.Conv1d(192, 64, 3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose1d(64, 48, kernel_size=4, stride=2, padding=1)
        self.dec1 = nn.Sequential(nn.Conv1d(96, 48, 3, padding=1), nn.ReLU())
        self.head = nn.Conv1d(48, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.freq_proj(x).squeeze(2)  # [B, 32, 512]
        e1 = self.enc1(s0)                 # [B, 48, 512]
        e2 = self.enc2(e1)                 # [B, 96, 256]
        e3 = self.enc3(e2)                 # [B, 128, 128]
        d2 = self.dec2(torch.cat([F.relu(self.up2(e3)), e2], dim=1))
        d1 = self.dec1(torch.cat([F.relu(self.up1(d2)), e1], dim=1))
        return self.head(d1)               # [B, 3, 512]


def roi_align_1d(feat: torch.Tensor, proposals_norm: torch.Tensor, bins: int = 8) -> torch.Tensor:
    """Differentiable 1D RoI-Align over feat [C, T] for proposals [M, 2] in [0, 1]."""
    c, t_len = feat.shape
    m = proposals_norm.shape[0]
    if m == 0:
        return torch.zeros((0, c * bins), device=feat.device, dtype=feat.dtype)
    rel = (torch.arange(bins, device=feat.device, dtype=feat.dtype) + 0.5) / bins  # [bins]
    starts = proposals_norm[:, 0:1] * (t_len - 1)
    ends = proposals_norm[:, 1:2] * (t_len - 1)
    pos = starts + (ends - starts) * rel[None, :]  # [M, bins]
    pos = pos.clamp(0.0, float(t_len - 1))
    lo = torch.floor(pos).long()
    hi = torch.clamp(lo + 1, max=t_len - 1)
    alpha = (pos - lo.float()).unsqueeze(1)  # [M, 1, bins]
    feat_lo = feat[:, lo.reshape(-1)].reshape(c, m, bins).permute(1, 0, 2)
    feat_hi = feat[:, hi.reshape(-1)].reshape(c, m, bins).permute(1, 0, 2)
    pooled = (1.0 - alpha) * feat_lo + alpha * feat_hi  # [M, C, bins]
    return pooled.reshape(m, c * bins)


class RCNN1D(nn.Module):
    """R-CNN adaptation: runs standalone CNN independently on each cropped spectrogram proposal."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.roi_cnn = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.ReLU(),
        )
        self.cls_head = nn.Linear(48, num_classes)
        self.reg_head = nn.Linear(48, 2)

    def forward_crops(self, crops: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.roi_cnn(crops).mean(dim=(2, 3))
        return self.cls_head(h), self.reg_head(h)


class FastRCNN1D(nn.Module):
    """Fast R-CNN adaptation: single shared backbone + RoI-Align over external proposals."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.backbone = StreamTemporalEncoder()
        self.head = nn.Sequential(nn.Linear(128 * 8, 128), nn.ReLU())
        self.cls_head = nn.Linear(128, num_classes)
        self.reg_head = nn.Linear(128, 2)

    def forward_proposals(self, feat_map: torch.Tensor, proposals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        roi = roi_align_1d(feat_map, proposals, bins=8)
        h = self.head(roi)
        return self.cls_head(h), self.reg_head(h)


class FasterRCNN1D(nn.Module):
    """Faster R-CNN adaptation: shared backbone + learned Anchor RPN + RoI-Align detector."""

    ANCHOR_WIDTHS = (0.09, 0.16, 0.24)  # normalized widths (~46, 82, 123 frames)

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.backbone = StreamTemporalEncoder()  # [B, 128, 64]
        n_anchors = len(self.ANCHOR_WIDTHS)
        self.rpn_conv = nn.Sequential(nn.Conv1d(128, 128, 3, padding=1), nn.ReLU())
        self.rpn_obj = nn.Conv1d(128, n_anchors, 1)
        self.rpn_reg = nn.Conv1d(128, n_anchors * 2, 1)
        self.roi_head = nn.Sequential(nn.Linear(128 * 8, 128), nn.ReLU())
        self.cls_head = nn.Linear(128, num_classes)
        self.reg_head = nn.Linear(128, 2)

    def anchors(self, t_len: int, device: torch.device) -> torch.Tensor:
        centers = (torch.arange(t_len, device=device, dtype=torch.float32) + 0.5) / t_len
        anc = []
        for w in self.ANCHOR_WIDTHS:
            c = centers
            ww = torch.full_like(c, w)
            anc.append(torch.stack([c, ww], dim=1))
        return torch.cat(anc, dim=0)  # [3*T, 2]

    def forward_rpn(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)  # [B, 128, 64]
        r = self.rpn_conv(feat)
        b, _, t_len = feat.shape
        obj = self.rpn_obj(r).permute(0, 2, 1).reshape(b, -1)  # [B, 3*T] wait: permute or keep order matching anchors()
        # Match anchors() order [anchor_type outer, time inner]
        obj = self.rpn_obj(r).reshape(b, -1)                   # [B, 3*T]
        reg = self.rpn_reg(r).reshape(b, len(self.ANCHOR_WIDTHS), 2, t_len).permute(0, 1, 3, 2).reshape(b, -1, 2)
        return feat, obj, reg


class YOLO1D(nn.Module):
    """Single-stage YOLO-style temporal grid detector over 16 cells (predicts in 1 pass)."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.backbone = StreamTemporalEncoder()  # [B, 128, 64]
        self.grid_conv = nn.Sequential(
            nn.Conv1d(128, 128, 3, stride=2, padding=1), nn.ReLU(),  # 32
            nn.Conv1d(128, 128, 3, stride=2, padding=1), nn.ReLU(),  # 16 cells
        )
        # Per cell: [obj, offset_c (0..1 within cell), log_w, cls_logits (num_classes)]
        self.pred = nn.Conv1d(128, 3 + num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.grid_conv(self.backbone(x))  # [B, 128, 16]
        return self.pred(g).permute(0, 2, 1)  # [B, 16, 3 + C]


class RetinaNet1D(nn.Module):
    """Single-stage multi-scale FPN anchor detector trained with Focal Loss."""

    ANCHOR_WIDTHS = (0.10, 0.18)

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.backbone = StreamTemporalEncoder()  # [B, 128, 64]
        self.p4_down = nn.Conv1d(128, 128, 3, stride=2, padding=1)  # [B, 128, 32]
        self.cls_net = nn.Sequential(nn.Conv1d(128, 96, 3, padding=1), nn.ReLU(), nn.Conv1d(96, 2 * num_classes, 1))
        self.box_net = nn.Sequential(nn.Conv1d(128, 96, 3, padding=1), nn.ReLU(), nn.Conv1d(96, 2 * 2, 1))
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p3 = self.backbone(x)                  # [B, 128, 64]
        p4 = F.relu(self.p4_down(p3))          # [B, 128, 32]
        cls_list, box_list, anc_list = [], [], []
        b = x.shape[0]
        for feat in (p3, p4):
            t_len = feat.shape[2]
            c_out = self.cls_net(feat).reshape(b, 2, self.num_classes, t_len).permute(0, 1, 3, 2).reshape(b, -1, self.num_classes)
            b_out = self.box_net(feat).reshape(b, 2, 2, t_len).permute(0, 1, 3, 2).reshape(b, -1, 2)
            centers = (torch.arange(t_len, device=x.device, dtype=torch.float32) + 0.5) / t_len
            for w in self.ANCHOR_WIDTHS:
                anc_list.append(torch.stack([centers, torch.full_like(centers, w)], dim=1))
            cls_list.append(c_out)
            box_list.append(b_out)
        return torch.cat(cls_list, dim=1), torch.cat(box_list, dim=1), torch.cat(anc_list, dim=0)


# ====================================================================
# Stage C: Temporal & Conversational Context Models
# ====================================================================

class ContextLstm(nn.Module):
    def __init__(self, in_dim: int = 130, hidden: int = 96, num_classes: int = 6, bidirectional: bool = True) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2))
        self.lstm = nn.LSTM(hidden, hidden, num_layers=1, batch_first=True, bidirectional=bidirectional)
        out_d = hidden * (2 if bidirectional else 1)
        self.cls_head = nn.Linear(out_d, num_classes)
        self.esc_head = nn.Linear(out_d, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, _ = self.lstm(self.proj(x))
        return self.cls_head(h), torch.sigmoid(self.esc_head(h))


class ExplicitAttentionBlock(nn.Module):
    """ONNX-friendly explicit multi-head self-attention returning attention weights."""

    def __init__(self, d_model: int = 128, n_heads: int = 4, dropout: float = 0.2, causal: bool = True) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if self.causal:
            mask = torch.triu(torch.ones((t, t), device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask[None, None, :, :], -1e4)
        attn = torch.softmax(scores, dim=-1)  # [B, H, T, T]
        ctx = torch.matmul(self.drop(attn), v).transpose(1, 2).reshape(b, t, d)
        x = self.norm1(x + self.drop(self.out_proj(ctx)))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x, attn.mean(dim=1)  # mean over heads -> [B, T, T]


class ContextTransformer(nn.Module):
    """ONNX-exportable Conversational Transformer predicting turn emotions & escalation prob."""

    def __init__(
        self, in_dim: int = 130, d_model: int = 128, n_heads: int = 4,
        num_layers: int = 2, num_classes: int = 6, causal: bool = True, max_len: int = 64
    ) -> None:
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(in_dim, d_model), nn.LayerNorm(d_model), nn.ReLU())
        pe = torch.zeros(1, max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        self.layers = nn.ModuleList([ExplicitAttentionBlock(d_model, n_heads, 0.2, causal) for _ in range(num_layers)])
        self.cls_head = nn.Linear(d_model, num_classes)
        self.esc_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = x.shape[1]
        h = self.in_proj(x) + self.pe[:, :t, :]
        last_attn = None
        for layer in self.layers:
            h, last_attn = layer(h)
        return self.cls_head(h), torch.sigmoid(self.esc_head(h)), last_attn


# ====================================================================
# Stage D: Generative Spectrogram Augmentation (CVAE & cGAN)
# ====================================================================

class SpecCVAE(nn.Module):
    """Conditional VAE generating normalized 64x128 Log-Mel spectrograms."""

    def __init__(self, latent_dim: int = 16, num_classes: int = 6) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1), nn.ReLU(),   # [16, 32, 64]
            nn.Conv2d(16, 32, 4, stride=2, padding=1), nn.ReLU(),  # [32, 16, 32]
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),  # [64, 8, 16]
        )
        self.fc_mu = nn.Linear(64 * 8 * 16 + num_classes, latent_dim)
        self.fc_logvar = nn.Linear(64 * 8 * 16 + num_classes, latent_dim)
        self.fc_dec = nn.Linear(latent_dim + num_classes, 64 * 8 * 16)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
        )

    def encode(self, x: torch.Tensor, y_onehot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x).flatten(1)
        hc = torch.cat([h, y_onehot], dim=1)
        return self.fc_mu(hc), self.fc_logvar(hc)

    def decode(self, z: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc_dec(torch.cat([z, y_onehot], dim=1))).reshape(-1, 64, 8, 16)
        return self.dec(h)

    def forward(self, x: torch.Tensor, y_onehot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, y_onehot)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.decode(z, y_onehot), mu, logvar


class SpecCGANGenerator(nn.Module):
    def __init__(self, noise_dim: int = 32, num_classes: int = 6) -> None:
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(noise_dim + num_classes, 64 * 8 * 16), nn.ReLU())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
        )

    def forward(self, z: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
        h = self.fc(torch.cat([z, y_onehot], dim=1)).reshape(-1, 64, 8, 16)
        return self.net(h)


class SpecCGANDiscriminator(nn.Module):
    def __init__(self, num_classes: int = 6) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(1 + num_classes, 16, 4, stride=2, padding=1)), nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(16, 32, 4, stride=2, padding=1)), nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(32, 64, 4, stride=2, padding=1)), nn.LeakyReLU(0.2),
        )
        self.head = nn.utils.spectral_norm(nn.Linear(64, 1))

    def forward(self, x: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
        y_map = y_onehot[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        h = self.conv(torch.cat([x, y_map], dim=1)).mean(dim=(2, 3))
        return self.head(h)
