import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.tal import dist2bbox, make_anchors

__all__ = ["Detect_AFSFF"]


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).
    Proposed in Generalized Focal Loss: https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class FeatureResize(nn.Module):
    """Resize one feature map to target level size/channels."""

    def __init__(self, c1, c2, src_level, tgt_level):
        super().__init__()
        delta = tgt_level - src_level
        self.mode = "same"

        if delta == 0:
            self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        elif delta < 0:
            self.mode = "up"
            self.proj = Conv(c1, c2, 1, 1)
        elif delta == 1:
            self.mode = "down2"
            self.proj = Conv(c1, c2, 3, 2)
        elif delta == 2:
            self.mode = "down4"
            self.proj = Conv(c1, c2, 3, 2)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        elif delta == 3:
            self.mode = "down8"
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.proj = Conv(c1, c2, 3, 2)
        else:
            raise ValueError(f"Unsupported level gap: src={src_level}, tgt={tgt_level}")

    def forward(self, x, target_size):
        if self.mode == "same":
            y = self.proj(x)
        elif self.mode == "up":
            y = F.interpolate(self.proj(x), size=target_size, mode="nearest")
        elif self.mode == "down2":
            y = self.proj(x)
        elif self.mode == "down4":
            y = self.pool(self.proj(x))
        else:
            y = self.proj(self.pool2(self.pool1(x)))

        if y.shape[-2:] != target_size:
            y = F.interpolate(y, size=target_size, mode="nearest")
        return y


class SFFM(nn.Module):
    """Spatial Feature Fusion Module."""

    def __init__(self, n_inputs=4):
        super().__init__()
        self.n_inputs = n_inputs
        self.pwconv = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def _spatial_attention(self, x):
        ap = torch.mean(x, dim=1, keepdim=True)
        mp = torch.max(x, dim=1, keepdim=True)[0]
        return self.pwconv(ap + mp)

    def forward(self, feats):
        if len(feats) != self.n_inputs:
            raise ValueError(f"SFFM expects {self.n_inputs} inputs, but got {len(feats)}.")
        logits = torch.cat([self._spatial_attention(f) for f in feats], dim=1)
        weights = F.softmax(logits, dim=1)
        fused = sum(f * weights[:, i : i + 1] for i, f in enumerate(feats))
        return fused, weights


class AdaptiveFusionLevel(nn.Module):
    """Fuse {P2->k, P3->k, P4->k, P5->k} into F_k."""

    def __init__(self, level, ch):
        super().__init__()
        self.level = level
        self.resizers = nn.ModuleList(FeatureResize(ch[i], ch[level], i, level) for i in range(len(ch)))
        self.sffm = SFFM(n_inputs=len(ch))

    def forward(self, x):
        target_size = x[self.level].shape[-2:]
        resized = [op(feat, target_size) for op, feat in zip(self.resizers, x)]
        fused, _ = self.sffm(resized)
        return fused


class Detect_AFSFF(nn.Module):
    """YOLOv8 Detect head with adaptive feature fusion and SFFM."""

    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)

        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        self.fusions = nn.ModuleList(AdaptiveFusionLevel(i, ch) for i in range(self.nl))

    def forward(self, x):
        x = [fusion(x) for fusion in self.fusions]  # F2, F3, F4, F5
        shape = x[0].shape
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        if self.export and self.format in ("tflite", "edgetpu"):
            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
            dbox /= img_size

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        m = self
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)
