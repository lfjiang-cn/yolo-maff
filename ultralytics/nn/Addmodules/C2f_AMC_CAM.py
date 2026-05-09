import torch
import torch.nn as nn

__all__ = ["AMC_CAM", "Bottleneck_AMC_CAM", "C2f_AMC_CAM"]


def autopad(k, p=None, d=1):  # kernel, padding, dilation
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


class AMC_CAM(nn.Module):
    """
    Aggregated Multi-scale Context Channel Attention Module.
    Returns refined features by default: F' = F * sigmoid(L(F) + G(F)).
    """

    def __init__(self, channels, reduction=16, dilations=(1, 2, 3)):
        super().__init__()
        if isinstance(dilations, int):
            dilations = (dilations,)
        if len(dilations) == 0:
            raise ValueError("dilations must contain at least one element.")

        inter_channels = max(channels // reduction, 1)
        local_channels = max(channels // len(dilations), 1)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.global_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            # GAP outputs 1x1 spatial maps; GroupNorm is stable for batch=1 training.
            nn.GroupNorm(1, channels),
        )

        self.local_reduce = nn.Sequential(
            nn.Conv2d(channels, local_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(local_channels),
            nn.ReLU(inplace=True),
        )
        self.local_branches = nn.ModuleList(
            nn.Conv2d(
                local_channels,
                local_channels,
                kernel_size=3,
                stride=1,
                padding=d,
                dilation=d,
                bias=False,
            )
            for d in dilations
        )

        local_cat_channels = local_channels * len(dilations)
        self.local_fuse = nn.Sequential(
            nn.BatchNorm2d(local_cat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(local_cat_channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def attention(self, x):
        """Return attention map W(F) with shape [B, C, H, W]."""
        g = self.global_att(self.gap(x))
        l = self.local_reduce(x)
        l = torch.cat([branch(l) for branch in self.local_branches], dim=1)
        l = self.local_fuse(l)
        return self.sigmoid(l + g)

    def forward(self, x, return_weight=False):
        w = self.attention(x)
        return w if return_weight else x * w


class Bottleneck_AMC_CAM(nn.Module):
    """Bottleneck block using AMC-CAM guided residual fusion."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, reduction=16, dilations=(1, 2, 3)):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.attn = AMC_CAM(c2, reduction=reduction, dilations=dilations)

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        if not self.add:
            return y
        m = self.attn.attention(x + y)
        return m * x + (1.0 - m) * y


class C2f_AMC_CAM(nn.Module):
    """C2f with AMC-CAM-guided bottlenecks."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, reduction=16, dilations=(1, 2, 3)):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck_AMC_CAM(
                self.c,
                self.c,
                shortcut=shortcut,
                g=g,
                k=((3, 3), (3, 3)),
                e=1.0,
                reduction=reduction,
                dilations=dilations,
            )
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


if __name__ == "__main__":
    x = torch.randn(1, 64, 32, 32)
    model = C2f_AMC_CAM(64, 64, n=2, shortcut=True)
    print(model(x).shape)
