"""
sod_model.py
All the model architecture code lives here.

We have two models:
  - BaselineSODNet: a plain encoder-decoder, no skip connections. Simple reference point.
  - ImprovedSODNet: adds ASPP, attention gates, and deeper conv blocks. This is the main model.

The loss function (CombinedLoss) combines three terms — Focal BCE, IoU, and SSIM —
because each one catches a different type of prediction error.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Basic building block ──────────────────────────────────────────────────────

class ConvBNReLU(nn.Module):
    """
    The standard Conv -> BatchNorm -> ReLU sequence we use everywhere.
    BatchNorm is kept separate from bias (bias=False) since BN has its own
    learnable shift parameter that does the same job.
    Supports dilated convolutions for the ASPP module.
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ── Baseline model ────────────────────────────────────────────────────────────

class BaselineSODNet(nn.Module):
    """
    A straightforward encoder-decoder without skip connections.

    The encoder compresses the image down to an 8x8 feature map,
    and the decoder gradually upsamples it back to the original size.
    Without skip connections, fine spatial details are permanently lost
    at the bottleneck — this is the main limitation of the baseline.

    We still use BatchNorm and Dropout to keep training stable and reduce overfitting.
    """
    def __init__(self, img_size=128):
        super().__init__()
        self.img_size = img_size

        def _enc(ic, oc):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1, bias=False),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True),
            )

        self.enc1, self.enc2 = _enc(3, 64),   _enc(64,  128)
        self.enc3, self.enc4 = _enc(128, 256), _enc(256, 512)
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout here because the model has 4.6M params trained on ~700 images
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
        )

        def _dec(ic, oc):
            return nn.Sequential(
                nn.ConvTranspose2d(ic, oc, 2, stride=2, bias=False),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True),
            )

        self.dec4, self.dec3 = _dec(512, 256), _dec(256, 128)
        self.dec2, self.dec1 = _dec(128, 64),  _dec(64,  32)

        # 1x1 conv collapses channels to a single saliency probability map
        self.head = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Sigmoid())
        self._init_weights()

    def _init_weights(self):
        # Kaiming init works well with ReLU activations
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        return self.head(self.dec1(self.dec2(self.dec3(self.dec4(b)))))


# ── Improved model components ─────────────────────────────────────────────────

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.

    The bottleneck feature map is only 8x8, so a regular 3x3 conv has a tiny
    receptive field relative to the original image. ASPP fixes this by running
    parallel dilated convolutions with different dilation rates (1, 2, 4),
    each seeing context at a different scale. A global average pooling branch
    captures the overall image-level context. All four outputs are then fused.

    This gives the model a much better understanding of what's salient at
    the whole-image level, not just locally.
    """
    def __init__(self, in_ch, out_ch, rates=(1, 2, 4)):
        super().__init__()
        self.branches = nn.ModuleList([
            ConvBNReLU(in_ch, out_ch, kernel_size=3, padding=r, dilation=r)
            for r in rates
        ])
        # The global average branch compresses everything to a single vector
        # then upsamples it back — gives the model a "big picture" view
        self.gap_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        n = len(rates) + 1
        self.fusion = nn.Sequential(
            nn.Conv2d(out_ch * n, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        h, w  = x.shape[2:]
        gap   = F.interpolate(self.gap_branch(x), size=(h, w), mode="bilinear", align_corners=False)
        feats = [b(x) for b in self.branches] + [gap]
        return self.fusion(torch.cat(feats, dim=1))


class AttentionGate(nn.Module):
    """
    Attention gate for skip connections.

    When we merge encoder features with decoder features (skip connections),
    not all encoder features are useful — some regions are just background.
    This module learns to produce a soft weight map (alpha) that highlights
    which spatial locations in the skip feature are relevant, based on what
    the decoder is currently looking for (the gating signal g).

    Result: the skip features are filtered before being merged, which
    reduces noise and helps the model focus on salient regions.
    """
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.W_g  = nn.Conv2d(gate_ch, inter_ch, 1, bias=False)
        self.W_x  = nn.Conv2d(skip_ch, inter_ch, 1, bias=False)
        self.psi  = nn.Sequential(nn.Conv2d(inter_ch, 1, 1), nn.Sigmoid())
        self.bn_g = nn.BatchNorm2d(inter_ch)
        self.bn_x = nn.BatchNorm2d(inter_ch)

    def forward(self, g, x):
        # Make sure spatial sizes match before adding — rounding can cause off-by-one
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)
        alpha = self.psi(F.relu(self.bn_g(self.W_g(g)) + self.bn_x(self.W_x(x))))
        return x * alpha  # suppress irrelevant spatial locations


class DoubleConv(nn.Module):
    """
    Two back-to-back ConvBNReLU layers.
    Having two convolutions per stage lets the model learn richer features
    before each pooling/upsampling step. Same idea as VGG blocks.
    """
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [ConvBNReLU(in_ch, out_ch), ConvBNReLU(out_ch, out_ch)]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ── Improved model ────────────────────────────────────────────────────────────

class ImprovedSODNet(nn.Module):
    """
    The improved model. Builds on the baseline with four key additions:

    1. Skip connections: feed encoder features directly to the decoder so
       spatial detail isn't permanently lost at the bottleneck.

    2. ASPP bottleneck: captures multi-scale context at the 8x8 feature map
       where a plain conv would only see a tiny neighbourhood.

    3. Attention gates: filter the skip features before merging them with the
       decoder path, so background clutter doesn't pollute the predictions.

    4. Double conv blocks per encoder stage: deeper feature extraction
       before each MaxPool step.
    """

    def __init__(self, img_size=128, dropout_rate=0.3):
        super().__init__()
        self.img_size = img_size

        # Encoder: each stage doubles the channels and halves the spatial size
        self.enc1 = DoubleConv(3,   32)
        self.enc2 = DoubleConv(32,  64)
        self.enc3 = DoubleConv(64,  128)
        self.enc4 = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2, 2)

        # ASPP replaces a plain bottleneck conv — much better at capturing context
        self.bottleneck = nn.Sequential(
            ASPP(256, 256, rates=(1, 2, 4)),
            nn.Dropout2d(dropout_rate),
        )

        # One attention gate per skip connection, sized to match the channels at each level
        self.att4 = AttentionGate(gate_ch=128, skip_ch=256, inter_ch=64)
        self.att3 = AttentionGate(gate_ch=64,  skip_ch=128, inter_ch=32)
        self.att2 = AttentionGate(gate_ch=32,  skip_ch=64,  inter_ch=16)
        self.att1 = AttentionGate(gate_ch=16,  skip_ch=32,  inter_ch=8)

        # Decoder: upsample, then concat with the attended skip, then refine
        self.up4  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec4 = DoubleConv(128 + 256, 128, dropout=dropout_rate)  # 128 up + 256 skip

        self.up3  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = DoubleConv(64 + 128, 64, dropout=dropout_rate)    # 64 up + 128 skip

        self.up2  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = DoubleConv(32 + 64, 32)                           # 32 up + 64 skip

        self.up1  = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = DoubleConv(16 + 32, 16)                           # 16 up + 32 skip

        self.head = nn.Sequential(nn.Conv2d(16, 1, 1), nn.Sigmoid())
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Encode: progressively shrink spatial size, grow channels
        e1 = self.enc1(x)             # (B, 32,  H,    W)
        e2 = self.enc2(self.pool(e1)) # (B, 64,  H/2,  W/2)
        e3 = self.enc3(self.pool(e2)) # (B, 128, H/4,  W/4)
        e4 = self.enc4(self.pool(e3)) # (B, 256, H/8,  W/8)

        b = self.bottleneck(self.pool(e4)) # (B, 256, H/16, W/16)

        # Decode: upsample, then concatenate with the attended skip connection
        u4 = self.up4(b)
        d4 = self.dec4(torch.cat([u4, self.att4(u4, e4)], dim=1))

        u3 = self.up3(d4)
        d3 = self.dec3(torch.cat([u3, self.att3(u3, e3)], dim=1))

        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, self.att2(u2, e2)], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, self.att1(u1, e1)], dim=1))

        return self.head(d1) # (B, 1, H, W) — saliency probability map


# ── Loss function ─────────────────────────────────────────────────────────────

class _SSIMLoss(nn.Module):
    """
    Structural Similarity loss computed with a Gaussian window.

    Plain BCE treats every pixel independently. SSIM measures local structure
    (means, variances, correlations) which makes it sensitive to how well the
    predicted mask matches the shape and texture of the ground truth — not
    just whether individual pixels are correct.

    We return 1 - SSIM so it behaves like a regular loss (lower = better).
    """
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        kernel = self._gaussian_kernel(window_size, sigma).view(1, 1, window_size, window_size)
        self.register_buffer("kernel", kernel)  # moves to GPU automatically with the model
        self.pad = window_size // 2
        self.C1  = 0.01 ** 2  # small stability constants from the original SSIM paper
        self.C2  = 0.03 ** 2

    @staticmethod
    def _gaussian_kernel(size, sigma):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.unsqueeze(1) @ g.unsqueeze(0)

    def forward(self, pred, target):
        k   = self.kernel.to(pred.device)
        mu1 = F.conv2d(pred,   k, padding=self.pad)
        mu2 = F.conv2d(target, k, padding=self.pad)

        mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
        s1  = F.conv2d(pred   * pred,   k, padding=self.pad) - mu1_sq
        s2  = F.conv2d(target * target, k, padding=self.pad) - mu2_sq
        s12 = F.conv2d(pred   * target, k, padding=self.pad) - mu12

        ssim_map = ((2 * mu12 + self.C1) * (2 * s12 + self.C2)) / \
                   ((mu1_sq + mu2_sq + self.C1) * (s1 + s2 + self.C2))
        return 1.0 - ssim_map.mean()


class CombinedLoss(nn.Module):
    """
    Three-term loss designed specifically for saliency detection:

      Focal BCE  — handles class imbalance. Most pixels are background,
                   so plain BCE gets dominated by easy negatives. Focal loss
                   down-weights those easy pixels and focuses on the hard ones.

      IoU loss   — directly optimises the metric we care about. BCE doesn't
                   know anything about how well the predicted region overlaps
                   with the ground truth mask as a whole.

      SSIM loss  — enforces structural similarity. Keeps the predicted mask
                   sharp and coherent rather than blurry or noisy.

    Total: L = focal_BCE + 1.0 * (1 - IoU) + 0.5 * (1 - SSIM)
    """

    def __init__(self, focal_gamma=2.0, iou_weight=1.0, ssim_weight=0.5, eps=1e-6):
        super().__init__()
        self.focal_gamma = focal_gamma
        self.iou_weight  = iou_weight
        self.ssim_weight = ssim_weight
        self.eps         = eps
        self._ssim       = _SSIMLoss()

    def _focal_bce(self, pred, target):
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        p_t = pred * target + (1.0 - pred) * (1.0 - target)
        return (((1.0 - p_t) ** self.focal_gamma) * bce).mean()

    def _iou_loss(self, pred, target):
        inter = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
        return (1.0 - (inter + self.eps) / (union + self.eps)).mean()

    def forward(self, pred, target):
        focal = self._focal_bce(pred, target)
        iou   = self._iou_loss(pred, target)
        ssim  = self._ssim(pred, target)
        return focal + self.iou_weight * iou + self.ssim_weight * ssim


# ── Helpers ───────────────────────────────────────────────────────────────────

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(model_type="baseline", img_size=128, dropout_rate=0.3):
    if model_type == "improved":
        return ImprovedSODNet(img_size=img_size, dropout_rate=dropout_rate)
    return BaselineSODNet(img_size=img_size)


if __name__ == "__main__":
    for name, ModelCls in [("baseline", BaselineSODNet), ("improved", ImprovedSODNet)]:
        m   = ModelCls(img_size=128)
        x   = torch.randn(2, 3, 128, 128)
        out = m(x)
        assert out.shape == (2, 1, 128, 128), f"Bad shape: {out.shape}"
        assert 0.0 <= out.min() and out.max() <= 1.0
        print(f"{name:10s}  output={tuple(out.shape)}  params={count_parameters(m):,}")

    loss_fn = CombinedLoss()
    pred    = torch.sigmoid(torch.randn(2, 1, 128, 128))
    target  = (torch.rand(2, 1, 128, 128) > 0.5).float()
    print(f"\nCombinedLoss = {loss_fn(pred, target).item():.4f}")
    print("sod_model.py OK")
