"""
models/fsha_net.py
------------------
FSHA-Net: Full-Scale Hybrid Attention Network

Key design decisions
--------------------
1. U-Net 3+ style full-scale skip connections:
   Every decoder node receives features from ALL encoder scales.

2. Attention gates on every skip connection:
   Soft attention weights multiply into skip features BEFORE concatenation,
   suppressing background and focusing on vessel regions.

3. Deep supervision:
   Three output heads at different decoder scales, all upsampled to 256x256.
   Training loss = 0.5*L_main + 0.3*L_aux1 + 0.2*L_aux2

Returns tuple (main, aux1, aux2) in training mode.
Returns main logit only in eval mode.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ────────────────────────────────────────────────────────────

class CBR(nn.Module):
    """Conv -> BN -> ReLU (single)."""
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x): return self.b(x)


class DoubleCBR(nn.Module):
    """Two consecutive CBR blocks."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.b = nn.Sequential(CBR(in_ch, out_ch), CBR(out_ch, out_ch))
    def forward(self, x): return self.b(x)


# ── Attention Gate ─────────────────────────────────────────────────────────────

class AttentionGate(nn.Module):
    """
    Additive soft attention gate (Oktay et al., 2018).

    Computes alpha = sigmoid( Wx(skip) + Wg(gate) )
    Output = skip * alpha   (attended skip feature)

    skip_ch : channels of the encoder skip feature
    gate_ch : channels of the gating signal from decoder
    """
    def __init__(self, skip_ch, gate_ch, inter_ch=None):
        super().__init__()
        inter_ch = inter_ch or (skip_ch // 2)
        self.Wx  = nn.Sequential(
            nn.Conv2d(skip_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.Wg  = nn.Sequential(
            nn.Conv2d(gate_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=False),
            nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_skip, g):
        # Align spatial size of gating signal to skip feature
        if g.shape[2:] != x_skip.shape[2:]:
            g = F.interpolate(g, size=x_skip.shape[2:],
                              mode="bilinear", align_corners=True)
        alpha = self.psi(self.relu(self.Wx(x_skip) + self.Wg(g)))  # (B,1,H,W)
        return x_skip * alpha                                        # attended skip


# ── Encoder ────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """Standard 5-level encoder returning feature maps at every scale."""
    def __init__(self, in_ch=3, f=64):
        super().__init__()
        self.e1 = DoubleCBR(in_ch, f)
        self.e2 = nn.Sequential(nn.MaxPool2d(2), DoubleCBR(f,    f*2))
        self.e3 = nn.Sequential(nn.MaxPool2d(2), DoubleCBR(f*2,  f*4))
        self.e4 = nn.Sequential(nn.MaxPool2d(2), DoubleCBR(f*4,  f*8))
        self.bn = nn.Sequential(nn.MaxPool2d(2), DoubleCBR(f*8,  f*16))

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        b  = self.bn(e4)
        return e1, e2, e3, e4, b


# ── Full-scale decoder node ────────────────────────────────────────────────────

class FSDecodeNode(nn.Module):
    """
    U-Net 3+ decoder node.

    Receives ALL encoder skip features (with attention) + gating signal,
    resizes every stream to target_size, projects to node_ch channels each,
    then fuses them with a DoubleCBR.
    """
    def __init__(self, enc_chs, gate_ch, node_ch, target_size):
        super().__init__()
        self.target = target_size
        n = len(enc_chs) + 1          # number of streams (enc + gate)

        # One attention gate per encoder skip
        self.atts  = nn.ModuleList([
            AttentionGate(ch, gate_ch, node_ch // 2) for ch in enc_chs])

        # Project each attended skip to node_ch
        self.projs = nn.ModuleList([
            CBR(ch, node_ch, k=1, p=0) for ch in enc_chs])

        # Project gate stream to node_ch
        self.g_proj = CBR(gate_ch, node_ch, k=1, p=0)

        # Fuse all streams
        self.fuse   = DoubleCBR(n * node_ch, node_ch)

    def forward(self, skips, gate):
        streams = []
        for feat, att, proj in zip(skips, self.atts, self.projs):
            attended = att(feat, gate)
            resized  = F.interpolate(attended, size=self.target,
                                     mode="bilinear", align_corners=True)
            streams.append(proj(resized))

        g_up = F.interpolate(gate, size=self.target,
                             mode="bilinear", align_corners=True)
        streams.append(self.g_proj(g_up))
        return self.fuse(torch.cat(streams, dim=1))


# ── FSHA-Net ───────────────────────────────────────────────────────────────────

class FSHANet(nn.Module):
    """
    Full-Scale Hybrid Attention Network.

    Encoder  : 5 levels (64 -> 128 -> 256 -> 512 -> 1024)
    Decoder  : 4 FSDecodeNodes, each aggregating all 4 encoder skips
    Heads    : main (256x256), aux1 (from d2), aux2 (from d3)
    """
    def __init__(self, in_ch=3, out_ch=1, base_f=64):
        super().__init__()
        f = base_f
        self.encoder = Encoder(in_ch, f)

        enc_chs = [f, f*2, f*4, f*8]   # channels of the 4 encoder skips
        bot_ch  = f * 16               # bottleneck channels

        # Decoder nodes (coarse -> fine)
        self.d4 = FSDecodeNode(enc_chs, bot_ch, f*8,  (32,  32))
        self.d3 = FSDecodeNode(enc_chs, f*8,   f*4,  (64,  64))
        self.d2 = FSDecodeNode(enc_chs, f*4,   f*2,  (128, 128))
        self.d1 = FSDecodeNode(enc_chs, f*2,   f,    (256, 256))

        # Output heads
        self.out_main = nn.Conv2d(f,    out_ch, 1)   # finest scale
        self.out_aux1 = nn.Conv2d(f*2,  out_ch, 1)   # medium scale
        self.out_aux2 = nn.Conv2d(f*4,  out_ch, 1)   # coarse scale

    def forward(self, x):
        H, W = x.shape[2:]
        e1, e2, e3, e4, b = self.encoder(x)
        skips = [e1, e2, e3, e4]

        d4 = self.d4(skips, b)
        d3 = self.d3(skips, d4)
        d2 = self.d2(skips, d3)
        d1 = self.d1(skips, d2)

        main = self.out_main(d1)
        aux1 = F.interpolate(self.out_aux1(d2), (H, W),
                             mode="bilinear", align_corners=True)
        aux2 = F.interpolate(self.out_aux2(d3), (H, W),
                             mode="bilinear", align_corners=True)

        # Return tuple in training (needed for deep supervision loss)
        # Return only main logit in eval (clean interface for metrics)
        if self.training:
            return main, aux1, aux2
        return main
