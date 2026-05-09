"""CNN heads run at inference: tip → base → bow/stern.

All three consume the same DINOv3 ViT-S/16 stride-8 feature map.

  TileHead
      Heatmap of mast-tip pixels across the whole image (every visible boat).

  TipAttnBaseHead
      Given a 21x34 feature window centred on a tip cluster, predicts the
      mast-base offset. Tip is at fixed position (row=2, col=10).

  TipBaseBowSternAttnHead
      Same backbone as TipAttnBaseHead with two conditioning tokens
      (tip + base) and two output channels (bow, stern).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_2d_pos_enc(H: int, W: int, dim: int) -> torch.Tensor:
    """Sin/cos 2D positional encoding, shape (H, W, dim)."""
    assert dim % 4 == 0, "dim must be divisible by 4 for 2D sin/cos"
    pe = torch.zeros(H, W, dim)
    half = dim // 2
    div = torch.exp(torch.arange(0, half, 2).float()
                    * (-math.log(10000.0) / half))
    pos_y = torch.arange(H).float().unsqueeze(1)
    pos_x = torch.arange(W).float().unsqueeze(1)
    sin_y, cos_y = torch.sin(pos_y * div), torch.cos(pos_y * div)
    sin_x, cos_x = torch.sin(pos_x * div), torch.cos(pos_x * div)
    pe[..., :half] = torch.cat([sin_y, cos_y], -1).unsqueeze(1).expand(-1, W, -1)
    pe[..., half:] = torch.cat([sin_x, cos_x], -1).unsqueeze(0).expand(H, -1, -1)
    return pe


class TileHead(nn.Module):
    """Per-cell tip heatmap: 4 conv blocks → 1-channel logit."""

    def __init__(self, in_dim: int = 768, hidden: int = 64):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, hidden, 1)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.c1 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.c2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden)
        self.c3 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.out = nn.Conv2d(hidden, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.proj(x)))
        x = F.relu(self.bn2(self.c1(x)))
        x = F.relu(self.bn3(self.c2(x)))
        x = F.relu(self.c3(x))
        return self.out(x)


class TipAttnBaseHead(nn.Module):
    """Tip-conditioned transformer that predicts the mast base offset.

    Input: feature window (B, D, win_h, win_w) with the tip at (tip_row, tip_col).
    Output: (B, 1, win_h, win_w) per-cell logit for the base location.
    """

    def __init__(self, in_dim: int = 384, hidden: int = 128,
                 n_heads: int = 4, n_layers: int = 2,
                 win_h: int = 34, win_w: int = 21,
                 tip_row: int = 2, tip_col: int = 10):
        super().__init__()
        self.win_h, self.win_w = win_h, win_w
        self.tip_row, self.tip_col = tip_row, tip_col
        self.proj = nn.Conv2d(in_dim, hidden, 1)
        self.norm_in = nn.LayerNorm(hidden)
        pe = build_2d_pos_enc(win_h, win_w, hidden)
        self.register_buffer("pos_enc", pe.view(win_h * win_w, hidden))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden, nhead=n_heads,
                dim_feedforward=hidden * 2,
                dropout=0.0, activation="gelu",
                batch_first=True, norm_first=True)
            for _ in range(n_layers)])
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        B, _, H, W = feats.shape
        assert (H, W) == (self.win_h, self.win_w)
        x = self.proj(feats).permute(0, 2, 3, 1).reshape(B, H * W, -1)
        x = self.norm_in(x + self.pos_enc.unsqueeze(0))
        for blk in self.blocks:
            x = blk(x)
        tip_idx = self.tip_row * self.win_w + self.tip_col
        tip_feat = x[:, tip_idx, :]
        combined = torch.cat([x, tip_feat.unsqueeze(1).expand(-1, H * W, -1)], -1)
        return self.out_proj(combined).view(B, H, W).unsqueeze(1)


class TipBaseBowSternAttnHead(nn.Module):
    """Same backbone as TipAttnBaseHead, conditioned on (tip + base) and
    predicting two channels (bow, stern).

    Input:
        feats    : (B, D, win_h, win_w)
        base_idx : (B,)  cell index of the base inside the window
    Output:
        (B, 2, win_h, win_w)  channel 0 = bow, channel 1 = stern
    """

    def __init__(self, in_dim: int = 384, hidden: int = 128,
                 n_heads: int = 4, n_layers: int = 2,
                 win_h: int = 34, win_w: int = 21,
                 tip_row: int = 2, tip_col: int = 10,
                 n_out: int = 2):
        super().__init__()
        self.win_h, self.win_w = win_h, win_w
        self.tip_row, self.tip_col = tip_row, tip_col
        self.n_out = n_out
        self.proj = nn.Conv2d(in_dim, hidden, 1)
        self.norm_in = nn.LayerNorm(hidden)
        pe = build_2d_pos_enc(win_h, win_w, hidden)
        self.register_buffer("pos_enc", pe.view(win_h * win_w, hidden))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden, nhead=n_heads,
                dim_feedforward=hidden * 2,
                dropout=0.0, activation="gelu",
                batch_first=True, norm_first=True)
            for _ in range(n_layers)])
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_out))

    def forward(self, feats: torch.Tensor,
                base_idx: torch.Tensor) -> torch.Tensor:
        B, _, H, W = feats.shape
        x = self.proj(feats).permute(0, 2, 3, 1).reshape(B, H * W, -1)
        x = self.norm_in(x + self.pos_enc.unsqueeze(0))
        for blk in self.blocks:
            x = blk(x)
        D = x.size(-1)
        tip_idx = self.tip_row * self.win_w + self.tip_col
        tip_feat = x[:, tip_idx, :]
        base_feat = x.gather(1, base_idx.view(B, 1, 1).expand(B, 1, D)).squeeze(1)
        cond = (tip_feat + base_feat) * 0.5
        combined = torch.cat([x, cond.unsqueeze(1).expand(-1, H * W, -1)], -1)
        logit = self.out_proj(combined).permute(0, 2, 1)
        return logit.reshape(B, self.n_out, H, W)
