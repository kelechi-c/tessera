"""Strictly tile-local nonlinear residual encoder (`RectTileDown`) and active-tile canvas builder.

Pairs with `tessera.latent.rect` (`2:1` k-d tree rectangular layouts, `SHAPES`, `COARSE_SHAPES`):
  - `RectTileDown`: zero-initialized CoordConv residual on top of the orthonormal linear projection
    (`project_rect_tokens`), preserving exact tile locality (zero cross-tile bleed) while allowing
    64-256 px tiles to represent sub-tile edge locations and compensate FLUX.2 VAE boundary halos.
  - `build_active_rect_canvas`: runs `RectTileDown` + linear projection/lift ONLY on the active
    rectangles present in `shape_map [B, 2, 64, 64]` instead of encoding all 12 dense pyramid levels.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import rect as RL


def basis_to_device(basis, device, dtype=torch.float32):
    """Move 12-shape linear basis dict to `device` and precompute 1D plane bases once."""
    dev = torch.device(device)
    out = {}
    for shape, entry in basis.items():
        sh, sw = int(shape[0]), int(shape[1])
        item = {
            "mode": entry["mode"],
            "U": entry["U"].to(device=dev, dtype=dtype),
        }
        if entry["mode"] != "dc_pca96":
            item["P"] = RL._plane_basis_1d(sh, sw, dev, dtype)
        else:
            item["P_gate"] = RL._plane_basis_1d(sh, sw, dev, torch.float32)
        out[(sh, sw)] = item
    return out


class RectTileDown(nn.Module):
    """Strictly tile-local nonlinear residual encoder over any rectangular shape `(sh, sw)`.

    Adds a zero-initialized 128-d residual delta onto the closed-form orthonormal projection
    `project_rect_tokens`. Uses tile-local CoordConv `(u, v) in [-1, 1]` + `4x4` avg/max pooling
    with a shape-size embedding, keeping weight naming compatible with pilot checkpoints.
    """

    def __init__(self, c=96, hid=256):
        super().__init__()
        self.c = int(c)
        self.hid = int(hid)
        self.conv1 = nn.Conv2d(RL.CH + 2, self.c, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, self.c)
        self.conv2 = nn.Conv2d(self.c, self.c, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, self.c)
        self.shape_mlp = nn.Sequential(
            nn.Linear(2, self.hid),
            nn.SiLU(),
            nn.Linear(self.hid, self.hid),
        )
        self.head = nn.Sequential(
            nn.Linear(self.c * 32, self.hid),
            nn.SiLU(),
            nn.Linear(self.hid, self.hid),
            nn.SiLU(),
        )
        self.out = nn.Linear(self.hid, RL.TOK)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self._uv_cache = {}
        self._svec_cache = {}

    def _uv_and_svec(self, sh, sw, device, dtype):
        key = (int(sh), int(sw), str(device), dtype)
        cached = self._uv_cache.get(key)
        if cached is None:
            ys = torch.linspace(-1.0, 1.0, int(sh), device=device, dtype=dtype)
            xs = torch.linspace(-1.0, 1.0, int(sw), device=device, dtype=dtype)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            uv = torch.stack((yy, xx), dim=0).unsqueeze(0)
            s_vec = torch.tensor([[math.log2(int(sh)) / 6.0, math.log2(int(sw)) / 6.0]], device=device, dtype=dtype)
            self._uv_cache[key] = uv
            self._svec_cache[key] = s_vec
            return uv, s_vec
        return cached, self._svec_cache[key]

    def forward_patches(self, patches, sh, sw):
        """Encode `[M, 32, sh, sw]` isolated rectangular tile patches -> `[M, 128]` residual deltas."""
        m = patches.shape[0]
        if m == 0:
            return patches.new_zeros((0, RL.TOK))
        w_dt = self.conv1.weight.dtype
        x = patches.to(w_dt)
        uv, s_vec = self._uv_and_svec(sh, sw, x.device, w_dt)
        h = F.silu(self.gn1(self.conv1(torch.cat((x, uv.expand(m, -1, -1, -1)), dim=1))))
        h = F.silu(self.gn2(self.conv2(h)))
        pa = F.adaptive_avg_pool2d(h, (4, 4)).flatten(1)
        pm = F.adaptive_max_pool2d(h, (4, 4)).flatten(1)
        feat = self.head(torch.cat((pa, pm), dim=1)) + self.shape_mlp(s_vec)
        return self.out(feat).to(patches.dtype)

    def encode_shape(self, z_norm, shape, basis):
        """Dense encode all non-overlapping `(sh, sw)` tiles in `[B, 32, 64, 64]` -> `[B, gh, gw, 128]`."""
        sh, sw = int(shape[0]), int(shape[1])
        if sh == 2 and sw == 2:
            return RL.project_rect_tokens(z_norm, 2, 2, None)
        b, c, h, w = z_norm.shape
        gh, gw = h // sh, w // sw
        tok_lin = RL.project_rect_tokens(z_norm, sh, sw, basis[(sh, sw)])
        patches = (
            z_norm.unfold(2, sh, sh)
            .unfold(3, sw, sw)
            .permute(0, 2, 3, 1, 4, 5)
            .reshape(b * gh * gw, c, sh, sw)
        )
        delta = self.forward_patches(patches, sh, sw).view(b, gh, gw, RL.TOK)
        return tok_lin + delta


def encode_rect_pyramid_td(z_norm, basis, tiledown=None):
    """Full 13-shape token pyramid `[B, 64//sh, 64//sw, 128]`, optionally with `RectTileDown`."""
    if tiledown is None:
        return RL.encode_rect_pyramid(z_norm, basis)
    pyr = {(2, 2): RL.project_rect_tokens(z_norm, 2, 2, None)}
    for shape in basis:
        sh, sw = int(shape[0]), int(shape[1])
        pyr[(sh, sw)] = tiledown.encode_shape(z_norm, (sh, sw), basis)
    return pyr


def build_active_rect_canvas(z_norm, shape_map, basis, tiledown=None, floor=0.20, tau=0.20, border_open=False):
    """Sparse active-tile canvas builder matching `RL.build_rect_canvas(encode_rect_pyramid_td(...))`.

    Runs linear projection, `RectTileDown`, lifting, and AC gating ONLY on the active rectangles in `shape_map`,
    reducing encoder FLOPs and activation memory by ~10x during training while preserving exact gradients.
    """
    b, _, _, _ = z_norm.shape
    dev, dt = z_norm.device, z_norm.dtype
    sh_m, sw_m = shape_map[:, 0], shape_map[:, 1]
    canvas = z_norm.clone()
    ac_gate = torch.ones((b, 1, RL.SIDE, RL.SIDE), device=dev, dtype=torch.float32)

    for (sh, sw), entry in basis.items():
        sh, sw = int(sh), int(sw)
        act = (sh_m[:, ::sh, ::sw] == sh) & (sw_m[:, ::sh, ::sw] == sw)
        if not bool(act.any()):
            continue
        gh, gw = RL.SIDE // sh, RL.SIDE // sw
        patches_act = (
            z_norm.unfold(2, sh, sh)
            .unfold(3, sw, sw)
            .permute(0, 2, 3, 1, 4, 5)[act]
        )
        m_act = patches_act.shape[0]
        x3 = patches_act.float().reshape(m_act, RL.CH, sh * sw)
        U = entry["U"].to(device=dev, dtype=torch.float32)

        if entry["mode"] == "dc_pca96":
            dc = x3.mean(dim=2)
            ac = (x3 - dc.unsqueeze(2)).reshape(m_act, RL.CH * sh * sw)
            tok_lin = torch.cat((dc, ac @ U), dim=1)
        else:
            P = entry.get("P")
            if P is None or P.device != dev or P.dtype != torch.float32:
                P = RL._plane_basis_1d(sh, sw, dev, torch.float32)
            p_c = x3 @ P.T
            res = (x3 - p_c @ P).reshape(m_act, RL.CH * sh * sw)
            tok_lin = torch.cat((p_c.reshape(m_act, RL.CH * 3), res @ U), dim=1)

        tok = tok_lin + tiledown.forward_patches(patches_act.float(), sh, sw).float() if tiledown is not None else tok_lin

        if entry["mode"] == "dc_pca96":
            dc_t = tok[:, :RL.CH].unsqueeze(2)
            ac_hat = (tok[:, RL.CH:] @ U.T).view(m_act, RL.CH, sh * sw)
            up_3 = ac_hat + dc_t
            P_g = entry.get("P_gate")
            if P_g is None or P_g.device != dev or P_g.dtype != torch.float32:
                P_g = RL._plane_basis_1d(sh, sw, dev, torch.float32)
            non_planar = ac_hat - (ac_hat @ P_g.T) @ P_g
            rms = non_planar.square().mean(dim=(-2, -1)).clamp_min(1e-12).sqrt()
        else:
            p_c_t = tok[:, :RL.CH * 3].view(m_act, RL.CH, 3)
            r_hat = (tok[:, RL.CH * 3:] @ U.T).view(m_act, RL.CH, sh * sw)
            up_3 = p_c_t @ P + r_hat
            res_t = tok[:, RL.CH * 3:]
            rms = res_t.square().mean(dim=-1).clamp_min(1e-12).sqrt()

        up_act = up_3.view(m_act, RL.CH, sh, sw).to(dt)
        g_act = torch.tanh(torch.clamp(rms - float(floor), min=0.0) / float(tau))

        full_up = z_norm.new_zeros((b, gh, gw, RL.CH, sh, sw))
        full_up[act] = up_act
        full_up = full_up.permute(0, 3, 1, 4, 2, 5).reshape(b, RL.CH, RL.SIDE, RL.SIDE)
        m_cell = (sh_m == sh) & (sw_m == sw)
        canvas = torch.where(m_cell[:, None], full_up, canvas)

        full_g = torch.zeros((b, gh, gw), device=dev, dtype=torch.float32)
        full_g[act] = g_act
        g_up = full_g.repeat_interleave(sh, dim=1).repeat_interleave(sw, dim=2).unsqueeze(1)
        ac_gate = torch.where(m_cell[:, None], g_up, ac_gate)

    if border_open:
        _, _, b1 = RL.tile_border_masks(shape_map)
        ac_gate = torch.maximum(ac_gate, b1)

    mask = ((sh_m > 2) | (sw_m > 2))[:, None]

    lsh = torch.log2(sh_m.float() / 2.0) / 4.0
    lsw = torch.log2(sw_m.float() / 2.0) / 4.0
    shape_feat = torch.stack((lsh, lsw, (lsh + lsw) * 0.5, lsh - lsw), dim=1).to(dt)

    grid = torch.arange(RL.SIDE, device=dev, dtype=torch.long)
    yy = grid[:, None].expand(RL.SIDE, RL.SIDE)[None]
    xx = grid[None, :].expand(RL.SIDE, RL.SIDE)[None]
    py = ((yy % sh_m).float() / (sh_m.float() - 1.0).clamp_min(1.0)) * 2.0 - 1.0
    px = ((xx % sw_m).float() / (sw_m.float() - 1.0).clamp_min(1.0)) * 2.0 - 1.0
    pos = torch.stack((py, px), dim=1).to(dt)
    return canvas, mask, shape_feat, pos, ac_gate
