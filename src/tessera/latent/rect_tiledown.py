"""Strictly tile-local nonlinear residual encoder (`RectTileDown`) and active-tile canvas builder.

Pairs with `tessera.latent.rect` (`2:1` k-d tree rectangular layouts, `SHAPES`, `COARSE_SHAPES`):
  - `RectTileDown`: zero-initialized CoordConv residual on top of the orthonormal linear projection
    (`project_rect_tokens`), preserving exact tile locality (zero cross-tile bleed) while allowing
    64-256 px tiles to represent sub-tile edge locations and compensate FLUX.2 VAE boundary halos.
  - `RectTileLift`: zero-initialized, wire-valid (token-only) per-tile nonlinear residual decoder on top of the
    linear lift, gated by the wire `ac_gate`; output is per-tile zero-mean so the token DC stays exact.
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


def _lock_dc_delta(delta, mode):
    """Zero the DC coefficients of a `[..., 128]` residual delta out-of-place (autograd + `torch.compile` safe).

    `dc_pca96`: token `[:32]` is the per-channel spatial mean.  `plane_pca32`: token `[:96]` is `[32, 3]`
    (DC, linear-y, linear-x) per channel, so the DC slot is every third entry of the first 96."""
    idx = torch.arange(RL.TOK, device=delta.device)
    if mode == "dc_pca96":
        keep = idx >= RL.CH
    else:
        keep = (idx >= RL.CH * 3) | (idx % 3 != 0)                           # [32, 3] slot 0 is DC
    return delta * keep.to(delta.dtype)


def _plane_gate_basis(entry, sh, sw, device):
    """3-mode `[3, sh*sw]` plane basis in float32 (cached `P` / `P_gate` when present, else rebuilt)."""
    P = entry.get("P") if entry["mode"] != "dc_pca96" else entry.get("P_gate")
    if P is None or P.device != device or P.dtype != torch.float32:
        P = RL._plane_basis_1d(sh, sw, device, torch.float32)
    return P


def _input_nonplanar_gate(x3, P, floor, tau):
    """Smooth gate `[M, 1]` from the input patch's true non-planar residual RMS (exact 0 when `rms <= floor`)."""
    res_np = x3 - (x3 @ P.T) @ P
    rms_in = res_np.square().mean(dim=(-2, -1)).clamp_min(1e-12).sqrt()
    return torch.tanh(torch.clamp(rms_in - float(floor), min=0.0) / float(tau)).unsqueeze(-1)


def project_coarse_tile_zero_mean(resid, shape_map):
    """Subtract each coarse tile's per-channel spatial mean from `resid [B, C, 64, 64]` (out-of-place).

    Cells covered by `(2, 2)` leaves are left untouched.  Every coarse shape is processed unconditionally (no
    host sync on `m_cell.any()`), which gives the identical result and stays `torch.compile` friendly."""
    b, c, h, w = resid.shape
    sh_m, sw_m = shape_map[:, 0], shape_map[:, 1]
    out = resid
    for sh, sw in RL.COARSE_SHAPES:
        sh, sw = int(sh), int(sw)
        m_cell = (sh_m == sh) & (sw_m == sw)
        r6 = resid.reshape(b, c, h // sh, sh, w // sw, sw)
        zm = (r6 - r6.mean(dim=(3, 5), keepdim=True)).reshape(b, c, h, w)
        out = torch.where(m_cell[:, None], zm, out)
    return out


class RectTileDown(nn.Module):
    """Strictly tile-local nonlinear residual encoder over any rectangular shape `(sh, sw)`.

    Adds a zero-initialized 128-d residual delta onto the closed-form orthonormal projection
    `project_rect_tokens`. Uses tile-local CoordConv `(u, v) in [-1, 1]` + `4x4` avg/max pooling
    with a shape-size embedding, keeping weight naming compatible with pilot checkpoints.
    """

    def __init__(self, c=96, hid=256, use_1d_pool=False, prof_c=16, prof_len=32, dc_lock=False):
        super().__init__()
        self.c = int(c)
        self.hid = int(hid)
        self.use_1d_pool = bool(use_1d_pool)
        self.prof_c = int(prof_c)
        self.prof_len = int(prof_len)
        self.dc_lock = bool(dc_lock)
        # Plain (non-state-dict) gate settings shared with `encode_shape` / `encode_rect_pyramid_td`
        self.ac_floor = 0.20
        self.ac_tau = 0.20
        self.conv1 = nn.Conv2d(RL.CH + 2, self.c, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, self.c)
        self.conv2 = nn.Conv2d(self.c, self.c, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, self.c)
        self.shape_mlp = nn.Sequential(
            nn.Linear(2, self.hid),
            nn.SiLU(),
            nn.Linear(self.hid, self.hid),
        )
        head_in = self.c * 32
        if self.use_1d_pool:
            # Tile-local row/column profiles (avg + max over each axis) concatenated AFTER the 4x4 pooled features
            self.prof_proj = nn.Conv2d(self.c, self.prof_c, 1)
            head_in += self.prof_c * self.prof_len * 4
        self.head = nn.Sequential(
            nn.Linear(head_in, self.hid),
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
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            h = F.silu(self.gn1(self.conv1(torch.cat((x, uv.expand(m, -1, -1, -1)), dim=1))))
            h = F.silu(self.gn2(self.conv2(h)))
            pa = F.adaptive_avg_pool2d(h, (4, 4)).flatten(1)
            pm = F.adaptive_max_pool2d(h, (4, 4)).flatten(1)
            parts = [pa, pm]
            if self.use_1d_pool:
                hp = F.silu(self.prof_proj(h))
                L = self.prof_len
                parts += [
                    F.adaptive_avg_pool2d(hp, (L, 1)).flatten(1),
                    F.adaptive_max_pool2d(hp, (L, 1)).flatten(1),
                    F.adaptive_avg_pool2d(hp, (1, L)).flatten(1),
                    F.adaptive_max_pool2d(hp, (1, L)).flatten(1),
                ]
            feat = self.head(torch.cat(parts, dim=1)) + self.shape_mlp(s_vec)
            out = self.out(feat)
        return out.to(patches.dtype)

    def gated_delta(self, patches, sh, sw, entry, floor=None, tau=None):
        """`forward_patches` + optional DC lock and input non-planar gate -> `[M, 128]` float32 delta."""
        delta = self.forward_patches(patches, sh, sw).float()
        if not self.dc_lock:
            return delta
        floor = getattr(self, "ac_floor", 0.20) if floor is None else floor
        tau = getattr(self, "ac_tau", 0.20) if tau is None else tau
        delta = _lock_dc_delta(delta, entry["mode"])
        m = patches.shape[0]
        x3 = patches.float().reshape(m, RL.CH, sh * sw)
        P = _plane_gate_basis(entry, sh, sw, x3.device)
        return delta * _input_nonplanar_gate(x3, P, floor, tau)

    def encode_shape(self, z_norm, shape, basis, ac_floor=None, ac_tau=None):
        """Dense encode all non-overlapping `(sh, sw)` tiles in `[B, 32, 64, 64]` -> `[B, gh, gw, 128]`.

        With `dc_lock`, `ac_floor` / `ac_tau` (default: `self.ac_floor` / `self.ac_tau`) set the input gate."""
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
        if self.dc_lock:
            delta = self.gated_delta(patches, sh, sw, basis[(sh, sw)], floor=ac_floor, tau=ac_tau)
            return (tok_lin.float() + delta.view(b, gh, gw, RL.TOK)).to(tok_lin.dtype)
        delta = self.forward_patches(patches, sh, sw).view(b, gh, gw, RL.TOK)
        return tok_lin + delta


class RectTileLift(nn.Module):
    """Strictly tile-local nonlinear token -> latent residual decoder (wire-valid: sees ONLY the 128-float token).

    Maps `tok [M, 128]` + tile shape `(sh, sw)` to a per-tile zero-mean residual `[M, 32, sh, sw]` that is added
    onto `RL.lift_rect_tokens(tok)` (gated by the wire `ac_gate(tok)`).  Two branches feed a tile-local CoordConv
    refiner: a separable rank-`rank` row/column profile branch (sharp axis-aligned edges at any position) and a
    coarse `grid_s x grid_s` 2D branch (bilinearly resized to the tile).  The last conv is zero-initialized, so
    the step-0 output is exactly 0 and the canvas equals the linear lift.
    """

    def __init__(self, hid=256, rank=8, prof_len=32, grid_s=8):
        super().__init__()
        self.hid = int(hid)
        self.rank = int(rank)
        self.prof_len = int(prof_len)
        self.grid_s = int(grid_s)
        self.inp = nn.Sequential(
            nn.Linear(RL.TOK + 2, self.hid),
            nn.SiLU(),
            nn.Linear(self.hid, self.hid),
            nn.SiLU(),
        )
        self.prof = nn.Linear(self.hid, 2 * self.rank * self.prof_len)
        self.prof_mix = nn.Conv2d(2 * self.rank, RL.CH, 1)
        self.grid = nn.Linear(self.hid, RL.CH * self.grid_s * self.grid_s)
        self.ref1 = nn.Conv2d(RL.CH + 2, RL.CH, 3, padding=1)
        self.ref2 = nn.Conv2d(RL.CH, RL.CH, 3, padding=1)
        nn.init.zeros_(self.ref2.weight)
        nn.init.zeros_(self.ref2.bias)
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

    def forward_tokens(self, tok, sh, sw):
        """`[M, 128]` wire tokens of `(sh, sw)` tiles -> `[M, 32, sh, sw]` per-tile zero-mean residual."""
        sh, sw = int(sh), int(sw)
        m = tok.shape[0]
        if m == 0:
            return tok.new_zeros((0, RL.CH, sh, sw))
        w_dt = self.prof.weight.dtype
        x = tok.to(w_dt)
        uv, s_vec = self._uv_and_svec(sh, sw, x.device, w_dt)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            h = self.inp(torch.cat((x, s_vec.expand(m, -1)), dim=1))
            rc = self.prof(h).view(m, 2 * self.rank, self.prof_len)
            row = F.interpolate(rc[:, :self.rank], size=sh, mode="linear", align_corners=False)      # [M, r, sh]
            col = F.interpolate(rc[:, self.rank:], size=sw, mode="linear", align_corners=False)      # [M, r, sw]
            row = row.unsqueeze(-1).expand(m, self.rank, sh, sw)
            col = col.unsqueeze(-2).expand(m, self.rank, sh, sw)
            b1 = self.prof_mix(torch.cat((row * col, row + col), dim=1))                            # [M, 32, sh, sw]
            g2 = self.grid(h).view(m, RL.CH, self.grid_s, self.grid_s)
            b2 = F.interpolate(g2, size=(sh, sw), mode="bilinear", align_corners=False)
            f = torch.cat((b1 + b2, uv.expand(m, -1, -1, -1)), dim=1)
            out = self.ref2(F.silu(self.ref1(f))).float()
        out = out - out.mean(dim=(-2, -1), keepdim=True)
        return out.to(tok.dtype)


def lift_rect_tokens_tl(tok_grid, sh, sw, entry, tilelift=None, floor=0.20, tau=0.20, gate_scale="token"):
    """`RL.lift_rect_tokens` + optional gated `RectTileLift` residual: `[B, nh, nw, 128]` -> `[B, 32, 64, 64]`.

    `gate_scale`: see `RL.tile_ac_gate_rect` (default `"token"`)."""
    sh, sw = int(sh), int(sw)
    up = RL.lift_rect_tokens(tok_grid, sh, sw, entry)
    if tilelift is None or (sh, sw) == (2, 2):
        return up
    b, nh, nw, _ = tok_grid.shape
    g = RL.tile_ac_gate_rect(tok_grid, sh, sw, entry, floor=floor, tau=tau, gate_scale=gate_scale)  # [B, 1, nh, nw]
    tok_flat = tok_grid.reshape(b * nh * nw, RL.TOK).float()
    r = tilelift.forward_tokens(tok_flat, sh, sw).float() * g[:, 0].reshape(-1, 1, 1, 1)    # [M, 32, sh, sw]
    r_up = RL._unpack_rect_patches(r.reshape(b * nh * nw, RL.CH * sh * sw), b, sh, sw)
    return (up.float() + r_up).to(up.dtype)


def build_rect_canvas_tl(rect_pyr, shape_map, basis, tilelift=None, floor=0.20, tau=0.20, border_open=False,
                         gate_scale="token"):
    """Dense counterpart of `build_active_rect_canvas(..., tilelift=tilelift)` over a precomputed token pyramid.

    Identical to `RL.build_rect_canvas` when `tilelift is None`.  `gate_scale`: see `RL.tile_ac_gate_rect` (default
    `"token"`; `"pixel"` is an eval-time experiment, `build_active_rect_canvas` has no counterpart)."""
    if tilelift is None:
        return RL.build_rect_canvas(rect_pyr, shape_map, basis, floor=floor, tau=tau, border_open=border_open,
                                    gate_scale=gate_scale)
    canvas, mask, shape_feat, pos, ac_gate = RL.build_rect_canvas(
        rect_pyr, shape_map, basis, floor=floor, tau=tau, border_open=border_open, gate_scale=gate_scale)
    sh_m, sw_m = shape_map[:, 0], shape_map[:, 1]
    for (sh, sw), entry in basis.items():
        sh, sw = int(sh), int(sw)
        m = (sh_m == sh) & (sw_m == sw)
        if bool(m.any()):
            up = lift_rect_tokens_tl(rect_pyr[(sh, sw)], sh, sw, entry, tilelift=tilelift, floor=floor, tau=tau,
                                     gate_scale=gate_scale)
            canvas = torch.where(m[:, None], up, canvas)
    return canvas, mask, shape_feat, pos, ac_gate


def encode_rect_pyramid_td(z_norm, basis, tiledown=None, ac_floor=None, ac_tau=None):
    """Full 13-shape token pyramid `[B, 64//sh, 64//sw, 128]`, optionally with `RectTileDown`.

    `ac_floor` / `ac_tau` default to `tiledown.ac_floor` / `tiledown.ac_tau` (0.20 when absent)."""
    if tiledown is None:
        return RL.encode_rect_pyramid(z_norm, basis)
    floor = getattr(tiledown, "ac_floor", 0.20) if ac_floor is None else ac_floor
    tau = getattr(tiledown, "ac_tau", 0.20) if ac_tau is None else ac_tau
    pyr = {(2, 2): RL.project_rect_tokens(z_norm, 2, 2, None)}
    for shape in basis:
        sh, sw = int(shape[0]), int(shape[1])
        pyr[(sh, sw)] = tiledown.encode_shape(z_norm, (sh, sw), basis, ac_floor=floor, ac_tau=tau)
    return pyr


def build_active_rect_canvas(z_norm, shape_map, basis, tiledown=None, tilelift=None, floor=0.20, tau=0.20,
                             border_open=False, return_td_delta=False):
    """Sparse active-tile canvas builder matching `RL.build_rect_canvas(encode_rect_pyramid_td(...))`.

    Runs linear projection, `RectTileDown`, lifting, and AC gating ONLY on the active rectangles in `shape_map`,
    reducing encoder FLOPs and activation memory by ~10x during training while preserving exact gradients.
    `floor` / `tau` set both the wire `ac_gate` and (with `tiledown.dc_lock`) the input-RMS delta gate, so callers
    must keep `tiledown.ac_floor` / `ac_tau` equal to them for dense/sparse parity.

    `tilelift` (`RectTileLift`) adds its wire-gated per-tile residual onto the lifted active tiles, matching
    `build_rect_canvas_tl(encode_rect_pyramid_td(...), tilelift=tilelift)`.  With `return_td_delta=True` a sixth
    output `td_delta_l2` (scalar mean squared `RectTileDown` token delta over active coarse tiles) is returned.
    """
    b, _, _, _ = z_norm.shape
    dev, dt = z_norm.device, z_norm.dtype
    sh_m, sw_m = shape_map[:, 0], shape_map[:, 1]
    canvas = z_norm.clone()
    ac_gate = torch.ones((b, 1, RL.SIDE, RL.SIDE), device=dev, dtype=torch.float32)
    d_sq = z_norm.new_zeros((), dtype=torch.float32)
    d_cnt = 0

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

        if tiledown is None:
            tok = tok_lin
        else:
            if getattr(tiledown, "dc_lock", False):
                # DC-locked delta, gated by the INPUT patch's non-planar RMS: flat / pure-plane tiles get exact tok_lin
                delta = tiledown.gated_delta(patches_act.float(), sh, sw, entry, floor=floor, tau=tau)
            else:
                delta = tiledown.forward_patches(patches_act.float(), sh, sw).float()
            tok = tok_lin + delta
            if return_td_delta:
                d_sq = d_sq + delta.square().sum()
                d_cnt += m_act * RL.TOK

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

        g_act = torch.tanh(torch.clamp(rms - float(floor), min=0.0) / float(tau))
        up_4 = up_3.view(m_act, RL.CH, sh, sw)
        if tilelift is not None:
            up_4 = up_4 + tilelift.forward_tokens(tok, sh, sw).float() * g_act[:, None, None, None]
        up_act = up_4.to(dt)

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
    if return_td_delta:
        td_delta_l2 = d_sq / float(d_cnt) if d_cnt > 0 else d_sq
        return canvas, mask, shape_feat, pos, ac_gate, td_delta_l2
    return canvas, mask, shape_feat, pos, ac_gate
