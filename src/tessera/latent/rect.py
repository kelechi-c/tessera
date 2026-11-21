"""2:1 k-d tree rectangular pure-latent layouts: 512 px image as disjoint axis-aligned rectangles (16..256 px).

A layout is a `shape_map` [B, 2, 64, 64] (channel 0 = tile height `sh`, channel 1 = tile width `sw` in 8 px cells),
covering the 64 x 64 FLUX.2 latent grid with disjoint rectangles from `SHAPES` (13 shapes, aspect ratio <= 2:1):
  - Squares:    (2, 2), (4, 4), (8, 8), (16, 16), (32, 32)
  - Horizontal: (2, 4), (4, 8), (8, 16), (16, 32)
  - Vertical:   (4, 2), (8, 4), (16, 8), (32, 16)

Binary k-d tree rules (9 levels, level 0 = 32x32 roots -> level 8 = 2x2 leaves):
  - Square (s, s) with s > 2: action in {0: leaf, 1: split-H -> two (s//2, s), 2: split-V -> two (s, s//2)}
  - Horizontal (s//2, s):     action in {0: leaf, 2: split-V -> two (s//2, s//2)}
  - Vertical   (s, s//2):     action in {0: leaf, 1: split-H -> two (s//2, s//2)}
  - Leaf (2, 2):              always leaf (action = 0; true 16 px FLUX.2 latent)

Every coarse shape (sh, sw) != (2, 2) maps to a 128-float token via an orthonormal DC+AC basis:
  - Small shapes (sh * sw <= 16, i.e. 2x4, 4x2, 4x4): 32 DC + 96 PCA AC
  - Coarse shapes (sh * sw >= 32, i.e. 4x8..32x32):   32 DC + 32 linear-x + 32 linear-y + 32 PCA AC
    (exact on BOTH flat Mondrian boxes AND smooth linear sky gradients!).
"""
import math
import torch

SIDE = 64
CH = 32
TOK = CH * 4                               # 128 floats per token

# 13 allowed shapes ordered from finest (level 8) to coarsest (level 0)
SHAPES = (
    (2, 2),
    (2, 4), (4, 2),
    (4, 4),
    (4, 8), (8, 4),
    (8, 8),
    (8, 16), (16, 8),
    (16, 16),
    (16, 32), (32, 16),
    (32, 32),
)
COARSE_SHAPES = SHAPES[1:]
SHAPE_SET = set(SHAPES)

ACT_LEAF = 0
ACT_SPLIT_H = 1                            # cut horizontally into top + bottom halves (sh // 2, sw)
ACT_SPLIT_V = 2                            # cut vertically into left + right halves (sh, sw // 2)
N_LEVELS = 9                               # level 0 (area 1024 = 32x32) .. level 8 (area 4 = 2x2)


def level_of(sh, sw):
    """Depth level 0..8: level = 10 - log2(sh * sw).  32x32 -> 0, 16x32 -> 1, ..., 2x2 -> 8."""
    return 10 - int(round(math.log2(int(sh) * int(sw))))


def valid_actions(sh, sw):
    """Allowed k-d split actions for a tile of shape (sh, sw) preserving aspect ratio <= 2:1."""
    sh, sw = int(sh), int(sw)
    if sh == 2 and sw == 2:
        return (ACT_LEAF,)
    if sh == sw:
        return (ACT_LEAF, ACT_SPLIT_H, ACT_SPLIT_V)
    if sw > sh:                            # horizontal 2:1 -> only vertical cut restores square
        return (ACT_LEAF, ACT_SPLIT_V)
    return (ACT_LEAF, ACT_SPLIT_H)         # vertical 2:1 -> only horizontal cut restores square


def valid_action_mask(sh_t, sw_t):
    """[...], [...] long tensors -> [..., 3] bool mask of allowed actions."""
    m = torch.zeros((*sh_t.shape, 3), dtype=torch.bool, device=sh_t.device)
    m[..., ACT_LEAF] = True
    can_h = (sh_t > 2) & (sh_t >= sw_t)
    can_v = (sw_t > 2) & (sw_t >= sh_t)
    m[..., ACT_SPLIT_H] = can_h
    m[..., ACT_SPLIT_V] = can_v
    return m


def all_grid_rect(b=1, device=None):
    """[B, 2, 64, 64] all-(2, 2) finest-grid layout (1,024 tokens)."""
    return torch.full((b, 2, SIDE, SIDE), 2, dtype=torch.long, device=device)


def leaves_rect(shape_map_1):
    """Single layout [2, 64, 64] -> list of (r0, c0, sh, sw) in raster order of top-left corners."""
    sm = shape_map_1.detach().cpu()
    if sm.ndim == 4:
        sm = sm[0]
    sh_m = sm[0].numpy()
    sw_m = sm[1].numpy()
    seen = torch.zeros((SIDE, SIDE), dtype=torch.bool).numpy()
    out = []
    for r in range(SIDE):
        for c in range(SIDE):
            if seen[r, c]:
                continue
            sh, sw = int(sh_m[r, c]), int(sw_m[r, c])
            seen[r:r + sh, c:c + sw] = True
            out.append((r, c, sh, sw))
    return out


def tokens_of_rect(shape_map):
    """[B, 2, 64, 64] -> [B] token counts."""
    if shape_map.ndim == 3:
        shape_map = shape_map[None]
    area = (shape_map[:, 0] * shape_map[:, 1]).float()
    return (1.0 / area).flatten(1).sum(1).round().long()


def validate_rect(shape_map):
    """Check that every item in [B, 2, 64, 64] is a valid disjoint tiling from SHAPES."""
    if shape_map.ndim == 3:
        shape_map = shape_map[None]
    for b in range(shape_map.shape[0]):
        sm = shape_map[b].detach().cpu()
        sh_m, sw_m = sm[0], sm[1]
        cov = torch.zeros((SIDE, SIDE), dtype=torch.int32)
        for r, c, sh, sw in leaves_rect(sm):
            if (sh, sw) not in SHAPE_SET:
                return False
            if r + sh > SIDE or c + sw > SIDE:
                return False
            if r % sh != 0 or c % sw != 0:
                return False
            if not bool((sh_m[r:r + sh, c:c + sw] == sh).all() and (sw_m[r:r + sh, c:c + sw] == sw).all()):
                return False
            cov[r:r + sh, c:c + sw] += 1
        if not bool((cov == 1).all()):
            return False
    return True


def kdtree_trace(shape_map_1):
    """Decompose a valid 2:1 k-d tree `shape_map` [2, 64, 64] into breadth-first visited nodes.

    Returns list of dicts per visited node:
      (r0, c0, sh, sw, level, action)
    ordered by level 0..8 (coarsest -> finest), then raster order.  Works for any valid 2:1 k-d tree tiling."""
    sm = shape_map_1.detach().cpu()
    if sm.ndim == 4:
        sm = sm[0]
    sh_m, sw_m = sm[0], sm[1]

    def _action(r0, c0, sh, sw):
        if int(sh_m[r0, c0]) == sh and int(sw_m[r0, c0]) == sw:
            return ACT_LEAF
        # Check if all cells in [r0:r0+sh, c0:c0+sw] stay strictly inside a horizontal cut vs vertical cut
        can_h = (sh > 2) and (sh >= sw)
        can_v = (sw > 2) and (sw >= sh)
        if can_h and not can_v:
            return ACT_SPLIT_H
        if can_v and not can_h:
            return ACT_SPLIT_V
        # Square (s, s): check whether a vertical cut or horizontal cut is respected (no tile crosses the midline)
        mh = r0 + sh // 2
        mv = c0 + sw // 2
        # A tile crosses horizontal midline iff some cell at row mh-1 has tile extending across mh
        # Since all tiles are power-of-2 aligned, a tile in [r0, r0+sh) crosses mh iff its height == sh
        # (already ruled out above if sw == sh; if sw < sh, height == sh means vertical half!).
        h_ok = not bool((sh_m[r0:r0 + sh, c0:c0 + sw] == sh).any())
        v_ok = not bool((sw_m[r0:r0 + sh, c0:c0 + sw] == sw).any())
        if v_ok and not h_ok:
            return ACT_SPLIT_V
        if h_ok and not v_ok:
            return ACT_SPLIT_H
        # Both cuts valid (e.g., 4-way split into squares): canonically split-H first
        return ACT_SPLIT_H

    nodes = []
    frontier = [(0, 0, 32, 32), (0, 32, 32, 32), (32, 0, 32, 32), (32, 32, 32, 32)]
    while frontier:
        nxt = []
        for r0, c0, sh, sw in frontier:
            act = _action(r0, c0, sh, sw)
            nodes.append((r0, c0, sh, sw, level_of(sh, sw), act))
            if act == ACT_SPLIT_H:
                nxt.append((r0, c0, sh // 2, sw))
                nxt.append((r0 + sh // 2, c0, sh // 2, sw))
            elif act == ACT_SPLIT_V:
                nxt.append((r0, c0, sh, sw // 2))
                nxt.append((r0, c0 + sw // 2, sh, sw // 2))
        frontier = nxt
    return nodes


# ---------------------------------------------------------------- orthonormal DC+plane+AC linear basis -------------------
def _extract_rect_patches(z512, sh, sw):
    b, c, h, w = z512.shape
    nh, nw = h // sh, w // sw
    return z512.reshape(b, c, nh, sh, nw, sw).permute(0, 2, 4, 1, 3, 5).reshape(b * nh * nw, c * sh * sw)


def _unpack_rect_patches(x_flat, b, sh, sw):
    nh, nw = SIDE // sh, SIDE // sw
    return x_flat.reshape(b, nh, nw, CH, sh, sw).permute(0, 3, 1, 4, 2, 5).reshape(b, CH, SIDE, SIDE)


def _plane_basis_1d(sh, sw, device, dtype=torch.float64):
    """Orthonormal 3-mode spatial basis [3, sh*sw]: DC (1), linear-y, linear-x."""
    y = torch.linspace(-1.0, 1.0, sh, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, sw, device=device, dtype=dtype)
    b_dc = torch.ones((sh, sw), device=device, dtype=dtype)
    b_dc = b_dc / b_dc.norm()
    b_y = y[:, None].expand(sh, sw).clone()
    b_y = b_y / b_y.norm().clamp_min(1e-12)
    b_x = x[None, :].expand(sh, sw).clone()
    b_x = b_x / b_x.norm().clamp_min(1e-12)
    return torch.stack((b_dc.reshape(-1), b_y.reshape(-1), b_x.reshape(-1)), dim=0)   # [3, sh*sw]


def fit_rect_basis(z512n, shapes=COARSE_SHAPES):
    """Fit orthonormal 128-d linear bases for all 12 coarse rectangular shapes on normalised z512n [N, 32, 64, 64].

    Returns dict {(sh, sw): {'mode': 'dc_pca96'|'plane_pca32', 'U': [32*sh*sw, k]}} where:
      - Small shapes (sh * sw <= 16): 32 exact DC + 96 PCA AC (`U` has 96 orthonormal columns, zero spatial mean)
      - Coarse shapes (sh * sw >= 32): 32 DC + 32 linear-y + 32 linear-x + 32 PCA residual (`U` has 32 orthonormal
        columns orthogonal to DC, y, and x) -> exact on BOTH flat boxes AND smooth sky gradients!
    """
    z = z512n.detach().double().cpu()
    basis = {}
    for sh, sw in shapes:
        sh, sw = int(sh), int(sw)
        X = _extract_rect_patches(z, sh, sw)                                  # [M, 32*sh*sw]
        m = X.shape[0]
        x3 = X.view(m, CH, sh * sw)
        if sh * sw <= 16:
            ac = (x3 - x3.mean(dim=2, keepdim=True)).reshape(m, CH * sh * sw)
            _, _, Vt = torch.linalg.svd(ac, full_matrices=False)
            k = min(96, Vt.shape[0])
            U = torch.zeros(CH * sh * sw, 96, dtype=torch.float64)
            U[:, :k] = Vt[:k].T
            basis[(sh, sw)] = {"mode": "dc_pca96", "U": U.float()}
        else:
            P = _plane_basis_1d(sh, sw, z.device, torch.float64)              # [3, sh*sw]
            proj = (x3 @ P.T) @ P                                             # [M, 32, sh*sw]
            res = (x3 - proj).reshape(m, CH * sh * sw)
            _, _, Vt = torch.linalg.svd(res, full_matrices=False)
            k = min(32, Vt.shape[0])
            U = torch.zeros(CH * sh * sw, 32, dtype=torch.float64)
            U[:, :k] = Vt[:k].T
            basis[(sh, sw)] = {"mode": "plane_pca32", "U": U.float()}
    return basis


def project_rect_tokens(z512n, sh, sw, entry):
    """[B, 32, 64, 64] -> [B, nh, nw, 128] tokens for shape (sh, sw)."""
    b = z512n.shape[0]
    nh, nw = SIDE // sh, SIDE // sw
    if sh == 2 and sw == 2:
        return _extract_rect_patches(z512n.float(), 2, 2).view(b, nh, nw, TOK).to(z512n.dtype)
    X = _extract_rect_patches(z512n.float(), sh, sw)
    m = X.shape[0]
    x3 = X.view(m, CH, sh * sw)
    U = entry["U"].to(device=z512n.device, dtype=torch.float32)
    if entry["mode"] == "dc_pca96":
        dc = x3.mean(dim=2)                                                   # [M, 32]
        ac = (x3 - dc.unsqueeze(2)).reshape(m, CH * sh * sw)
        ac_c = ac @ U                                                         # [M, 96]
        tok = torch.cat((dc, ac_c), dim=1)                                    # [M, 128]
    else:
        P = _plane_basis_1d(sh, sw, z512n.device, torch.float32)              # [3, sh*sw]
        p_c = x3 @ P.T                                                        # [M, 32, 3]
        res = (x3 - p_c @ P).reshape(m, CH * sh * sw)
        r_c = res @ U                                                         # [M, 32]
        tok = torch.cat((p_c.reshape(m, CH * 3), r_c), dim=1)                 # [M, 128]
    return tok.view(b, nh, nw, TOK).to(z512n.dtype)


def lift_rect_tokens(tok_grid, sh, sw, entry):
    """[B, nh, nw, 128] -> [B, 32, 64, 64] lifted latent canvas."""
    b, nh, nw, _ = tok_grid.shape
    if sh == 2 and sw == 2:
        return _unpack_rect_patches(tok_grid.reshape(b * nh * nw, TOK).float(), b, 2, 2).to(tok_grid.dtype)
    tok = tok_grid.reshape(b * nh * nw, TOK).float()
    m = tok.shape[0]
    U = entry["U"].to(device=tok_grid.device, dtype=torch.float32)
    if entry["mode"] == "dc_pca96":
        dc = tok[:, :CH].unsqueeze(2)                                         # [M, 32, 1]
        ac_hat = (tok[:, CH:] @ U.T).view(m, CH, sh * sw)
        x_hat = (ac_hat + dc).reshape(m, CH * sh * sw)
    else:
        P = _plane_basis_1d(sh, sw, tok_grid.device, torch.float32)           # [3, sh*sw]
        p_c = tok[:, :CH * 3].view(m, CH, 3)
        r_hat = (tok[:, CH * 3:] @ U.T).view(m, CH, sh * sw)
        x_hat = (p_c @ P + r_hat).reshape(m, CH * sh * sw)
    return _unpack_rect_patches(x_hat, b, sh, sw).to(tok_grid.dtype)


def encode_rect_pyramid(z512n, basis):
    """[B, 32, 64, 64] -> dict {(sh, sw): [B, 64//sh, 64//sw, 128]} for all 13 shapes."""
    pyr = {(2, 2): project_rect_tokens(z512n, 2, 2, None)}
    for (sh, sw), entry in basis.items():
        pyr[(sh, sw)] = project_rect_tokens(z512n, sh, sw, entry)
    return pyr


def tile_ac_gate_rect(tok_grid, sh, sw, entry, floor=0.20, tau=0.20):
    """Smooth dead-zone non-planar energy gate in [0, 1] per tile -> [B, 1, nh, nw].

    Subtracts the 3-mode linear plane (DC + y + x) on ALL 12 coarse shapes before measuring residual RMS, so both
    flat Mondrian boxes AND smooth linear sky gradients stay exact 0 while edges/textures open smoothly to 1."""
    b, nh, nw, _ = tok_grid.shape
    tok = tok_grid.float()
    if entry["mode"] == "plane_pca32":
        res = tok[..., CH * 3:]                                               # 32 non-planar residual coeffs
        rms = res.square().mean(dim=-1).clamp_min(1e-12).sqrt()               # [B, nh, nw]
    else:
        # Lift dc_pca96 token to [M, 32, sh*sw] and project out linear plane (DC, y, x)
        m = b * nh * nw
        U = entry["U"].to(device=tok.device, dtype=torch.float32)
        ac_hat = (tok.reshape(m, TOK)[:, CH:] @ U.T).view(m, CH, sh * sw)
        P = _plane_basis_1d(sh, sw, tok.device, torch.float32)                # [3, sh*sw]
        non_planar = ac_hat - (ac_hat @ P.T) @ P
        rms = non_planar.square().mean(dim=(-2, -1)).clamp_min(1e-12).sqrt().view(b, nh, nw)
    return torch.tanh(torch.clamp(rms - float(floor), min=0.0) / float(tau)).unsqueeze(1)


def tile_border_masks(shape_map):
    """Compute exact inter-tile boundary masks on the 64 x 64 latent grid from `shape_map` [B, 2, 64, 64].

    Returns:
      border_x [B, 1, 64, 63] bool: vertical boundary between col c and c+1 (belongs to different tiles)
      border_y [B, 1, 63, 64] bool: horizontal boundary between row r and r+1 (belongs to different tiles)
      border_1cell [B, 1, 64, 64] float32: 1.0 on cells touching any coarse-tile boundary, 0.0 on tile interiors
    """
    b = shape_map.shape[0]
    dev = shape_map.device
    sh_m, sw_m = shape_map[:, 0], shape_map[:, 1]
    grid = torch.arange(SIDE, device=dev, dtype=torch.long)
    yy = grid[:, None].expand(SIDE, SIDE)[None]
    xx = grid[None, :].expand(SIDE, SIDE)[None]
    # Unique tile ID per cell = (r0 * 64 + c0)
    r0 = (yy // sh_m) * sh_m
    c0 = (xx // sw_m) * sw_m
    tid = r0 * SIDE + c0                                                      # [B, 64, 64]
    bx = (tid[:, :, 1:] != tid[:, :, :-1]).unsqueeze(1)                       # [B, 1, 64, 63]
    by = (tid[:, 1:, :] != tid[:, :-1, :]).unsqueeze(1)                       # [B, 1, 63, 64]
    # Cells adjacent to a boundary where at least one side is a coarse tile (> 2x2)
    coarse = ((sh_m > 2) | (sw_m > 2)).unsqueeze(1)
    bx_c = bx & (coarse[:, :, :, 1:] | coarse[:, :, :, :-1])
    by_c = by & (coarse[:, :, 1:, :] | coarse[:, :, :-1, :])
    b1 = torch.zeros((b, 1, SIDE, SIDE), device=dev, dtype=torch.bool)
    b1[:, :, :, 1:] |= bx_c
    b1[:, :, :, :-1] |= bx_c
    b1[:, :, 1:, :] |= by_c
    b1[:, :, :-1, :] |= by_c
    return bx_c, by_c, b1.float()


def build_rect_canvas(rect_pyr, shape_map, basis, floor=0.20, tau=0.20, border_open=False):
    """Build (canvas [B,32,64,64], mask [B,1,64,64], shape_feat [B,4,64,64], pos [B,2,64,64], ac_gate [B,1,64,64])."""
    z2 = lift_rect_tokens(rect_pyr[(2, 2)], 2, 2, None)
    b, _, _, _ = z2.shape
    dev, dt = z2.device, z2.dtype
    sh_m, sw_m = shape_map[:, 0], shape_map[:, 1]
    canvas = z2.clone()
    ac_gate = torch.ones((b, 1, SIDE, SIDE), device=dev, dtype=torch.float32)

    for (sh, sw), entry in basis.items():
        m = (sh_m == sh) & (sw_m == sw)
        if bool(m.any()):
            up = lift_rect_tokens(rect_pyr[(sh, sw)], sh, sw, entry)
            canvas = torch.where(m[:, None], up, canvas)
            g = tile_ac_gate_rect(rect_pyr[(sh, sw)], sh, sw, entry, floor=floor, tau=tau)
            g_up = g.repeat_interleave(sh, dim=-2).repeat_interleave(sw, dim=-1)
            ac_gate = torch.where(m[:, None], g_up, ac_gate)

    if border_open:
        _, _, b1 = tile_border_masks(shape_map)
        ac_gate = torch.maximum(ac_gate, b1)

    mask = ((sh_m > 2) | (sw_m > 2))[:, None]

    # 4D log-shape features per cell: (log2(sh/2), log2(sw/2), log2(sh*sw/4), log2(sh/sw))
    lsh = torch.log2(sh_m.float() / 2.0) / 4.0
    lsw = torch.log2(sw_m.float() / 2.0) / 4.0
    larea = (lsh + lsw) * 0.5
    lasp = lsh - lsw
    shape_feat = torch.stack((lsh, lsw, larea, lasp), dim=1).to(dt)           # [B, 4, 64, 64]

    # Tile-local normalised coordinates in [-1, 1]
    grid = torch.arange(SIDE, device=dev, dtype=torch.long)
    yy = grid[:, None].expand(SIDE, SIDE)[None]                               # [1, 64, 64]
    xx = grid[None, :].expand(SIDE, SIDE)[None]
    py = ((yy % sh_m).float() / (sh_m.float() - 1.0).clamp_min(1.0)) * 2.0 - 1.0
    px = ((xx % sw_m).float() / (sw_m.float() - 1.0).clamp_min(1.0)) * 2.0 - 1.0
    pos = torch.stack((py, px), dim=1).to(dt)                                 # [B, 2, 64, 64]
    return canvas, mask, shape_feat, pos, ac_gate
