"""Layout search for the 2:1 k-d tree tokenizer: top-down splitting and bottom-up sibling pruning, both verified by the
Judge, plus the one-shot rate-distortion tree (`rd_layout`).  A layout is a `shape_map` [B, 2, 64, 64] holding the tile
height and width (in 8 px latent cells) of the leaf covering each cell."""
import torch
import torch.nn.functional as F

from tessera.latent import rect as RL
from tessera.latent import rect_tiledown as RTD
from tessera.rect.model import CH, TileUp

IN_CH_RECT = CH + 4 + 2 + 1      # TileUp input channels of the rectangular tokenizer: canvas, shape (4), position, noise


def apply_sibling_merges(shape_map, parents):
    """Return a copy of `shape_map` [1, 2, 64, 64] with each parent region painted as one `(sh, sw)` leaf."""
    sm = shape_map.clone()
    for r0, c0, sh, sw in parents:
        sm[:, 0, r0:r0 + sh, c0:c0 + sw] = sh
        sm[:, 1, r0:r0 + sh, c0:c0 + sw] = sw
    return sm


def best_hv_split_fn(err_maps):
    """Callable `(r0, c0, sh, sw) -> ACT_SPLIT_H | ACT_SPLIT_V` selecting lower lift-error child shape."""
    def _choose(r0, c0, sh, sw):
        eh = float(err_maps[(sh // 2, sw)][r0:r0 + sh, c0:c0 + sw].mean())
        ev = float(err_maps[(sh, sw // 2)][r0:r0 + sh, c0:c0 + sw].mean())
        return RL.ACT_SPLIT_H if eh <= ev else RL.ACT_SPLIT_V
    return _choose


@torch.no_grad()
def compute_lift_err_maps(z512_1, rect_pyr_n, basis_dev, mv, sv, tilelift=None, ac_floor=0.20, ac_tau=0.20):
    """Per-shape channel-normalized latent lift MAE map `[64, 64]` for fast H vs V split selection.

    With `tilelift`, each shape is lifted through `RectTileLift` (gated like the canvas) before
    measuring error."""
    z512_f = z512_1.float()
    sd_ch = sv.float().clamp_min(1e-3)
    err_maps = {(2, 2): torch.zeros((RL.SIDE, RL.SIDE), device=z512_f.device)}
    for (sh, sw), entry in basis_dev.items():
        zhat = RTD.lift_rect_tokens_tl(rect_pyr_n[(sh, sw)], sh, sw, entry, tilelift=tilelift,
                                       floor=ac_floor, tau=ac_tau) * sv + mv
        err_maps[(sh, sw)] = ((zhat - z512_f).abs() / sd_ch).mean(1)[0]
    return err_maps


def load_tokenizer_ckpt(ckpt_path, dev, load_tiledown=True):
    """Load the EMA `(TileUp, RectTileDown, basis_dev, mean, std, basis_cpu)` from a `tileup_rect.pt` checkpoint.

    Stage 1b metadata (`td_1d_pool`, `td_prof_c`, `td_prof_len`, `td_dc_lock`, `zero_mean_resid`, `ac_floor`,
    `ac_tau`; legacy defaults `False, 16, 32, False, False, 0.20, 0.20`) is attached as plain attributes:
    `tu._zero_mean_resid`, `tu._ac_floor`, `tu._ac_tau` and `td.ac_floor`, `td.ac_tau`.  A `RectTileLift`
    (`"tilelift"` + `tl_hid`, `tl_rank`, `tl_prof_len`) is attached as `tu._tilelift` (`None` when absent).
    Pass `load_tiledown=False` on decode-only paths to skip instantiating `RectTileDown`."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    widths = tuple(int(x) for x in ck.get("widths", (128, 192, 256)))
    attn_blocks = int(ck.get("attn_blocks", 2))
    attn_32 = bool(ck.get("attn_32", False))
    tu = TileUp(c=widths, in_ch=IN_CH_RECT, attn_blocks=attn_blocks, attn_32=attn_32).to(dev)
    tu.load_state_dict(ck["model"])
    tu.eval().requires_grad_(False)
    ac_floor = float(ck.get("ac_floor", 0.20))
    ac_tau = float(ck.get("ac_tau", 0.20))
    tu._zero_mean_resid = bool(ck.get("zero_mean_resid", False))
    tu._ac_floor, tu._ac_tau = ac_floor, ac_tau

    td = None
    if load_tiledown and "tiledown" in ck:
        td = RTD.RectTileDown(
            c=int(ck.get("td_c", 96)), hid=int(ck.get("td_hid", 256)),
            use_1d_pool=bool(ck.get("td_1d_pool", False)), prof_c=int(ck.get("td_prof_c", 16)),
            prof_len=int(ck.get("td_prof_len", 32)), dc_lock=bool(ck.get("td_dc_lock", False)),
        ).to(dev)
        td.load_state_dict(ck["tiledown"])
        td.eval().requires_grad_(False)
        td.ac_floor, td.ac_tau = ac_floor, ac_tau

    tl = None
    if "tilelift" in ck:
        tl = RTD.RectTileLift(hid=int(ck.get("tl_hid", 256)), rank=int(ck.get("tl_rank", 8)),
                              prof_len=int(ck.get("tl_prof_len", 32))).to(dev)
        tl.load_state_dict(ck["tilelift"])
        tl.eval().requires_grad_(False)
    # Plain attribute (not a registered submodule): keeps `tu.state_dict()` / param counts TileUp-only
    object.__setattr__(tu, "_tilelift", tl)

    basis_cpu = ck["basis"]
    mean = ck["mean"].float()
    std = ck["std"].float()

    basis_dev = RTD.basis_to_device(basis_cpu, dev, dtype=torch.float32)
    return tu, td, basis_dev, mean, std, basis_cpu


@torch.no_grad()
def predict_rect_canvas_td(model, z512, rect_pyr_n, shape_map, basis_dev, mean, std, seed=0):
    """Reconstruct raw `[B, 32, 64, 64]` latent canvas from precomputed `rect_pyr_n` + `TileUp`.

    `model._spike_k = k` (absent or 0 = off) soft-clips isolated spikes of the residual on leaves with
    `sh * sw >= model._spike_min_area` (default 64), see `spike_clamp_resid`."""
    dev = z512.device
    use_cuda = dev.type == "cuda"
    mv, sv = mean.view(1, RL.CH, 1, 1).to(dev), std.view(1, RL.CH, 1, 1).to(dev)
    sm_dev = shape_map.to(dev)
    canvas, mask, sf, pos, ac_gate = RTD.build_rect_canvas_tl(
        rect_pyr_n, sm_dev, basis_dev, tilelift=getattr(model, "_tilelift", None),
        floor=getattr(model, "_ac_floor", 0.20), tau=getattr(model, "_ac_tau", 0.20),
    )
    gen = torch.Generator(device=dev).manual_seed(seed)
    nz = torch.randn((len(z512), 1, RL.SIDE, RL.SIDE), device=dev, generator=gen, dtype=canvas.dtype)
    if model is not None:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
            resid = model.residual(canvas, sf, pos, nz)
        resid = resid.float()
        if getattr(model, "_zero_mean_resid", False):
            resid = RTD.project_coarse_tile_zero_mean(resid, sm_dev)
        spike_k = float(getattr(model, "_spike_k", 0.0) or 0.0)
        if spike_k > 0:
            resid = spike_clamp_resid(resid, sm_dev, spike_k, int(getattr(model, "_spike_min_area", 64)))
            if getattr(model, "_zero_mean_resid", False):
                resid = RTD.project_coarse_tile_zero_mean(resid, sm_dev)
        z_n = canvas + ac_gate * resid
    else:
        z_n = canvas
    return torch.where(mask, z_n * sv + mv, z512.float())


@torch.no_grad()
def prune_kdtree_siblings_td(model, tiledown, z512_1, basis_dev, mean, std, dec, judge, shape_map, max_rounds=8,
                             max_area=1024, ev_init=None, y_init=None, bisect_dirty=True, max_calls=0, *, seed=0):
    """Judge-verified bottom-up pruning of a passing top-down layout (e.g. from `split_kdtree_topdown_td`).

    Each round collects sibling leaf pairs sharing an immediate k-d parent (`sibling_merge_candidates`), orders them by
    the parent shape's latent lift error (lowest first) and tries them in ordered chunks: a chunk is accepted only if
    the full `TileUp` forward + decode of the merged layout passes `judge(...)["passed"]`.  A failing chunk is split:
    with a Judge `bad8` map, candidates whose merged region contains bad cells ("dirty") are separated from the clean
    rest, which is retried first as one chunk; dirty ones (and chunks without `bad8`) are halved down to singletons.
    A failing singleton is rejected for good (never retried in later rounds), so the total cost is O(n) Judge calls
    worst case and O(log n) per round when most merges are safe.  `bisect_dirty=False` is a faster, lossier mode:
    dirty candidates are deferred to the next round instead of bisected.  Acceptance always needs a Judge pass.
    Rounds repeat while merges are accepted (merged parents can form new sibling pairs).  Every merge keeps a valid
    non-overlapping dyadic 2:1 k-d tiling; the four 32x32 roots are never merged.  `max_calls > 0` caps total Judge
    calls per image and stops cleanly with the last verified passing layout.  A non-passing input is returned
    unchanged.  Returns the same keys as `split_kdtree_topdown_td` plus `merges` and `prune_calls`."""
    dev = z512_1.device
    use_cuda = dev.type == "cuda"
    z512_f = z512_1.float()
    mv, sv = mean.view(1, RL.CH, 1, 1).to(dev), std.view(1, RL.CH, 1, 1).to(dev)
    zn = (z512_f - mv) / sv
    rect_pyr_n = RTD.encode_rect_pyramid_td(zn, basis_dev, tiledown=tiledown)
    err_maps = compute_lift_err_maps(z512_f, rect_pyr_n, basis_dev, mv, sv, tilelift=getattr(model, "_tilelift", None),
                                     ac_floor=getattr(model, "_ac_floor", 0.20), ac_tau=getattr(model, "_ac_tau", 0.20))
    calls = 0

    def _eval(sm_c):
        nonlocal calls
        z = predict_rect_canvas_td(model, z512_f, rect_pyr_n, sm_c, basis_dev, mean, std, seed=seed)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
            y = dec(z).float().clamp(-1, 1)
        calls += 1
        return y, judge(y)

    sm = shape_map.to(dev).clone()
    if ev_init is None or y_init is None:
        y_cur, ev_cur = _eval(sm)
    else:
        y_cur, ev_cur = y_init, ev_init

    def _result(merges):
        return dict(shape_map=sm, tokens=int(RL.tokens_of_rect(sm)[0]), y=y_cur, passed=bool(ev_cur["passed"]),
                    score=float(ev_cur["score"]), mae32=float(ev_cur.get("mae32", 0.0)), merges=merges,
                    prune_calls=calls)

    if not bool(ev_cur["passed"]):
        return _result(0)

    def _touches(bad, p):
        r0, c0, sh, sw = p
        return bool(bad[r0:r0 + sh, c0:c0 + sw].any())

    merges = 0
    rejected = set()

    for _ in range(max(1, int(max_rounds))):
        if max_calls > 0 and calls >= int(max_calls):
            break
        cands = [p for p in sibling_merge_candidates(sm, max_area=max_area) if p not in rejected]
        if not cands:
            break
        cands.sort(key=lambda p: (float(err_maps[(p[2], p[3])][p[0]:p[0] + p[2], p[1]:p[1] + p[3]].mean()), p))
        queue = [cands]
        merged_round = 0
        while queue:
            if max_calls > 0 and calls >= int(max_calls):
                break
            chunk = queue.pop(0)
            sm_try = apply_sibling_merges(sm, chunk)
            y_try, ev_try = _eval(sm_try)
            if bool(ev_try["passed"]):
                sm, y_cur, ev_cur = sm_try, y_try, ev_try
                merged_round += len(chunk)
                continue
            if len(chunk) == 1:
                rejected.add(chunk[0])
                continue
            bad = ev_try.get("bad8")
            if bad is not None and tuple(bad.shape[-2:]) == (RL.SIDE, RL.SIDE):
                bad_b = torch.as_tensor(bad).reshape(RL.SIDE, RL.SIDE).bool().cpu()
                dirty = [p for p in chunk if _touches(bad_b, p)]
                clean = [p for p in chunk if not _touches(bad_b, p)]
                if dirty and (clean or not bisect_dirty):
                    if bisect_dirty:
                        h = len(dirty) // 2
                        nxt = [dirty[:h], dirty[h:]] if h > 0 else [dirty]
                    else:
                        nxt = []                                # deferred: retried next round, not rejected
                    queue[:0] = ([clean] if clean else []) + nxt
                    continue
            h = len(chunk) // 2
            queue[:0] = [chunk[:h], chunk[h:]]
        merges += merged_round
        if merged_round == 0 or (max_calls > 0 and calls >= int(max_calls)):
            break                                               # nothing changed: deferred ones would fail again
    return _result(merges)


def sibling_merge_candidates(shape_map_1, max_area=1024):
    """Parents `(r0, c0, sh, sw)` of the current k-d tree (via `RL.kdtree_trace`) whose two children are both
    leaves, i.e. sibling pairs that merging restores to one valid dyadic tile.  `max_area` caps the merged tile."""
    nodes = RL.kdtree_trace(shape_map_1)
    leaf_set = {(r, c, sh, sw) for r, c, sh, sw, _, a in nodes if a == RL.ACT_LEAF}
    out = []
    for r0, c0, sh, sw, _, act in nodes:
        if act == RL.ACT_LEAF or sh * sw > int(max_area) or (sh, sw) not in RL.SHAPE_SET:
            continue
        if act == RL.ACT_SPLIT_H:
            kids = ((r0, c0, sh // 2, sw), (r0 + sh // 2, c0, sh // 2, sw))
        else:
            kids = ((r0, c0, sh, sw // 2), (r0, c0 + sw // 2, sh, sw // 2))
        if all(k in leaf_set for k in kids):
            out.append((r0, c0, sh, sw))
    return out


def spike_clamp_resid(resid, shape_map, k, min_area=64):
    """Per-tile spike clamp of a `[B, C, 64, 64]` residual on leaves with `sh * sw >= min_area`.

    Per tile and channel: `lp` = 3x3 box low-pass inside the tile (replicate padding at the tile border),
    `hp = r - lp`, `thr = k * median(|hp|)`. Cells with |hp| <= thr are unchanged; above thr the excess is squashed,
    `sign(hp) * (thr + thr * tanh((|hp| - thr) / thr))`, so the output is bounded by 2*thr. The low-pass is untouched.
    Returns the input elsewhere."""
    b, c = resid.shape[:2]
    sh_m, sw_m = shape_map[:, 0], shape_map[:, 1]
    out = resid
    for sh, sw in RL.COARSE_SHAPES:
        sh, sw = int(sh), int(sw)
        if sh * sw < int(min_area):
            continue
        cells = (sh_m == sh) & (sw_m == sw)
        if not bool(cells.any()):
            continue
        gh, gw = RL.SIDE // sh, RL.SIDE // sw
        t = resid.reshape(b, c, gh, sh, gw, sw).permute(0, 2, 4, 1, 3, 5).reshape(b * gh * gw, c, sh, sw)
        lp = F.avg_pool2d(F.pad(t, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
        hp = t - lp
        thr = (float(k) * hp.abs().flatten(2).median(dim=-1).values).clamp_min(1e-6)[..., None, None]
        a = hp.abs()
        # identity below thr; above it the excess is squashed (C1 at thr, bounded by 2*thr)
        t2 = lp + torch.where(a > thr, hp.sign() * (thr + thr * torch.tanh((a - thr) / thr)), hp)
        full = t2.reshape(b, gh, gw, c, sh, sw).permute(0, 3, 1, 4, 2, 5).reshape(b, c, RL.SIDE, RL.SIDE)
        out = torch.where(cells[:, None], full, out)
    return out


@torch.no_grad()
def split_kdtree_topdown_td(model, tiledown, z512_1, basis_dev, mean, std, dec, judge, max_rounds=48, *, seed=0):
    """Top-down greedy 2:1 k-d tree labeling with optional `RectTileDown` (CPU and CUDA safe): every round splits every
    splittable leaf overlapping a Judge `bad8` cell, until the Judge passes."""
    dev = z512_1.device
    use_cuda = dev.type == "cuda"
    z512_f = z512_1.float()
    mv, sv = mean.view(1, RL.CH, 1, 1).to(dev), std.view(1, RL.CH, 1, 1).to(dev)
    zn = (z512_f - mv) / sv
    rect_pyr_n = RTD.encode_rect_pyramid_td(zn, basis_dev, tiledown=tiledown)
    err_maps = compute_lift_err_maps(z512_f, rect_pyr_n, basis_dev, mv, sv, tilelift=getattr(model, "_tilelift", None),
                                     ac_floor=getattr(model, "_ac_floor", 0.20), ac_tau=getattr(model, "_ac_tau", 0.20))
    choose_hv = best_hv_split_fn(err_maps)

    def _decode(z):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
            return dec(z).float().clamp(-1, 1)

    leaves = [(0, 0, 32, 32), (0, 32, 32, 32), (32, 0, 32, 32), (32, 32, 32, 32)]
    calls = 0
    y_cur, ev_cur, sm = None, None, None

    for _ in range(max(1, int(max_rounds))):
        sm = torch.zeros((1, 2, RL.SIDE, RL.SIDE), dtype=torch.long, device=dev)
        for r0, c0, sh, sw in leaves:
            sm[0, 0, r0:r0 + sh, c0:c0 + sw] = sh
            sm[0, 1, r0:r0 + sh, c0:c0 + sw] = sw
        z = predict_rect_canvas_td(model, z512_f, rect_pyr_n, sm, basis_dev, mean, std, seed=seed)
        y_cur = _decode(z)
        ev_cur = judge(y_cur)
        calls += 1
        if ev_cur["passed"]:
            break

        splittable = [t for t in leaves if t[2] > 2 or t[3] > 2]
        if not splittable:
            break

        bad = ev_cur.get("bad8")
        to_split = []
        if bad is not None:
            bad_b = torch.as_tensor(bad, device=dev, dtype=torch.bool)
            to_split = [t for t in splittable if bool(bad_b[t[0]:t[0] + t[2], t[1]:t[1] + t[3]].any())]
        if not to_split:
            scored = sorted(
                splittable,
                key=lambda t: float(err_maps[(t[2], t[3])][t[0]:t[0] + t[2], t[1]:t[1] + t[3]].mean()),
                reverse=True,
            )
            to_split = scored[:max(1, len(scored) // 4)]

        split_set = set(to_split)
        nxt = []
        for t in leaves:
            if t not in split_set:
                nxt.append(t)
                continue
            r0, c0, sh, sw = t
            acts = [a for a in RL.valid_actions(sh, sw) if a != RL.ACT_LEAF]
            act = acts[0] if len(acts) == 1 else choose_hv(r0, c0, sh, sw)
            if act == RL.ACT_SPLIT_H:
                nxt.extend(((r0, c0, sh // 2, sw), (r0 + sh // 2, c0, sh // 2, sw)))
            else:
                nxt.extend(((r0, c0, sh, sw // 2), (r0, c0 + sw // 2, sh, sw // 2)))
        leaves = nxt

    toks = int(RL.tokens_of_rect(sm)[0])
    out = dict(
        shape_map=sm,
        tokens=toks,
        y=y_cur,
        passed=bool(ev_cur["passed"]),
        score=float(ev_cur["score"]),
        mae32=float(ev_cur.get("mae32", 0.0)),
        calls=calls,
    )
    return out


def _shapes(root):
    out = []
    todo = [(root, root)]
    seen = set()
    while todo:
        t = todo.pop()
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
        for a in RL.valid_actions(*t):
            if a == RL.ACT_SPLIT_H:
                todo.append((t[0] // 2, t[1]))
            elif a == RL.ACT_SPLIT_V:
                todo.append((t[0], t[1] // 2))
    return sorted(out, key=lambda t: t[0] * t[1])


def node_distortion(err_maps, sh, sw, beta, side=RL.SIDE, area_pow=0.0, tau=0.0):
    """D grid `[side/sh, side/sw]`: mean + beta * max of err_maps[(sh, sw)] over each aligned tile.
    `tau` (default 0 = spec): noise floor, D = relu(mean + beta*max - tau) (flat tiles whose lift error is below the
    Judge-tolerated floor cost nothing).  `area_pow` (default 0 = spec, area-independent) scales D by (sh*sw/4)**area_pow (1 = per-cell-summed error)."""
    if (sh, sw) == (2, 2) or (sh, sw) not in err_maps:
        return torch.zeros((side // sh, side // sw))
    e = err_maps[(sh, sw)].float().cpu()[None, None]
    d = (F.avg_pool2d(e, (sh, sw)) + beta * F.max_pool2d(e, (sh, sw)))[0, 0]
    if tau:
        d = (d - float(tau)).clamp_min(0.0)
    return d * (sh * sw / 4.0) ** float(area_pow) if area_pow else d


def rd_dp(err_maps, lam, beta, side=RL.SIDE, area_pow=0.0, tau=0.0):
    """Exact DP.  Returns (cost, act) dicts keyed by shape: cost grid and chosen action grid (0 leaf / 1 H / 2 V)."""
    root = side // 2
    cost, act = {}, {}
    for sh, sw in _shapes(root):
        leaf = node_distortion(err_maps, sh, sw, beta, side, area_pow, tau) + float(lam)
        best, a = leaf, torch.zeros_like(leaf, dtype=torch.long)
        for k in RL.valid_actions(sh, sw):
            if k == RL.ACT_SPLIT_H:
                c = cost[(sh // 2, sw)]
                sp = c[0::2] + c[1::2]
            elif k == RL.ACT_SPLIT_V:
                c = cost[(sh, sw // 2)]
                sp = c[:, 0::2] + c[:, 1::2]
            else:
                continue
            take = sp < best - 1e-12                   # ties keep the coarser choice (first wins H over V)
            best = torch.where(take, sp, best)
            a = torch.where(take, torch.full_like(a, k), a)
        cost[(sh, sw)], act[(sh, sw)] = best, a
    return cost, act


def rd_layout(err_maps, lam, beta, side=RL.SIDE, area_pow=0.0, tau=0.0):
    """Optimal layout as shape_map `[2, side, side]` (long, CPU)."""
    root = side // 2
    _, act = rd_dp(err_maps, lam, beta, side, area_pow, tau)
    sm = torch.zeros((2, side, side), dtype=torch.long)
    stack = [(r, c, root, root) for r in (0, root) for c in (0, root)]
    while stack:
        r0, c0, sh, sw = stack.pop()
        a = int(act[(sh, sw)][r0 // sh, c0 // sw])
        if a == RL.ACT_SPLIT_H:
            stack += [(r0, c0, sh // 2, sw), (r0 + sh // 2, c0, sh // 2, sw)]
        elif a == RL.ACT_SPLIT_V:
            stack += [(r0, c0, sh, sw // 2), (r0, c0 + sw // 2, sh, sw // 2)]
        else:
            sm[0, r0:r0 + sh, c0:c0 + sw] = sh
            sm[1, r0:r0 + sh, c0:c0 + sw] = sw
    return sm
