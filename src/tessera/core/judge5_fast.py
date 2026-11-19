"""Judge v5, search-speed path (round 3, S2).  Same gates, same frozen T / scales / grain threshold as judge5 + judge5_r2 +
judge5_search, but an evaluation computes only the y-side terms and stays on the device:

  * x- and g-side quantities (LPIPS / DISTS trunk outputs of x, pooled DISTS statistics of x, Lab-blur of x, fine maps, jump
    maps, masks, g's own distances) are computed once per image in `FastContext`;
  * Lab (float64), tap-sum blur, fine energy, seam, window pooling, the 18 gate values and the attribution map are torch
    ops on the device with a fixed operation order; one `.cpu()` of the 18 gate values per evaluation (the bad map stays on
    the device unless it is needed);
  * the candidate is quantised to uint8 on the device (same clamp / round as the PNG writer), no PNG in the loop;
  * `JudgeFast` caches the verdict by the uint8 decode, so an exact repeat costs one hash.
LPIPS and DISTS both use a VGG16 trunk but with different input scaling and pooling (max vs L2) after the first stage, so
the trunk is not shared.
"""
import hashlib
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import judge5 as J
from . import judge5_r2 as R
from . import judge5_search as S

KS = (16, 64, 256)
G8 = 64
BIG = 1e4


def _lab_t(x01, d65=None):
    """sRGB [B, 3, H, W] in [0, 1] -> Lab float32, computed in float64 elementwise (no reductions: thread/stream independent)."""
    a = x01.double()
    lin = torch.where(a > .04045, torch.pow((a + .055) / 1.055, 2.4), a / 12.92)
    M = [[.4124564, .3575761, .1804375], [.2126729, .7151522, .0721750], [.0193339, .1191920, .9503041]]
    xyz = torch.cat([(M[i][0] * lin[:, 0:1] + M[i][1] * lin[:, 1:2]) + M[i][2] * lin[:, 2:3] for i in range(3)], 1)
    if d65 is None:
        d65 = torch.tensor([.95047, 1., 1.08883], dtype=torch.float64, device=a.device).view(1, 3, 1, 1)
    xyz = xyz / d65
    f = torch.where(xyz > .008856, torch.pow(xyz.clamp_min(1e-8), 1.0 / 3.0), 7.787 * xyz + 16 / 116)
    lab = torch.cat([116 * f[:, 1:2] - 16, 500 * (f[:, 0:1] - f[:, 1:2]), 200 * (f[:, 1:2] - f[:, 2:3])], 1)
    return lab.float()


def _lp_feats(lp, x_m11):
    from lpips import normalize_tensor
    outs = lp.net.forward(lp.scaling_layer(x_m11))
    return [normalize_tensor(o) for o in outs]


def _lp_map(lp, fx, fy, hw):
    from lpips.lpips import upsample
    val = None
    for k in range(lp.L):
        r = upsample(lp.lins[k]((fx[k] - fy[k]) ** 2), out_HW=hw)
        val = r if val is None else val + r
    return val


def _plan(dx):
    """Per DISTS layer: {k: (kernel, stride)} in layer pixels (the same windows as judge5._dists_terms) and the base block bb
    = the smallest stride: the feature map is block-averaged once at bb, every window size is then an average of block means
    (exact: equal blocks), so only one pass touches the full-resolution maps."""
    H = dx[0].shape[-2]
    out = []
    for a in dx:
        s = H // a.shape[-2]
        ks = {k: (k // s, max(1, (k // s) // 2)) for k in KS if k // s >= 2}
        out.append((ks, min(st for kk, st in ks.values()) if ks else None))
    return out


def _levels(a, plan_i):
    ks, bb = plan_i
    if not ks:
        return {}
    base = F.avg_pool2d(a, bb, stride=bb) if bb > 1 else a
    return {k: F.avg_pool2d(base, kk // bb, stride=st // bb) for k, (kk, st) in ks.items()}


def _dists_tex_b(alpha, xlev, ylev):
    """Batched DISTS texture term map [B, nw, nw] per window size."""
    C1 = 1e-6
    out = {}
    for k in KS:
        tex = 0
        for i, (lx, ly) in enumerate(zip(xlev, ylev)):
            if k not in lx:
                continue
            ma, mb = lx[k], ly[k]
            tex = tex + (alpha[i] * (1 - (2 * ma * mb + C1) / (ma * ma + mb * mb + C1))).sum(1)
        out[k] = tex
    return out


def _dists_tex(alpha, xlev, ylev):
    """DISTS texture term map [nw, nw] per window size: xlev / ylev = per-layer {k: pooled mean} (the structure term is not a gate)."""
    return {k: v[0] for k, v in _dists_tex_b(alpha, xlev, ylev).items()}


def _luma(u8):
    a = u8.float()
    return a[..., 0] * .299 + a[..., 1] * .587 + a[..., 2] * .114


def _jumps(L):
    H, W = L.shape[-2:]
    prefix = L.shape[:-2]
    dv = (L[..., :, 1:] - L[..., :, :-1]).abs().reshape(*prefix, H // 16, 16, W - 1).mean(-2)
    dh = (L[..., 1:, :] - L[..., :-1, :]).abs().reshape(*prefix, H - 1, W // 16, 16).mean(-1)
    return dv, dh


def _blockiness(J_, axis):
    b = torch.zeros_like(J_)
    if axis == 1:
        b[..., :, 1:-1] = torch.clamp_min(J_[..., :, 1:-1] - .5 * (J_[..., :, :-2] + J_[..., :, 2:]), 0)
    else:
        b[..., 1:-1, :] = torch.clamp_min(J_[..., 1:-1, :] - .5 * (J_[..., :-2, :] + J_[..., 2:, :]), 0)
    return b


class FastContext:
    """Everything about (x, g) that an evaluation needs, on the device."""

    def __init__(self, x_u8, g_u8, thr, dev):
        """x- and g-side quantities, all computed with the same device functions as the y side (so y = g gives exactly 0)."""
        self.dev, self.thr = dev, thr
        self._fr = self.fr_scale = self.fr_T = self._sc = None
        with torch.no_grad(), torch.backends.mkldnn.flags(enabled=False):
            J.DEV = str(dev)
            M = J._models(False)
            if next(M["lp"].parameters()).device != torch.device(dev):
                M["lp"] = M["lp"].to(dev)
                M["dists"] = M["dists"].to(dev)
            self.M, self.lp = M, M["lp"]
            self.alpha = [a.to(dev) for a in M["alpha"]]
            self.H, self.W = x_u8.shape[:2]
            J._grid_ok(self.H, self.W)
            self._d65 = torch.tensor([.95047, 1., 1.08883], dtype=torch.float64, device=dev).view(1, 3, 1, 1)
            k_g = J._gauss_kernel(J.SIG_LAB).to(dev)
            self._blur_r = (k_g.numel() - 1) // 2
            self._blur_kx = k_g.view(1, 1, 1, -1).expand(3, 1, 1, -1).contiguous()
            self._blur_ky = k_g.view(1, 1, -1, 1).expand(3, 1, -1, 1).contiguous()
            cols = torch.arange(1, self.W, device=dev)
            self._seam_cols = cols[cols % J.SEAM_PERIOD == 0]
            rows = torch.arange(1, self.H, device=dev)
            self._seam_rows = rows[rows % J.SEAM_PERIOD == 0]

            ux, ug = (torch.from_numpy(np.ascontiguousarray(a)).to(dev) for a in (x_u8, g_u8))
            x01, g01 = (u[None].permute(0, 3, 1, 2).float() / 255.0 for u in (ux, ug))
            hw = x01.shape[-2:]
            # Keep x and g at B=1 so Conv2d reduction order matches B=1 evaluation (y == g => score == 0.0).
            self.lpx = _lp_feats(self.lp, x01 * 2 - 1)
            lpg = _lp_feats(self.lp, g01 * 2 - 1)
            self.tg_lp = _lp_map(self.lp, self.lpx, lpg, hw)
            self.dx = [f.float() for f in M["dists"].forward_once(x01)]
            dg = [f.float() for f in M["dists"].forward_once(g01)]
            self.plan = _plan(self.dx)
            self.xlev = [_levels(a, p) for a, p in zip(self.dx, self.plan)]
            self.tg_d = _dists_tex(self.alpha, self.xlev, [_levels(a, p) for a, p in zip(dg, self.plan)])
            self.lx = self._blur(_lab_t(x01, self._d65))
            self.tg_color = (self._blur(_lab_t(g01, self._d65)) - self.lx).square().sum(1, keepdim=True).sqrt()
            self.sx = J._fine_std_grid(x01)
            self.act = (self.sx > J.sharp.ACT).float()
            self.ratio_g = J._fine_std_grid(g01) / (self.sx + 1e-4)
            self.den = {k: F.avg_pool2d(self.act, (k - 16) // 8 + 1, stride=k // 16)[0, 0] for k in KS}
            self.w = None if thr is None else (torch.from_numpy(J.grain_windows_16(x_u8)).to(dev) < thr).float()
            self.jx, self.jg = _jumps(_luma(ux)), _jumps(_luma(ug))
            self.bg = [_blockiness(b - a, 1 - ax) for ax, (a, b) in enumerate(zip(self.jx, self.jg))]
            self.x_u8, self.g_u8 = x_u8, g_u8

    def _blur(self, x):
        if str(self.dev) == "cpu":
            return J._blur(x, J.SIG_LAB)
        r = self._blur_r
        out = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), self._blur_kx, groups=3)
        return F.conv2d(F.pad(out, (0, 0, r, r), mode="reflect"), self._blur_ky, groups=3)

    # ---- y side -----------------------------------------------------------------
    def _seam_batch(self, uys):
        jy = _jumps(_luma(uys))
        B = uys.shape[0]
        S_ = torch.zeros(B, self.H, self.W, device=self.dev)
        d0 = _blockiness(jy[0] - self.jx[0][None], 1) - self.bg[0][None]
        if self._seam_cols.numel():
            cols = self._seam_cols
            S_[:, :, cols] = torch.maximum(S_[:, :, cols], d0[:, :, cols - 1].repeat_interleave(16, 1))
        d1 = _blockiness(jy[1] - self.jx[1][None], 0) - self.bg[1][None]
        if self._seam_rows.numel():
            rows = self._seam_rows
            S_[:, rows, :] = torch.maximum(S_[:, rows, :], d1[:, rows - 1, :].repeat_interleave(16, 2))
        t = S_[:, None]
        return {k: F.max_pool2d(t, k, stride=k // 2)[:, 0] for k in (16, 64)}

    def _seam(self, uy):
        return {k: v[0] for k, v in self._seam_batch(uy[None]).items()}

    def window_terms_batch(self, uys):
        """uint8 [B, H, W, 3] device tensor -> dict of batched window maps [B, nh, nw] (device)."""
        y01 = uys.permute(0, 3, 1, 2).float() / 255.0
        r = {}
        lpy = _lp_map(self.lp, self.lpx, _lp_feats(self.lp, y01 * 2 - 1), y01.shape[-2:])
        dy = [f.float() for f in self.M["dists"].forward_once(y01)]
        ly = self._blur(_lab_t(y01, self._d65))
        color_y = (ly - self.lx).square().sum(1, keepdim=True).sqrt()
        ratio_y = J._fine_std_grid(y01) / (self.sx + 1e-4)
        tex = _dists_tex_b(self.alpha, self.xlev, [_levels(a, p) for a, p in zip(dy, self.plan)])
        lp_diff = lpy - self.tg_lp
        col_diff = color_y - self.tg_color
        fine_diff = (self.ratio_g - ratio_y) * self.act
        for k in KS:
            r[f"lpips_{k}"] = F.avg_pool2d(lp_diff, k, stride=k // 2)[:, 0]
            r[f"dists_tex_{k}"] = tex[k] - self.tg_d[k][None]
            r[f"color_{k}"] = F.avg_pool2d(col_diff, k, stride=k // 2)[:, 0]
            kk, st = (k - 16) // 8 + 1, k // 16
            num = F.avg_pool2d(fine_diff, kk, stride=st)[:, 0]
            den = self.den[k][None]
            r[f"fine_{k}"] = torch.where(den > 0, num / den.clamp_min(1e-9), torch.zeros_like(num))
        sm = self._seam_batch(uys)
        r["seam_16"], r["seam_64"] = sm[16], sm[64]
        return r

    def window_terms(self, uy):
        """uint8 [H, W, 3] device tensor -> dict of window maps (device) for the verdict."""
        return {k: v[0] for k, v in self.window_terms_batch(uy[None]).items()}

    def _block_batch(self, a16, k):
        n, st = {64: 7, 256: 31}[k], k // 16
        w = self.w[None, None]
        num = F.avg_pool2d((a16 * self.w[None])[:, None], n, stride=st)[:, 0]
        den = F.avg_pool2d(w, n, stride=st)[:, 0]
        return torch.where(den > 0, num / den.clamp_min(1e-30), torch.zeros_like(num))

    def _block(self, a16, k):
        return self._block_batch(a16[None], k)[0]

    def _term_gate_vals(self, t, a16, a64, a256):
        """Return ([B, 4] tensor of gate values in float64, (g16, g64, g256) tensors for attribution)."""
        if t == "lpips" and self.thr is not None:
            m16 = torch.where(self.w[None] > 0, a16, torch.full_like(a16, -math.inf))
            mx = m16.amax((1, 2))
            mx = torch.where(torch.isinf(mx), torch.zeros_like(mx), mx)
            b64, b256 = self._block_batch(a16, 64), self._block_batch(a16, 256)
            cols = [mx, b64.amax((1, 2)), b256.amax((1, 2)), b256.mean((1, 2))]
            return torch.stack([c.double() for c in cols], dim=1), (m16, b64, b256)
        cols = [a16.amax((1, 2)), a64.amax((1, 2)), a256.amax((1, 2)), a256.mean((1, 2))]
        return torch.stack([c.double() for c in cols], dim=1), (a16, a64, a256)

    def gates_batch(self, r):
        """batched window maps -> (tensor [B, 18] in R.KEYS order on device, batched arrs dict)."""
        blocks, arrs = [], {}
        for t in R.TERMS:
            v4, tup = self._term_gate_vals(t, r[f"{t}_16"], r[f"{t}_64"], r[f"{t}_256"])
            blocks.append(v4)
            arrs[(t, 16)], arrs[(t, 64)], arrs[(t, 256)] = tup
        s16, s64 = r["seam_16"], r["seam_64"]
        blocks.append(torch.stack([s16.amax((1, 2)).double(), s64.amax((1, 2)).double()], dim=1))
        arrs[("seam", 16)], arrs[("seam", 64)] = s16, s64
        return torch.cat(blocks, dim=1), arrs

    def gates(self, r):
        """window maps -> (vector [18] in R.KEYS order on the device, per-gate arrays used by the attribution)."""
        v_b, arrs_b = self.gates_batch({k: val[None] for k, val in r.items()})
        return v_b[0], {k: val[0] for k, val in arrs_b.items()}

    def attribution(self, arrs, z, fr):
        """Bad 8 px cells [64, 64] bool (device); the same rules as judge5_search.attribution."""
        dev = self.dev
        bad = torch.zeros(G8, G8, dtype=torch.bool, device=dev)
        sc = dict(zip(R.KEYS, fr.scale))
        for key, zg in zip(R.KEYS, z):
            if zg <= fr.T:
                continue
            term, kind = key.rsplit("_", 1)
            if kind == "max16":
                m = ((arrs[(term, 16)] / sc[key]) > fr.T).float()[None, None]
                bad |= F.max_pool2d(F.pad(m, (1, 1, 1, 1)), 2, stride=1)[0, 0] > 0
                continue
            a16 = arrs[(term, 16)]
            if term == "lpips":
                a16 = torch.where(torch.isfinite(a16), a16, torch.zeros_like(a16))
            cv = F.max_pool2d(F.pad(a16[None, None], (1, 1, 1, 1), value=-math.inf), 2, stride=1)[0, 0] / sc[f"{term}_max16"]
            if kind in ("max64", "max256"):
                k = int(kind[3:])
                wins = arrs[(term, k)] / sc[key]
                n, st = (8, 4) if k == 64 else (32, 16)
                u = F.unfold(cv[None, None], n, stride=st)[0]                     # [n*n, nwin]
                srt = u.sort(0).values
                m = n * n
                med = (srt[m // 2 - 1] + srt[m // 2]) / 2
                sel = (wins > fr.T).reshape(-1)[None] & (u > med[None])
                bad |= F.fold(sel.float()[None], (G8, G8), n, stride=st)[0, 0] > 0
            else:
                thr = torch.quantile(cv.reshape(-1), .9)
                bad |= cv >= thr
        return bad

    def _sc_t(self):
        if self._sc is None:
            self._sc = torch.from_numpy(self.fr_scale).to(self.dev)
        return self._sc

    def evaluate_u8_batch(self, uys, attribute=True, chunk=8):
        """Evaluate a batch [B, H, W, 3] of uint8 candidates in one GPU pass (chunked by `chunk` for VRAM safety)."""
        B = int(uys.shape[0])
        if B == 0:
            return []
        if B > chunk:
            out = []
            for a in range(0, B, chunk):
                out.extend(self.evaluate_u8_batch(uys[a:a + chunk], attribute=attribute, chunk=chunk))
            return out
        with torch.no_grad(), torch.backends.mkldnn.flags(enabled=False):
            rb = self.window_terms_batch(uys)
            v_b, arrs_b = self.gates_batch(rb)
        z_b = (v_b / self._sc_t()[None]).cpu().numpy()
        out = []
        for idx in range(B):
            z = z_b[idx]
            i = int(z.argmax())
            r = dict(score=float(z[i]), argmax=R.KEYS[i], passed=bool(z[i] <= self.fr_T), T=self.fr_T)
            if attribute:
                if r["passed"]:
                    r["bad8"] = torch.zeros(G8, G8, dtype=torch.bool, device=self.dev)
                else:
                    with torch.no_grad():
                        arrs_i = {k: val[idx] for k, val in arrs_b.items()}
                        r["bad8"] = self.attribution(arrs_i, z, self._fr)
            out.append(r)
        return out

    def passes_u8_batch(self, uys, pre_pass=None, chunk=8):
        """Bit-exact pass/fail verdict via cascaded early-exit (seam -> fine -> color -> lpips -> dists_tex)."""
        B = int(uys.shape[0])
        if B == 0:
            return []
        alive = [i for i in range(B) if pre_pass is None or pre_pass[i]]
        if not alive:
            return [False] * B
        if len(alive) > chunk:
            res = [False] * B
            for a in range(0, len(alive), chunk):
                sub = alive[a:a + chunk]
                sub_ok = self.passes_u8_batch(uys[sub], chunk=chunk)
                for idx, flag in zip(sub, sub_ok):
                    res[idx] = flag
            return res

        sc = self._sc_t()
        T = self.fr_T
        offs = {t: R.TERMS.index(t) * 4 for t in R.TERMS}

        def _narrow(cur_alive, vals, sl):
            keep = ((vals / sc[sl][None]) <= T).all(dim=1).cpu().tolist()
            return [idx for idx, k in zip(cur_alive, keep) if k]

        with torch.no_grad(), torch.backends.mkldnn.flags(enabled=False):
            sm = self._seam_batch(uys[alive])
            svals = torch.stack([sm[16].amax((1, 2)).double(), sm[64].amax((1, 2)).double()], dim=1)
            alive = _narrow(alive, svals, slice(16, 18))
            if not alive:
                return [False] * B

            y01 = uys[alive].permute(0, 3, 1, 2).float() / 255.0
            ratio_y = J._fine_std_grid(y01) / (self.sx + 1e-4)
            fine_diff = (self.ratio_g - ratio_y) * self.act
            fm = {
                k: torch.where(self.den[k][None] > 0,
                               F.avg_pool2d(fine_diff, (k - 16) // 8 + 1, stride=k // 16)[:, 0] / self.den[k][None].clamp_min(1e-9),
                               0.0)
                for k in KS
            }
            v_fine, _ = self._term_gate_vals("fine", fm[16], fm[64], fm[256])
            surv_local = _narrow(list(range(len(alive))), v_fine, slice(offs["fine"], offs["fine"] + 4))
            alive = [alive[i] for i in surv_local]
            if not alive:
                return [False] * B
            y01 = y01[surv_local]

            ly = self._blur(_lab_t(y01, self._d65))
            col_diff = (ly - self.lx).square().sum(1, keepdim=True).sqrt() - self.tg_color
            cm = {k: F.avg_pool2d(col_diff, k, stride=k // 2)[:, 0] for k in KS}
            v_col, _ = self._term_gate_vals("color", cm[16], cm[64], cm[256])
            surv_local = _narrow(list(range(len(alive))), v_col, slice(offs["color"], offs["color"] + 4))
            alive = [alive[i] for i in surv_local]
            if not alive:
                return [False] * B
            y01 = y01[surv_local]

            lpy = _lp_map(self.lp, self.lpx, _lp_feats(self.lp, y01 * 2 - 1), y01.shape[-2:])
            lp_diff = lpy - self.tg_lp
            lm = {k: F.avg_pool2d(lp_diff, k, stride=k // 2)[:, 0] for k in KS}
            v_lp, _ = self._term_gate_vals("lpips", lm[16], lm[64], lm[256])
            surv_local = _narrow(list(range(len(alive))), v_lp, slice(offs["lpips"], offs["lpips"] + 4))
            alive = [alive[i] for i in surv_local]
            if not alive:
                return [False] * B
            y01 = y01[surv_local]

            dy = [f.float() for f in self.M["dists"].forward_once(y01)]
            tex = _dists_tex_b(self.alpha, self.xlev, [_levels(a, p) for a, p in zip(dy, self.plan)])
            dm = {k: tex[k] - self.tg_d[k][None] for k in KS}
            v_d, _ = self._term_gate_vals("dists_tex", dm[16], dm[64], dm[256])
            alive = _narrow(alive, v_d, slice(offs["dists_tex"], offs["dists_tex"] + 4))

        out = [False] * B
        for idx in alive:
            out[idx] = True
        return out

    def evaluate_u8(self, uy, attribute=True):
        return self.evaluate_u8_batch(uy[None], attribute=attribute)[0]

    def bind(self, fr):
        self._fr, self.fr_scale, self.fr_T = fr, np.asarray(fr.scale, np.float64), fr.T
        self._sc = torch.from_numpy(self.fr_scale).to(self.dev)
        return self


def quantise_t(xm11):
    """[1, 3, H, W] in [-1, 1] (device) -> uint8 [H, W, 3] device tensor, as the PNG writer rounds."""
    return ((xm11[0].detach().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0)


def quantise_batch_t(xm11):
    """[B, 3, H, W] in [-1, 1] (device) -> uint8 [B, H, W, 3] device tensor."""
    return ((xm11.detach().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).contiguous()


def evaluate(x_u8, g_u8, y_u8, frozen, dev=None, attribute=False):
    """Verdict of record on three 8-bit images, fast path (builds the context each time)."""
    fr = frozen if isinstance(frozen, S.Frozen) else S.Frozen(frozen)
    dev = dev or J.DEV
    fc = FastContext(x_u8, g_u8, fr.thr, dev).bind(fr)
    r = fc.evaluate_u8(torch.from_numpy(np.ascontiguousarray(y_u8)).to(dev), attribute)
    if attribute:
        r["bad8"] = r["bad8"].cpu().numpy()
    return r


class JudgeFast(S.Judge5):
    """Drop-in for hires.run_oracle / vae16.run_oracle_vae16.  Caches verdicts by the uint8 decode."""

    def __init__(self, dev, frozen_json):
        super().__init__(dev, frozen_json)
        self.fc = None
        self.cache = {}
        self.pass_cache = {}
        self.n_cached = 0

    @torch.no_grad()
    def cellmap(self, xh, x, fx):
        key = (tuple(x.shape), round(float(x.sum()), 3), round(float(x[..., ::37, ::41].sum()), 3))
        if key != self._key:
            self._key = key
            self.cache = {}
            self.pass_cache = {}
            ux, ug = S.quantise(x), S.quantise(xh)
            self.v5 = None
            self.fc = FastContext(ux, ug, self.fr.thr, self.dev).bind(self.fr)
            self.last = None
            return torch.zeros(1, G8, G8, device=xh.device)
        t0 = time.time()
        uy = quantise_t(xh)
        h = hashlib.md5(uy.cpu().numpy().tobytes()).digest()
        r = self.cache.get(h)
        if r is None:
            r = self.fc.evaluate_u8(uy)
            self.cache[h] = r
            self.pass_cache[h] = bool(r["passed"])
        else:
            self.n_cached += 1
        self.sec += time.time() - t0
        self.n_eval += 1
        self.last = r
        return torch.where(r["bad8"].to(xh.device), BIG, 0.0)[None]

    @torch.no_grad()
    def evaluate_batch(self, xhs):
        """Evaluate a batch of candidate decodes [B, 3, H, W] (or list of [1, 3, H, W]).
        Returns list of (score, passed, bad8_bool_on_dev) matching `cellmap` + `last`."""
        if not len(xhs):
            return []
        t0 = time.time()
        xhs_t = xhs if torch.is_tensor(xhs) and xhs.ndim == 4 else torch.cat(list(xhs), 0)
        uys = quantise_batch_t(xhs_t.to(self.dev))
        uys_np = uys.cpu().numpy()
        hashes = [hashlib.md5(uys_np[i].tobytes()).digest() for i in range(len(uys_np))]
        results = [self.cache.get(h) for h in hashes]
        seen = {}
        uniq_pos = []
        for i, (h, r) in enumerate(zip(hashes, results)):
            if r is None and h not in seen:
                seen[h] = len(uniq_pos)
                uniq_pos.append(i)
        self.n_cached += len(results) - len(uniq_pos)
        if uniq_pos:
            miss_res = self.fc.evaluate_u8_batch(uys[uniq_pos], attribute=True)
            for pos, res in zip(uniq_pos, miss_res):
                h = hashes[pos]
                self.cache[h] = res
                self.pass_cache[h] = bool(res["passed"])
            for i, h in enumerate(hashes):
                if results[i] is None:
                    results[i] = self.cache[h]
        self.sec += time.time() - t0
        self.n_eval += len(results)
        self.last = results[-1]
        return [(r["score"], r["passed"], r["bad8"]) for r in results]

    @torch.no_grad()
    def passes_batch(self, xhs):
        """Fast pass/fail batch verdict with cascaded early-exit (skips LPIPS/DISTS on early failures)."""
        if not len(xhs):
            return []
        t0 = time.time()
        xhs_t = xhs if torch.is_tensor(xhs) and xhs.ndim == 4 else torch.cat(list(xhs), 0)
        uys = quantise_batch_t(xhs_t.to(self.dev))
        uys_np = uys.cpu().numpy()
        hashes = [hashlib.md5(uys_np[i].tobytes()).digest() for i in range(len(uys_np))]
        out = [self.pass_cache.get(h) for h in hashes]
        seen = {}
        uniq_pos = []
        for i, (h, p) in enumerate(zip(hashes, out)):
            if p is None and h not in seen:
                seen[h] = len(uniq_pos)
                uniq_pos.append(i)
        self.n_cached += len(out) - len(uniq_pos)
        if uniq_pos:
            flags = self.fc.passes_u8_batch(uys[uniq_pos])
            for pos, flag in zip(uniq_pos, flags):
                self.pass_cache[hashes[pos]] = bool(flag)
            for i, h in enumerate(hashes):
                if out[i] is None:
                    out[i] = self.pass_cache[h]
        self.sec += time.time() - t0
        self.n_eval += len(out)
        return out
