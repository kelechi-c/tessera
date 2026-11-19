"""Tightening sweep (N1): the v5 fast path with (i) threshold override J5_T (judge5_search.Frozen) and (ii) an extra gate `mae32`
(env J5_MAE32=2.5): max over the 16 x 16 grid of non-overlapping 32 px windows of (mean|y-x| - mean|g-x|), RGB 0..255, mean over
channels; the decode fails if that is > J5_MAE32.  Expressed as z = value / (J5_MAE32 / T) so the usual `z > T` rule applies.
Culprit map: the 8 px cells of every 32 px window above the limit, united with the usual v5 attribution.
Importing this module replaces judge5_fast.FastContext (used by JudgeFast.cellmap and judge5_fast.evaluate)."""
import os

import numpy as np
import torch
import torch.nn.functional as F

from . import judge5_fast as Fa
from . import judge5_r2 as R

_Orig = Fa.FastContext
LIM = float(os.environ.get("J5_MAE32", "2.5") or 0)   # tessera: mae32 gate on by default (2.5); J5_MAE32=0 turns it off


class FastContextM(_Orig):
    def __init__(self, x_u8, g_u8, thr, dev):
        super().__init__(x_u8, g_u8, thr, dev)
        self.ux_f = torch.from_numpy(np.ascontiguousarray(x_u8)).to(dev).float()
        self.ux = self.ux_f.to(torch.uint8)
        self.mg = self._mae_batch(torch.from_numpy(np.ascontiguousarray(g_u8)).to(dev)[None])[0]

    def _mae_batch(self, u_b):
        e = (u_b.float() - self.ux_f[None]).abs().mean(3)
        return F.avg_pool2d(e[:, None], 32, stride=32)[:, 0]

    def _mae(self, u):
        return self._mae_batch(u[None])[0]

    def evaluate_u8_batch(self, uys, attribute=True, chunk=8):
        if LIM <= 0:
            return super().evaluate_u8_batch(uys, attribute=attribute, chunk=chunk)
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
            m_b = self._mae_batch(uys) - self.mg[None]
            mx_b = m_b.amax((1, 2))
        z_b = (v_b / self._sc_t()[None]).cpu().numpy()
        mx_np = mx_b.cpu().numpy()
        zm_scale = LIM / self.fr_T
        out = []
        for idx in range(B):
            z = z_b[idx]
            i = int(z.argmax())
            score, arg = (float(z[i]), R.KEYS[i])
            mx = float(mx_np[idx])
            zm = mx / zm_scale
            if zm > score:
                score, arg = zm, "mae32"
            r = dict(score=score, argmax=arg, passed=bool(score <= self.fr_T), T=self.fr_T, mae32=mx)
            if attribute:
                bad = torch.zeros(Fa.G8, Fa.G8, dtype=torch.bool, device=self.dev)
                if not r["passed"]:
                    with torch.no_grad():
                        if z.max() > self.fr_T:
                            arrs_i = {k: val[idx] for k, val in arrs_b.items()}
                            bad |= self.attribution(arrs_i, z, self._fr)
                        bad |= (m_b[idx] > LIM).repeat_interleave(4, 0).repeat_interleave(4, 1)
                r["bad8"] = bad
            out.append(r)
        return out

    def passes_u8_batch(self, uys, pre_pass=None, chunk=8):
        if LIM <= 0:
            return super().passes_u8_batch(uys, pre_pass=pre_pass, chunk=chunk)
        B = int(uys.shape[0])
        if B == 0:
            return []
        with torch.no_grad():
            m_b = self._mae_batch(uys) - self.mg[None]
            mx_b = m_b.amax((1, 2))
            zm_scale = LIM / self.fr_T
            mae_ok = ((mx_b.double() / zm_scale) <= self.fr_T).cpu().tolist()
        if pre_pass is not None:
            mae_ok = [bool(a and b) for a, b in zip(mae_ok, pre_pass)]
        return super().passes_u8_batch(uys, pre_pass=mae_ok, chunk=chunk)

    def evaluate_u8(self, uy, attribute=True):
        return self.evaluate_u8_batch(uy[None], attribute=attribute)[0]


Fa.FastContext = FastContextM
