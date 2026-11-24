"""The frozen Judge v5 (+ mae32 gate) bound to one image, as the encoder uses it."""
import pathlib

import numpy as np
import torch


DEFAULT_JUDGE_SCALE = 3.0


def _u8(y):                                          # [1,3,H,W] in [-1,1] -> uint8 [H,W,3] numpy
    return ((y[0].detach().float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()


class Judge:
    """Judge v5 + mae32 (tessera's fast path) bound to one image: judge(y [1,3,512,512] in [-1,1]) -> dict."""

    def __init__(self, x_u8, g_u8, dev, judge_scale=DEFAULT_JUDGE_SCALE, color_floor_de=None, color_max_floor_de=None,
                 struct_scale=None, mae32_limit=None):
        """EXPERIMENT options (all default None = off = the exact Judge of record at judge_scale):
        struct_scale: scale for every NON-colour gate (lpips/fine/dists_tex/seam) and the mae32 limit (2.5 * struct_scale);
                      colour gates keep judge_scale.
        mae32_limit: absolute mae32 limit overriding 2.5*(struct_)scale (all other gates untouched).
        color_floor_de / color_max_floor_de: floor the pass threshold (scale_k * T, Lab dE of excess error over the grid
                      decode) of color_mean256 / color_max256 by raising scale_k to >= floor / T.  fr.scale feeds the
                      verdict z-scores, score/argmax, bad8 attribution and passes_u8_batch (prune merge acceptance)."""
        from tessera.core import judge5_fast as Fa
        from tessera.core import judge5_sweep  # activates the mae32 gate
        from tessera.core import judge5_search as S

        scale = float(judge_scale)
        mae32_lim = 2.5 * scale
        fr = S.Frozen(str(pathlib.Path(__file__).resolve().parents[1] / "data" / "frozen_v5.json"))
        fr.T = float(fr.T) * scale
        self.struct_scale = None if struct_scale is None else float(struct_scale)
        self.color_floor_de = None if not color_floor_de else float(color_floor_de)
        self.color_max_floor_de = None if not color_max_floor_de else float(color_max_floor_de)
        if self.struct_scale is not None or self.color_floor_de or self.color_max_floor_de:
            keys = list(S.R.KEYS)
            fr.scale = np.array(fr.scale, np.float64, copy=True)
            if self.struct_scale is not None:
                mae32_lim = 2.5 * self.struct_scale
                for i, k in enumerate(keys):
                    if not k.startswith("color_"):
                        fr.scale[i] *= self.struct_scale / scale
            for k, fl in (("color_mean256", self.color_floor_de), ("color_max256", self.color_max_floor_de)):
                if fl:
                    i = keys.index(k)
                    fr.scale[i] = max(float(fr.scale[i]), fl / fr.T)
        if mae32_limit is not None:      # EXPERIMENT: absolute mae32 limit (RGB levels), overrides 2.5*scale; None = unchanged
            mae32_lim = float(mae32_limit)
        judge5_sweep.LIM = mae32_lim
        self.mae32_lim = mae32_lim
        self.fr = fr
        self.fc = Fa.FastContext(x_u8, g_u8, fr.thr, dev).bind(fr)
        self.fc._mae32_lim = self.mae32_lim
        self.judge_scale = scale
        self.dev = dev

    def thresholds(self):
        """{gate key: pass threshold in the gate's own units (colour: Lab dE)} + mae32 (RGB 0..255)."""
        from tessera.core import judge5_r2 as R
        d = {k: float(s) * float(self.fr.T) for k, s in zip(R.KEYS, self.fr.scale)}
        d["mae32"] = float(self.mae32_lim)
        return d

    def __call__(self, y):
        r = self.fc.evaluate_u8(torch.from_numpy(_u8(y)).to(self.dev), True)
        r["bad8"] = r["bad8"].bool().cpu()
        return r
