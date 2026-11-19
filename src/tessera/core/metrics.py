"""Local perceptual distortion maps: LPIPS (VGG, spatial) pooled to 16 px cells plus DINOv3-B/16 patch distance (1 - cos).
At 512 px both are 32 x 32 maps aligned with the FLUX.2 token grid. Windows of 4, 8, 16 cells (64, 128, 256 px), stride half.
"""
import torch
import torch.nn.functional as F

WINDOWS = (4, 8, 16)
DINO = "facebook/dinov3-vitb16-pretrain-lvd1689m"


class Judge:
    def __init__(self, dev, dino=True):
        import lpips

        self.dev = dev
        self.lp = lpips.LPIPS(net="vgg", spatial=True, verbose=False).to(dev).eval().requires_grad_(False)
        self.dino = None
        if dino:
            from transformers import AutoModel

            self.dino = AutoModel.from_pretrained(DINO).to(dev).eval().requires_grad_(False)
            self.nreg = 1 + getattr(self.dino.config, "num_register_tokens", 0)
        self.mean = torch.tensor([.485, .456, .406], device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor([.229, .224, .225], device=dev).view(1, 3, 1, 1)

    def feats(self, x):
        y = ((x + 1) / 2 - self.mean) / self.std
        with torch.autocast("cuda", torch.bfloat16, enabled=x.is_cuda):  # always bf16
            h = self.dino(pixel_values=y).last_hidden_state[:, self.nreg:].float()
        n = int(h.shape[1] ** .5)
        return F.normalize(h, dim=-1).transpose(1, 2).reshape(x.shape[0], -1, n, n)

    def cells(self, x, y, fy=None):
        """d per 16 px cell, [B, H/16, W/16]: LPIPS mean in the cell + DINO patch distance. fy = cached feats of y."""
        lp = F.avg_pool2d(self.lp(x, y), 16)[:, 0]
        if self.dino is None:
            return lp, lp, torch.zeros_like(lp)
        fx = self.feats(x)
        fy = self.feats(y) if fy is None else fy
        dn = 1 - (fx * fy).sum(1)
        return lp + dn, lp, dn


def windows(d):
    """d [B, n, n] -> list of window score maps, one per size."""
    return [F.avg_pool2d(d[:, None], k, stride=k // 2)[:, 0] for k in WINDOWS if k <= d.shape[-1]]


def excess(d_hat, d_ref):
    """Worst window excess and whole-image excess of d_hat over the grid reference d_ref."""
    w = max(float((a - b).amax()) for a, b in zip(windows(d_hat), windows(d_ref)))
    return w, float(d_hat.mean() - d_ref.mean())


class Dists:
    """DISTS (Ding et al. 2020) with local statistics: the same structure and texture terms, but the feature means,
    variances and covariances are taken over each window (64/128/256 px, stride half) instead of the whole image.
    It tolerates resampled texture (rock that looks like rock passes), unlike LPIPS which compares position by position."""

    C1 = C2 = 1e-6

    def __init__(self, dev):
        import os

        import DISTS_pytorch
        from DISTS_pytorch import DISTS

        m = DISTS(load_weights=False)
        w = torch.load(os.path.join(os.path.dirname(DISTS_pytorch.__file__), "weights.pt"), map_location="cpu")
        m.alpha.data, m.beta.data = w["alpha"], w["beta"]
        self.m = m.to(dev).eval().requires_grad_(False)
        ws = m.alpha.sum() + m.beta.sum()
        self.alpha = [a.to(dev) for a in torch.split(m.alpha.data / ws, m.chns, dim=1)]
        self.beta = [b.to(dev) for b in torch.split(m.beta.data / ws, m.chns, dim=1)]

    @torch.no_grad()
    def feats(self, x):  # x in [-1, 1]
        return [f.float() for f in self.m.forward_once((x.float() + 1) / 2)]

    def _terms(self, a, b, P):
        ma, mb = P(a), P(b)
        va, vb = (P(a * a) - ma * ma).clamp_min(0), (P(b * b) - mb * mb).clamp_min(0)
        cov = P(a * b) - ma * mb
        return (2 * ma * mb + self.C1) / (ma * ma + mb * mb + self.C1), (2 * cov + self.C2) / (va + vb + self.C2)

    @torch.no_grad()
    def windows(self, x, y=None, fy=None, G=32):
        """Returns ([B, nw, nw] per window size, matching windows() on a G x G cell map), image DISTS [B]."""
        fx, fy = self.feats(x), (self.feats(y) if fy is None else fy)
        H = x.shape[-1]
        outs = []
        for k in WINDOWS:
            nw = (G - k) // (k // 2) + 1
            tot = 0
            for i, (a, b) in enumerate(zip(fx, fy)):
                ker = max(2, round(k * (H // G) * a.shape[-1] / H))
                s1, s2 = self._terms(a, b, lambda t: F.avg_pool2d(t, ker, max(1, ker // 2)))
                s = (self.alpha[i] * s1).sum(1) + (self.beta[i] * s2).sum(1)
                if s.shape[-1] != nw:
                    s = F.interpolate(s[:, None], size=(nw, nw), mode="bilinear", align_corners=True)[:, 0]
                tot = tot + s
            outs.append(1 - tot)
        img = 0
        for i, (a, b) in enumerate(zip(fx, fy)):
            s1, s2 = self._terms(a, b, lambda t: t.mean((2, 3), keepdim=True))
            img = img + (self.alpha[i] * s1).sum(1) + (self.beta[i] * s2).sum(1)
        return outs, 1 - img.flatten()
