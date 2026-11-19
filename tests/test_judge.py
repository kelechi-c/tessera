import json

import numpy as np
import pytest

from tessera import judge as J
from tessera.core import judge5_r2 as R, judge5_search as S


def test_frozen_json_is_complete():
    d = json.load(open(J.frozen_path()))["frozen"]
    assert set(d["scale"]) == set(R.KEYS) and len(R.KEYS) == 18
    fr = S.Frozen(J.frozen_path())
    assert fr.T == pytest.approx(1.7759080042869313) and fr.thr == pytest.approx(0.1) and (fr.scale > 0).all()


def test_gate_vector_on_synthetic_features():
    f = {}
    for t in R.TERMS:
        f[f"{t}_16"] = np.full((63, 63), .1, np.float32)
        f[f"{t}_64"] = np.full((57, 57), .1, np.float32)
        f[f"{t}_256"] = np.full((33, 33), .1, np.float32)
    f["grain_16"] = np.zeros((63, 63), np.float32)
    f["seam_16"], f["seam_64"] = np.zeros((63, 63)), np.zeros((57, 57))
    f["fine_16"][5, 5] = 2.0
    v = R.gate_vector(f, f, .1)
    assert v.shape == (18,) and v[R.KEYS.index("fine_max16")] == pytest.approx(2.0) and v[R.KEYS.index("color_max16")] == pytest.approx(.1)


@pytest.fixture(scope="module")
def images():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:512, :512]
    x = np.stack([(xx / 2) % 256, (yy / 2) % 256, ((xx + yy) / 4) % 256], -1).astype(np.uint8)
    x[100:200, 100:300] = (rng.random((100, 200, 3)) * 255).astype(np.uint8)
    return x


def test_judge_gates(images):
    x = images
    same = J.judge(x, x, x, device="cpu")
    assert same["passed"] and same["score"] == pytest.approx(0.0, abs=1e-6)      # candidate == original passes
    bad = x.copy()
    bad[256:320, 256:320] = 255 - bad[256:320, 256:320]                          # a 64 px block replaced by its negative
    r = J.judge(x, x, bad, device="cpu")
    assert not r["passed"] and r["score"] > r["T"]
    shifted = np.roll(x, 3, axis=1)                                              # candidate worse than the VAE round trip everywhere
    assert not J.judge(x, x, shifted, device="cpu")["passed"]


def test_mae32_gate_fires_alone(images):
    x = images
    cand = x.copy()
    cand[0:32, 0:32] = np.clip(cand[0:32, 0:32].astype(int) + 8, 0, 255).astype(np.uint8)    # +8 grey levels in one 32 px window
    r = J.judge(x, x, cand, device="cpu")
    assert r["mae32"] == pytest.approx(8.0, abs=.2) and not r["passed"] and r["argmax"] == "mae32"
