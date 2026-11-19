import random
import sys
import pathlib

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tessera.stream.grid import G, K, L, P, S, size  # noqa: E402


def random_layout(seed=0, p_split=.55, max_level=L - 1):
    """A random valid quadtree layout (lvl, two) on the 64 x 64 cell grid."""
    rng = random.Random(seed)
    lvl = torch.zeros(1, G, G, dtype=torch.long)
    two = torch.zeros(1, G, G, dtype=torch.bool)

    def walk(l, i, j):
        s = size(l)
        if l < max_level and rng.random() < p_split:
            for di in (0, 1):
                for dj in (0, 1):
                    walk(l + 1, 2 * i + di, 2 * j + dj)
            return
        lvl[0, i * s:(i + 1) * s, j * s:(j + 1) * s] = l
        if l < L - 1:
            two[0, i * s:(i + 1) * s, j * s:(j + 1) * s] = rng.random() < .3

    walk(0, 0, 0)
    return lvl, two


def fake_tab(x=None, seed=0):
    """A codec table with random payloads: enough for stream.tokens.pack (no bases needed)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(1, 3, S, S) if x is None else x
    tab = {}
    for l in range(L - 1):
        n = G // size(l)
        tab[l] = dict(n=n, mode={t: torch.randint(0, 4, (n * n,), generator=g) for t in (1, 2)},
                      grain=torch.randint(0, 2, (n * n,), generator=g).bool(),
                      pay={1: torch.randn(n * n, K, generator=g), 2: torch.randn(n * n, 2 * K, generator=g)}, rec={1: x, 2: x})
    tab[L - 1] = dict(rec={1: x, 2: x}, n=G)
    return tab


@pytest.fixture
def layout():
    return random_layout(3)
