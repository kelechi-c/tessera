"""CPU unit tests for 2:1 k-d tree rectangular layouts, orthonormal DC+plane+AC basis, BreadthGenRect, and loops."""
import pathlib
import tempfile

import torch

from tessera.gen import data as GD
from tessera.gen import model as GM
from tessera.gen import rect_model as RM
from tessera.latent import layout as LY
from tessera.latent import rect as RL
from tessera.latent import rect_router as RR
from tessera.latent import rect_tiledown as RTD


def _sample_kdtree_map(b=3, seed=0):
    gen = torch.Generator().manual_seed(seed)
    out = torch.zeros((b, 2, RL.SIDE, RL.SIDE), dtype=torch.long)
    for bi in range(b):
        frontier = [(0, 0, 32, 32), (0, 32, 32, 32), (32, 0, 32, 32), (32, 32, 32, 32)]
        while frontier:
            nxt = []
            for r0, c0, sh, sw in frontier:
                acts = RL.valid_actions(sh, sw)
                if len(acts) == 1 or float(torch.rand(1, generator=gen)) > 0.65:
                    out[bi, 0, r0:r0 + sh, c0:c0 + sw] = sh
                    out[bi, 1, r0:r0 + sh, c0:c0 + sw] = sw
                else:
                    sa = [a for a in acts if a != RL.ACT_LEAF]
                    a = sa[int(torch.randint(len(sa), (1,), generator=gen))]
                    if a == RL.ACT_SPLIT_H:
                        nxt += [(r0, c0, sh // 2, sw), (r0 + sh // 2, c0, sh // 2, sw)]
                    else:
                        nxt += [(r0, c0, sh, sw // 2), (r0, c0 + sw // 2, sh, sw // 2)]
            frontier = nxt
    return out


def test_rect_layout_and_kdtree_trace():
    sm = _sample_kdtree_map(4, seed=7)
    assert RL.validate_rect(sm)
    for bi in range(4):
        leaves = RL.leaves_rect(sm[bi])
        assert len(leaves) == int(RL.tokens_of_rect(sm[bi])[0])
        nodes = RL.kdtree_trace(sm[bi])
        trace_leaves = [(r, c, sh, sw) for r, c, sh, sw, _, a in nodes if a == RL.ACT_LEAF]
        assert len(trace_leaves) == len(leaves)
        # Every visited node must obey valid 2:1 actions
        for _, _, sh, sw, lv, a in nodes:
            assert a in RL.valid_actions(sh, sw)
            assert lv == RL.level_of(sh, sw)


def test_rect_basis_plane_exactness_and_ac_gate():
    torch.manual_seed(0)
    z = torch.randn(4, 32, 64, 64)
    basis = RL.fit_rect_basis(z)
    assert set(basis.keys()) == set(RL.COARSE_SHAPES)

    # Pure linear gradient image in latent space (models smooth sky + flat Mondrian boxes)
    yy = torch.linspace(-1.0, 1.0, 64)[None, None, :, None].expand(1, 32, 64, 64)
    xx = torch.linspace(-1.0, 1.0, 64)[None, None, None, :].expand(1, 32, 64, 64)
    plane_img = 1.5 + 0.7 * yy - 0.4 * xx
    pyr = RL.encode_rect_pyramid(plane_img, basis)

    for (sh, sw), entry in basis.items():
        rec = RL.lift_rect_tokens(pyr[(sh, sw)], sh, sw, entry)
        if entry["mode"] == "plane_pca32":
            assert (rec - plane_img).abs().max().item() < 1e-4
        # Plane-subtracted AC gate must be exact 0 on pure planes for ALL 12 coarse shapes!
        g = RL.tile_ac_gate_rect(pyr[(sh, sw)], sh, sw, entry)
        assert g.max().item() < 1e-4, f"non-zero gate on pure plane for {sh}x{sw}: {g.max().item()}"

    sm = _sample_kdtree_map(1, seed=3)
    canvas, mask, sf, pos, ac_gate = RL.build_rect_canvas(pyr, sm, basis)
    assert canvas.shape == (1, 32, 64, 64)
    assert mask.shape == (1, 1, 64, 64)
    assert sf.shape == (1, 4, 64, 64)
    assert pos.shape == (1, 2, 64, 64)
    assert ac_gate.shape == (1, 1, 64, 64)
    bx, by, b1 = RL.tile_border_masks(sm)
    assert bx.shape == (1, 1, 64, 63) and by.shape == (1, 1, 63, 64) and b1.shape == (1, 1, 64, 64)


def test_breadth_gen_rect_and_leaf_bias():
    torch.manual_seed(1)
    z = torch.randn(2, 32, 64, 64)
    basis = RL.fit_rect_basis(z)
    pyrs = [RL.encode_rect_pyramid(z[i:i + 1], basis) for i in range(2)]
    sm = _sample_kdtree_map(2, seed=5)
    stats = RM.ShapeStats().fit(pyrs)
    ex = [RM.make_rect_breadth_example(pyrs[i], sm[i], label=i, stats=stats) for i in range(2)]
    batch = RM.collate_rect_breadth(ex)

    bg = RM.BreadthGenRect(2, d=64, depth=2, heads=4, loops=2, head_w=64, head_depth=2, stats=stats)
    o = bg.loss(batch)
    assert torch.isfinite(o["total"])
    o["total"].backward()

    sm0, _ = bg.sample(0, mar_steps=2, diff_steps=4, leaf_bias=0.0, generator=torch.Generator().manual_seed(0))
    sm_hi, _ = bg.sample(0, mar_steps=2, diff_steps=4, leaf_bias=10.0, generator=torch.Generator().manual_seed(0))
    assert RL.validate_rect(sm0) and RL.validate_rect(sm_hi)
    # Huge positive leaf_bias must stop at the 4 root tiles (256x256 px)
    assert int(RL.tokens_of_rect(sm_hi)[0]) == 4


def test_looped_square_generators_roundtrip():
    torch.manual_seed(2)
    cfg = dict(d=64, depth=2, heads=4, loops=2, head_w=64, head_depth=2, tree=dict(d=32, depth=1, heads=4, loops=2))
    gm = GM.TesseraGen(2, cfg)
    bg = GM.BreadthGen(2, cfg)
    sm = LY.all_grid(1)[0]
    pyr = {2: torch.randn(32, 64, 64), 4: torch.randn(32, 32, 32), 8: torch.randn(32, 16, 16),
           16: torch.randn(32, 8, 8), 32: torch.randn(32, 4, 4)}
    gm.loss(GD.collate([GD.make_example(pyr, sm, 0)]))["total"].backward()
    bg.loss(GD.collate_breadth([GD.make_breadth_example(pyr, sm, 0)]))["total"].backward()
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "gen.pt"
        gm.save(p)
        gm2 = GM.TesseraGen.load(p)
        assert gm2.tokens.tf.loops == 2 and gm2.tree.tf.loops == 2


def test_rect_router_and_verify():
    torch.manual_seed(3)
    z = torch.randn(2, 32, 64, 64)
    basis = RL.fit_rect_basis(z)
    feat = RR.rect_router_features(z, basis)
    assert feat.shape == (2, RR.LatentRouterRect.IN_CH, 64, 64)
    router = RR.LatentRouterRect(width=32, dilations=(1, 2))
    sm_gt = _sample_kdtree_map(2, seed=11)
    loss = router.loss(feat, sm_gt)
    assert torch.isfinite(loss)
    loss.backward()
    sm_pred = router.route(feat)
    assert RL.validate_rect(sm_pred)
    sm_coarse = router.route(feat, leaf_bias=10.0)
    assert int(RL.tokens_of_rect(sm_coarse)[0]) == 4

    # Verify loop: fail once on top-left quadrant, then pass
    calls = [0]
    def check_fn(sm):
        calls[0] += 1
        if calls[0] == 1:
            bad = torch.zeros((64, 64), dtype=torch.bool)
            bad[:16, :16] = True
            return dict(passed=False, bad8=bad, score=3.0)
        return dict(passed=True, score=1.0)
    res = RR.verify_rect(sm_coarse[:1], check_fn)
    assert res["passed"] and not res["fallback"] and RL.validate_rect(res["shape_map"])
    assert res["tokens"] == 5                                                 # 1 of 4 roots split in half -> 5 tiles


def test_rect_tiledown_locality_and_zero_init():
    torch.manual_seed(4)
    z = torch.randn(2, 32, 64, 64)
    basis = RL.fit_rect_basis(z)
    td = RTD.RectTileDown(c=32, hid=64)

    pyr_lin = RL.encode_rect_pyramid(z, basis)
    pyr_td0 = RTD.encode_rect_pyramid_td(z, basis, tiledown=td)
    for sh_sw in RL.SHAPES:
        assert (pyr_td0[sh_sw] - pyr_lin[sh_sw]).abs().max().item() < 1e-6

    # Randomize zero-initialized output head and verify strict tile locality on an (8, 16) tile
    with torch.no_grad():
        td.out.weight.normal_(0.0, 0.05)
        td.out.bias.normal_(0.0, 0.05)
    td.eval()
    tok_a = td.encode_shape(z, (8, 16), basis)
    z_pert = z.clone()
    # Keep tile [gy=1, gx=2] -> rows [8:16], cols [32:48] unchanged; corrupt everything outside it
    keep = z_pert[:, :, 8:16, 32:48].clone()
    z_pert = z_pert + 5.0
    z_pert[:, :, 8:16, 32:48] = keep
    tok_b = td.encode_shape(z_pert, (8, 16), basis)
    assert (tok_a[:, 1, 2] - tok_b[:, 1, 2]).abs().max().item() < 1e-6
    assert (tok_a[:, 0, 0] - tok_b[:, 0, 0]).abs().max().item() > 1e-3


def test_build_active_rect_canvas_matches_dense_and_grads():
    torch.manual_seed(5)
    z = torch.randn(2, 32, 64, 64)
    basis = RL.fit_rect_basis(z)
    basis_dev = RTD.basis_to_device(basis, z.device, dtype=z.dtype)
    sm = _sample_kdtree_map(2, seed=19)
    td = RTD.RectTileDown(c=32, hid=64)
    with torch.no_grad():
        td.out.weight.normal_(0.0, 0.05)
        td.out.bias.normal_(0.0, 0.05)

    for bopen in (False, True):
        pyr_dense = RTD.encode_rect_pyramid_td(z, basis, tiledown=td)
        dense_out = RL.build_rect_canvas(pyr_dense, sm, basis, border_open=bopen)
        act_out = RTD.build_active_rect_canvas(z, sm, basis_dev, tiledown=td, border_open=bopen)
        for ta, tb in zip(act_out, dense_out):
            if ta.dtype == torch.bool:
                assert torch.equal(ta, tb)
            else:
                assert (ta - tb).abs().max().item() < 1e-5

    td.zero_grad()
    canvas, mask, _, _, ac_gate = RTD.build_active_rect_canvas(z, sm, basis_dev, tiledown=td)
    loss = ((canvas * mask) ** 2).mean() + (ac_gate * mask).mean()
    loss.backward()
    assert td.out.weight.grad is not None and torch.isfinite(td.out.weight.grad).all()
    assert float(td.out.weight.grad.abs().sum()) > 0.0
    assert td.conv1.weight.grad is not None and torch.isfinite(td.conv1.weight.grad).all()
    assert float(td.conv1.weight.grad.abs().sum()) > 0.0


def test_rect_router_integral_pooling_and_tiledown_feats():
    torch.manual_seed(6)
    z = torch.randn(2, 32, 64, 64)
    basis = RL.fit_rect_basis(z)
    td = RTD.RectTileDown(c=32, hid=64)
    f_lin = RR.rect_router_features(z, basis)
    f_td0 = RR.rect_router_features(z, basis, tiledown=td)
    assert (f_td0 - f_lin).abs().max().item() < 1e-6

    with torch.no_grad():
        td.out.weight.normal_(0.0, 0.05)
        td.out.bias.normal_(0.0, 0.05)
    f_td1 = RR.rect_router_features(z, basis, tiledown=td)
    assert (f_td1[:, 32:] - f_lin[:, 32:]).abs().max().item() > 1e-4

    router = RR.LatentRouterRect(width=32, dilations=(1, 2))
    h_1 = torch.randn(32, 64, 64, requires_grad=True)
    nodes = [(0, 0, 32, 32), (0, 32, 16, 32), (32, 16, 16, 8), (60, 60, 4, 4)]
    logits_vec = router.node_logits(h_1, nodes)

    # Brute-force reference matching original slice-mean implementation
    pooled_ref, s4_ref = [], []
    for r0, c0, sh, sw in nodes:
        pooled_ref.append(h_1[:, r0:r0 + sh, c0:c0 + sw].mean(dim=(1, 2)))
        lsh, lsw = torch.log2(torch.tensor(sh / 2.0)), torch.log2(torch.tensor(sw / 2.0))
        s4_ref.append((lsh, lsw, lsh + lsw, lsh - lsw))
    logits_ref = router.head(torch.stack(pooled_ref, 0) + router.shape_mlp(torch.tensor(s4_ref, dtype=h_1.dtype)))
    assert (logits_vec - logits_ref).abs().max().item() < 1e-5
    logits_vec.sum().backward()
    assert h_1.grad is not None and torch.isfinite(h_1.grad).all()
    assert float(h_1.grad.abs().sum()) > 0.0


def test_verify_rect_split_choice_fn():
    sm_coarse = RL.all_grid_rect(1) * 0 + 32
    calls = [0]

    def check_fn(sm):
        calls[0] += 1
        if calls[0] == 1:
            bad = torch.zeros((64, 64), dtype=torch.bool)
            bad[:16, :16] = True
            return dict(passed=False, bad8=bad, score=3.0)
        return dict(passed=True, score=1.0)

    res = RR.verify_rect(sm_coarse, check_fn, split_choice_fn=lambda r0, c0, sh, sw: RL.ACT_SPLIT_V)
    assert res["passed"] and RL.validate_rect(res["shape_map"])
    assert int(res["shape_map"][0, 0, 0, 0]) == 32 and int(res["shape_map"][0, 1, 0, 0]) == 16
