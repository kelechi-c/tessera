import torch

from test_latent_layout import rand_pyramid, rand_size_map
from tessera.gen import data as GD
from tessera.gen import model as GM
from tessera.latent import layout as LY


def _examples(n=3, stats=None, grid=False):
    out = []
    for i in range(n):
        sm = LY.all_grid(1)[0] if grid else rand_size_map(i, p_split=.35 + .15 * i)
        out.append(GD.make_example(rand_pyramid(i), sm, label=i % 2, stats=stats))
    return out


def test_level_stats_normalise_per_level_and_roundtrip():
    pyrs = [rand_pyramid(i) for i in range(3)]
    for p in pyrs:
        p[32] = p[32] * 5 + 3                      # a level with different statistics
    st = GD.LevelStats().fit(pyrs)
    assert abs(float(st.mean[32].mean()) - 3) < .5 and abs(float(st.std[32].mean()) - 5) < 1
    sm = rand_size_map(2)
    order = LY.leaves(sm)
    tok = LY.pack_tokens(pyrs[0], sm, order)
    sizes = [s for _, _, s in order]
    n = st.norm(tok, sizes)
    assert torch.allclose(st.denorm(n, sizes), tok, atol=1e-5)
    st2 = GD.LevelStats.from_state_dict(st.state_dict())
    assert torch.equal(st2.norm(tok, sizes), n)


def test_examples_and_collate_have_variable_lengths():
    ex = _examples(3)
    lens = [len(e["tokens"]) for e in ex]
    assert len(set(lens)) > 1
    for e in ex:
        assert len(e["bits"]) == len(e["node_pos"]) and len(e["tokens"]) == len(e["leaf_pos"])
        assert torch.equal(LY.from_tree_bits(e["bits"].tolist()), e["size_map"])
    b = GD.collate(ex)
    assert b["tokens"].shape == (3, max(lens), LY.TOK) and b["leaf_mask"].sum(1).tolist() == lens
    assert b["bits"][~b["node_mask"]].eq(-1).all()


def test_to_pyramid_inverts_make_example():
    st = GD.LevelStats().fit([rand_pyramid(i) for i in range(2)])
    pyr = rand_pyramid(5)
    sm = rand_size_map(5)
    e = GD.make_example(pyr, sm, stats=st)
    back = GD.to_pyramid(e["tokens"], sm, st)
    assert torch.allclose(LY.pack_tokens(back, sm), LY.pack_tokens(pyr, sm), atol=1e-5)


def test_losses_are_finite_and_padding_is_ignored():
    torch.manual_seed(0)
    m = GM.TesseraGen(2, "tiny")
    b = GD.collate(_examples(3))
    out = m.loss(b)
    assert all(torch.isfinite(v) for v in out.values())
    out["total"].backward()
    assert m.tokens.inp.weight.grad is not None and m.tree.out.weight.grad is not None
    # garbage in padded slots must not change the tree loss (the token loss masks padding the same way)
    b2 = {k: v.clone() for k, v in b.items()}
    b2["tokens"][~b2["leaf_mask"]] = 1e6
    b2["bits"][~b2["node_mask"]] = 1
    with torch.no_grad():
        l1 = float(m.tree.loss(b))
        l2 = float(m.tree.loss(b2))
        assert abs(l1 - l2) < 1e-5
        losses = []
        for bb in (b, b2):
            torch.manual_seed(3)
            losses.append(float(m.tokens.loss(bb, generator=torch.Generator().manual_seed(3))))
        assert abs(losses[0] - losses[1]) < 1e-4, losses


def test_tiny_overfit_tree_and_tokens():
    """Two fixed sequences: both losses fall well below where they start."""
    torch.manual_seed(0)
    m = GM.TesseraGen(2, dict(d=64, depth=2, heads=4, head_w=96, head_depth=2))
    b = GD.collate(_examples(2))
    opt = torch.optim.AdamW(m.parameters(), 2e-3)
    first = None
    for i in range(300):
        opt.zero_grad()
        out = m.loss(b)
        out["total"].backward()
        opt.step()
        if i == 0:
            first = {k: v.item() for k, v in out.items()}
    last = {k: v.item() for k, v in m.loss(b).items()}
    assert last["tree"] < .2 * first["tree"]
    assert last["tokens"] < .7 * first["tokens"]


def test_sampling_gives_valid_trees_and_matching_token_counts():
    torch.manual_seed(0)
    m = GM.TesseraGen(2, "tiny").eval()
    g = torch.Generator().manual_seed(0)
    for label in (0, 1):
        sm, tok = m.sample(label, steps=4, diff_steps=4, generator=g)
        assert LY.validate(sm[None])
        assert tok.shape == (len(LY.leaves(sm)), LY.TOK) and torch.isfinite(tok).all()
        pyr = GD.to_pyramid(tok, sm)
        assert set(pyr) == set(LY.SIZES)


def test_tree_sampler_follows_a_trained_tree():
    """Overfit the tree to one layout per class; greedy sampling must reproduce it exactly (valid by construction)."""
    torch.manual_seed(0)
    tree = GM.TreeAR(2, d=64, depth=2, heads=4)
    ex = [GD.make_example(rand_pyramid(0), rand_size_map(s, p_split=.5), label=s) for s in (0, 1)]
    b = GD.collate(ex)
    opt = torch.optim.AdamW(tree.parameters(), 3e-3)
    for _ in range(300):
        opt.zero_grad()
        tree.loss(b).backward()
        opt.step()
    tree.eval()
    for e in ex:
        sm = tree.sample(e["label"], temperature=1e-4)
        assert torch.equal(sm, e["size_map"])


def test_grid_baseline_runs_through_the_same_code():
    grid = LY.all_grid(1)[0]
    m = GM.TesseraGen(2, "tiny", fixed_layout=grid)
    assert m.tree is None
    b = GD.collate(_examples(2, grid=True))
    assert b["tokens"].shape[1] == LY.grid_tokens() and bool(b["leaf_mask"].all())
    out = m.loss(b)
    assert set(out) == {"tokens", "total"} and torch.isfinite(out["total"])
    sm, tok = m.eval().sample(0, steps=2, diff_steps=2, generator=torch.Generator().manual_seed(0))
    assert torch.equal(sm, grid) and tok.shape == (1024, LY.TOK)


def test_size_order_reveals_large_tiles_first():
    m = GM.TokenMAR(1, d=32, depth=1, heads=4, head_w=32, head_depth=1).eval()
    sm = rand_size_map(3, p_split=.5)
    pos = GD.pos_of(LY.leaves(sm))
    seen = []
    orig = m.head.sample

    def spy(z, *a, **k):
        seen.append(len(z))
        return orig(z, *a, **k)

    m.head.sample = spy
    tok = m.sample(pos, 0, steps=3, diff_steps=2, order="size", generator=torch.Generator().manual_seed(0))
    assert sum(seen) == len(pos) and tok.shape == (len(pos), LY.TOK)     # every leaf revealed exactly once
    seq = GM.reveal_order(pos, "size", torch.Generator().manual_seed(1))
    sizes = pos[seq, 2]
    assert sorted(seq.tolist()) == list(range(len(pos)))
    assert bool((sizes[:-1] >= sizes[1:]).all())                         # never a smaller tile before a larger one
    assert len(set(sizes.tolist())) > 1


def test_base_config_size_and_matched_baseline():
    """The costed comparison: about 150M transformer parameters in the token model; the tessera model adds only a
    small tree model over the grid baseline (built on the meta device, no memory)."""
    with torch.device("meta"):
        tes = GM.TesseraGen(1000, "base")
        grid = GM.TesseraGen(1000, "base", fixed_layout=LY.all_grid(1)[0])
    n_tf = GM.nparams(tes.tokens.tf)
    assert 120e6 < n_tf < 180e6, n_tf
    assert GM.nparams(tes.tokens) == GM.nparams(grid.tokens)
    assert GM.nparams(tes.tree) < .05 * GM.nparams(tes.tokens)


def test_head_overfits_fixed_tokens_and_samples_land_on_them():
    """The diffusion head alone, conditioned on two fixed vectors, learns two fixed tokens; DDIM samples must land
    near them (catches a broken noise schedule, which an isfinite check does not)."""
    torch.manual_seed(0)
    head = GM.DiffHead(16, w=128, depth=3)
    z = torch.randn(2, 16)
    x0 = torch.randn(2, LY.TOK).clamp(-2, 2)
    opt = torch.optim.AdamW(head.parameters(), 2e-3)
    for _ in range(1500):
        opt.zero_grad()
        head.loss(x0, z, repeats=32).backward()
        opt.step()
    head.eval()
    s = head.sample(z.repeat(4, 1), steps=50, generator=torch.Generator().manual_seed(0))
    err = (s - x0.repeat(4, 1)).abs().mean()
    assert err < .25, float(err)


def test_untrained_samples_stay_in_range():
    torch.manual_seed(0)
    m = GM.TesseraGen(2, "tiny").eval()
    _, tok = m.sample(0, steps=4, diff_steps=8, generator=torch.Generator().manual_seed(0))
    assert float(tok.abs().max()) <= GM.X0_CLIP + 1e-3 + 4                 # clip on x0, plus at most the final noise


def test_generator_checkpoint_keeps_stats_and_layout(tmp_path):
    st = GD.LevelStats().fit([rand_pyramid(i) for i in range(2)])
    m = GM.TesseraGen(3, "tiny", fixed_layout=LY.all_grid(1)[0], stats=st)
    m.save(tmp_path / "g.pt")
    m2 = GM.TesseraGen.load(tmp_path / "g.pt")
    assert m2.tree is None and torch.equal(m2.fixed_layout, m.fixed_layout)
    assert torch.equal(m2.stats.mean[32], st.mean[32])
    for k, v in m.state_dict().items():
        assert torch.equal(v, m2.state_dict()[k]), k


def test_label_dropout_uses_the_unconditional_index():
    lab = torch.zeros(1000, dtype=torch.long)
    d = GM.drop_labels(lab, 7, p=.3, generator=torch.Generator().manual_seed(0))
    share = float((d == 7).float().mean())
    assert .25 < share < .35 and set(d.tolist()) <= {0, 7}


def test_breadth_examples_and_level_causal_isolation():
    """BreadthGen examples match quadtree nodes, and level-causal attention leaks zero future-level info."""
    torch.manual_seed(0)
    ex = []
    for i in range(2):
        sm = rand_size_map(i + 1, p_split=.45 + .15 * i)
        ex.append(GD.make_breadth_example(rand_pyramid(i), sm, label=i))
    b = GD.collate_breadth(ex)
    m = GM.BreadthGen(2, "tiny").eval()
    # Perturb finer-level tokens (level >= 2) and check level 0/1 contextual representations are bit-identical
    b2 = {k: v.clone() for k, v in b.items()}
    fine = b2["node_level"] >= 2
    b2["node_tok"][fine] = 1e4
    with torch.no_grad():
        h1 = m.encode(b["node_tok"], b["node_pos"], b["node_level"], b["node_mask"], b["node_mask"], b["label"])
        h2 = m.encode(b2["node_tok"], b2["node_pos"], b2["node_level"], b2["node_mask"], b2["node_mask"], b2["label"])
    coarse = b["node_mask"] & (b["node_level"] < 2)
    assert torch.allclose(h1[coarse], h2[coarse], atol=1e-5)


def test_breadth_overfit_sample_progressive_and_ckpt(tmp_path):
    """BreadthGen overfits 2 trees/tokens, reproduces layouts, returns progressive snapshots, and roundtrips ckpt."""
    torch.manual_seed(0)
    pyrs = [rand_pyramid(i) for i in range(2)]
    st = GD.LevelStats().fit(pyrs)
    sms = [rand_size_map(10, p_split=.35), rand_size_map(11, p_split=.55)]
    ex = [GD.make_breadth_example(pyrs[i], sms[i], label=i, stats=st) for i in range(2)]
    b = GD.collate_breadth(ex)
    m = GM.BreadthGen(2, dict(d=64, depth=2, heads=4, head_w=96, head_depth=2), stats=st)
    opt = torch.optim.AdamW(m.parameters(), 3e-3)
    first = None
    for i in range(250):
        opt.zero_grad()
        out = m.loss(b, p_uncond=0.0, all_levels=True)
        out["total"].backward()
        opt.step()
        if i == 0:
            first = {k: v.item() for k, v in out.items()}
    last = {k: v.item() for k, v in m.loss(b, p_uncond=0.0, all_levels=True).items()}
    assert last["split"] < .25 * first["split"]
    assert last["tokens"] < .7 * first["tokens"]

    m.eval()
    sm, tok, snaps = m.sample(0, steps_per_level=2, diff_steps=6, tree_temp=1e-4, return_pyramid=True,
                              generator=torch.Generator().manual_seed(0))
    assert LY.validate(sm[None]) and torch.equal(sm, sms[0])
    assert tok.shape == (len(LY.leaves(sm)), LY.TOK) and torch.isfinite(tok).all()
    assert len(snaps) >= 1 and snaps[0]["size_px"] == 256

    # Checkpoint roundtrip + CFG + split_bias smoke
    m.save(tmp_path / "bg.pt")
    m2 = GM.BreadthGen.load(tmp_path / "bg.pt")
    sm_c, tok_c = m2.sample(1, steps_per_level=2, diff_steps=4, cfg_scale=1.5, split_bias=-0.5,
                            generator=torch.Generator().manual_seed(1))
    assert LY.validate(sm_c[None]) and tok_c.shape == (len(LY.leaves(sm_c)), LY.TOK)
