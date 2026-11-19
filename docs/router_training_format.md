# Router training data

The router (`tessera.router.Router`) predicts a layout in one pass.  It is to be trained by distillation from the search:
one record per image, written by `tessera encode ... --record layouts.jsonl` (one JSON object per line,
`tessera.router.LayoutRecord`):

| field | meaning |
|---|---|
| `format` | `tessera/layout-record/1` |
| `name`, `image_md5` | source file name; md5 of the 512 px uint8 RGB the search saw |
| `arm` | `A` (fixed codes) or `vae16` (FLUX 16 px leaves) |
| `lvl` | 4096 ints (row-major 64 x 64, 8 px cells): leaf level covering each cell, 0 = whole image ... 5 = 16 px, 6 = native 8 px cell |
| `two` | bit-packed 64 x 64 bool (`shape`, `hex`): the covering coded leaf has 2 tokens |
| `sel` | bit-packed 32 x 32 bool or null: which 16 px leaves are FLUX leaves (vae16 arm only) |
| `tokens` | token count of the stream (header token included) |
| `verdict` | judge v5 on the decode of the stored tokens: `score`, `passed`, `argmax`, `T`, `mae32` |

Target for training: `LayoutRecord.rungs()` = `core.route.rungs_of(lvl, two, vae16=sel is not None)`, i.e. per 8 px cell the rung
`2*level + two` (native = 12; with the FLUX ladder 16 px is terminal, rung 10).  Input features per image: `Ctx.features()`,
16 channels at 64 x 64 (per-rung cell excess (12), reference score, log fine-scale std, grain mask, mean colour).  Train on
records with `verdict.passed == true`; the loss is ordinal (12 sigmoid logits, P(rung > k)); at inference `project` makes the
rungs a valid quadtree and `verify` refines failing leaves with the same judge.  No training code ships in this package.
