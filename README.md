# neural-contraints-learn

Can a language model learn to control a brain-computer interface driven by its own activations? This repo is an attempt to reproduce, in an LLM, the monkey experiment of Sadtler et al. 2014, *Neural constraints on learning* (Nature 512:423, [doi:10.1038/nature13665](https://doi.org/10.1038/nature13665)). The typo in the repo name ("contraints") dates from the day the folder was created and was kept.

**Status (2026-11-06): the experiment is paused.** The pipeline works end to end and is well tested. The main finding, though, is that the model never learns to adapt to a changed decoder, so the question the experiment was built for (is within-manifold learning easier than outside-manifold learning?) cannot be answered with this design. Details in [What we found](#what-we-found).

---

## The idea in one page

**The monkey experiment.**
- Monkeys moved a cursor on a screen using only the activity of about 90 neurons in motor cortex. A decoder turned the activity into cursor velocity.
- The activity naturally lives in a low-dimensional subspace, about 10 dimensions, called the **intrinsic manifold**.
- Sadtler et al. silently changed the decoder in one of two ways:
  - **within-manifold**: the new decoder still reads the monkey's usual activity patterns, just mapped to different directions;
  - **outside-manifold**: moving the cursor now requires activity patterns the monkey never produces.
- Monkeys re-learned within-manifold changes in a few hours, but not outside-manifold ones.

**Our LLM version.** The model plays a navigation game in text:

```
User:  Trial 1. Marker (0, 0). Target (8, 0). Moves left in this trial: 36.
Model: r r u r
User:  Trial 1. Marker (3, 1). Target (8, 0). Moves left in this trial: 32.
```

The model replies with moves (`r l u d`), but the letters do **not** move the marker directly. Instead:
1. We read the model's residual-stream activation at the token that produced each letter.
2. We take 100 randomly chosen coordinates of that activation.
3. A 10-dimensional factor analysis gives the "manifold".
4. A linear decoder turns the result into a 2-D step for the marker.

The decoder is first fit so that writing `r` really does move the marker right (the "intuitive" decoder), so at the start the game feels normal. Then, silently, we rotate the decoder. Two numbers describe each rotation:

- **Rotation θ**: how far the decoder is turned away from the intuitive one.
- **Alignment a**: how much of the new decoder still lies inside the manifold. a = 1 is fully inside (within-manifold), a = 0.2 mostly outside.
- They are tied by **a ≥ cos²θ**. A rotation by θ keeps a share cos²θ of the original direction, which is guaranteed to be inside, so low alignment only exists at large θ. That is why the experimental grid is triangular.

A session is 8 baseline trials (intuitive decoder), then 80 perturbation trials (rotated decoder), then 8 washout trials (intuitive decoder again). The question is whether success recovers during the perturbation block, and whether it recovers better at high alignment. The model is never told anything about activations or decoders.

The full design, every deviation from the paper, and a detailed run log are in [`docs/DESIGN.md`](docs/DESIGN.md) (long; §7.2c to §7.2h and §7.3 are the run history).

---

## What we ran

All runs use **Qwen3.5-9B** (32 layers) unless noted, with greedy decoding and no reasoning.

| Run | Date (2026) | What | Status |
|---|---|---|---|
| `g3`, `g4`, `sham` | Aug 10 | First pilots, layer 21 | Invalid (two bugs, see below) |
| `v2`, `v3` | Aug 11 to 12 | Re-runs after the label fix | Invalid (read off-by-one) |
| `v4` | Aug 12 to 13 | Read fixed, layer 28 | Superseded |
| `v5` | Aug 13 to 14 | Full grid: 23 (θ, alignment) cells + no-perturbation control, 6 seeds, 80 perturbation trials | Decoder bug, superseded |
| `v6q` | Aug 14 to 17 | Same grid, second coordinate draw | Decoder bug, superseded |
| `v6g_d9`, `v6g_d17` | Aug 14 to 17 | Same grid on **gemma-4-31B-it** (60 layers, read at 53), two draws | Decoder bug, superseded |
| `v7q_d0`, `v7q_d13` | Aug 18 to 19 | Same grid, corrected decoder, two new draws | **The valid run** |

The bugs, in the order they were found:
1. **Label (Aug 11).** The decoder was fit to the letter the model was forced to write next during calibration. That letter was almost fully predictable from the prompt alone, so no decoder beat a prompt-only predictor. It is now fit to the model's own next-token distribution over `r/l/u/d` (`nc/labels.py`).
2. **Read off-by-one (Aug 12).** For a k-letter reply only k-1 moves were applied, always dropping the model's first move.
3. **Label fix never reached the session runner (found Aug 18).** `run.py` and `screen.py` still used the old label, so all of v5 and v6 (570 sessions) ran the wrong decoder. `tests/test_live_label.py` now guards this. v7 is the re-run.

Gemma was never re-run on the corrected decoder.

---

## What we found

**1. A 2-D readout of the next move exists, but only near the output** ([`figures/1_readout.png`](figures/1_readout.png)).
- From about 70% depth, a linear 2-D readout predicts the direction of the model's next move with held-out R² 0.80 to 0.90.
- But a predictor that sees only the prompt already reaches 0.80, so the activations add about 0.10 on top.
- At middle depths the readout is worse than the prompt.
- At 88% depth (layer 28/32), a decoder fit to the next-token distribution partly re-derives the output layer by construction. In practice the "neural" readout mostly reads which letter the model is about to write.

**2. How hard a perturbation is depends on the "effective rotation", not on the nominal one.**
- Effective rotation is the angle between where the marker actually moves and what the letter means. It is about 15° even with the intuitive decoder.
- It predicts impairment strongly: rank correlation −0.70 to −0.79 in every sweep.
- Matching perturbations on the screen's severity measure did not match difficulty (rank correlation with success −0.01).
- Nominal θ is not even monotone in effective rotation.

**3. At equal effective rotation, higher alignment does a bit better.** This is the direction the manifold hypothesis predicts, but it is weak with the corrected decoder. The table comes from `results/summaries/v6_replication.json`:

| Sweep | Decoder | Sessions | Alignment residual | Matched bands positive |
|---|---|---|---|---|
| Qwen3.5-9B draw 16 (v5) | old | 128 | +0.49 | 5/5 |
| Qwen3.5-9B draw 14 (v6q) | old | 124 | +0.01 | 4/5 |
| gemma-4-31B draw 17 | old | 98 | +0.38 | 3/3 |
| gemma-4-31B draw 9 | old | 94 | +0.34 | 4/4 |
| **Qwen3.5-9B draw 0 (v7)** | **corrected** | 112 | **+0.12** | 3/5 |
| **Qwen3.5-9B draw 13 (v7)** | **corrected** | 102 | **+0.36** | 2/4 |

How to read the table:
- **Alignment residual:** how much alignment still predicts success once effective rotation is accounted for.
- **Matched bands:** groups of sessions with similar effective rotation, and in how many of them higher alignment did better.
- **Draws:** different random sets of 100 coordinates. The effect varies about as much between draws as between models.

**4. The model does not learn. This is the headline.** Success in the first 20 vs the last 20 perturbation trials, v7, complete sessions only:

| Alignment | Draw 0: first 20 → last 20 | Draw 13: first 20 → last 20 |
|---|---|---|
| no perturbation | 1.00 → 1.00 | 1.00 → 0.98 |
| 1.0 (inside) | 0.37 → 0.29 | 0.40 → 0.30 |
| 0.7 | 0.39 → 0.33 | 0.38 → 0.38 |
| 0.4 | 0.43 → 0.38 | 0.32 → 0.25 |
| 0.2 (mostly outside) | 0.28 → 0.25 | 0.37 → 0.32 |

- Performance gets *worse* over the perturbation block, at every alignment, including fully inside the manifold. Sessions that got worse outnumber those that improved: 50 vs 24 in draw 0, 59 vs 23 in draw 13.
- Washout returns to about 1.00 immediately, so there is no after-effect.
- In Sadtler's terms there is no learning at all, so there is no within vs outside difference to measure. Finding 3 is therefore only about how much each perturbation hurts at the start, not about re-learning.

To reproduce this table from `results/trials.csv.gz`:

```python
import pandas as pd
d = pd.read_csv("results/trials.csv.gz")
d = d[d.run.str.startswith("v7")]
p = d[d.block == "perturbation"]
full = p.groupby(["run", "shard", "session"]).trial.transform("count") >= 80
p = p[full].sort_values("trial")
g = p.groupby(["run", "shard", "session"])
s = pd.DataFrame({"alignment": g.alignment.first().round(1),
                  "first20": g.acquired.apply(lambda x: x.iloc[:20].mean()),
                  "last20": g.acquired.apply(lambda x: x.iloc[-20:].mean())})
print(s.groupby([s.index.get_level_values("run"), "alignment"], dropna=False)[["first20", "last20"]].mean())
```

**Other measured facts worth knowing:**
- **Sequence length.** The model does not shorten its replies under perturbation to get tighter feedback.
- **Context growth.** Drift over a session is mild with no perturbation: success 0.94 → 0.88 over 56 trials.
- **Speed confound.** With the angle matched, low-alignment decoders also move the marker more slowly (up to about 50%), which hurts performance on its own.
- **Ratio metric.** Sadtler's learning metric divides by the initial impairment, so it explodes when there is no impairment (the no-perturbation control scored higher than every perturbed arm). Do not use it here.

---

## Why we think it failed, and open questions

- **Why no learning.** The target coordinates are in the prompt, so the model can play the game from arithmetic alone and has no reason to use the feedback. When the feedback contradicts its plan, it does not explore; it gets stuck or repeats itself. The stuck and repetition detectors fire often.
- **The readout is too close to the behaviour.** Because the decoder reads the next-letter intent, a within-manifold rotation is roughly a noisy remapping of the letters ("`r` now goes up-left").
- **A cheap diagnostic was never run.** Rotate the letters themselves, with no activations involved, and check whether the model learns that. It is the best case for the within-manifold arm: if the model cannot learn a plain letter remap, the activation version cannot work either.
- **Gemma** was never re-run on the corrected decoder.

---

## Repository layout

```
nc/                 the library
  task.py           board, alphabet, prompts, reply parsing, calibration paths
  manifold.py       factor analysis, decoder, perturbation construction, screens
  labels.py         the intent label (model's own next-letter distribution), prompt-only baselines
  metrics.py        binning, z-scoring, learning metrics
  session.py        closed-loop lockstep session runner, checkpointing, stuck/repetition detectors
tests/              105 unit tests, no GPU needed
scripts/
  calibrate.py      stage 1 (GPU): teacher-forced reads, caches activations and labels
  fit_draws.py      stage 2 (CPU): per-draw manifold + decoder + diagnostics
  screen.py         stage 3 (CPU): choose layer and perturbations (needs --fit-report)
  run.py            closed-loop sessions (GPU)
  analyze.py, analyze_grid.py, compare_draws.py    analysis
  figures.py, figures_runs.py, plot_trajectories.py   figures
  build_results.py  rebuilds results/ from outputs/ and artifacts/
  *_campaign.sh, *_sbatch.sh   SLURM campaigns used on Mila (self-chaining short-unkillable jobs)
docs/DESIGN.md      design, deviations from the paper, full run log
docs/READ_PATH_DECISION.md   pre-registered gate for the prefix-cached activation read
figures/            all figures; figures/runs/ per-run, figures/trajectories/ example sessions
results/            small copy of the results (see below)
```

**`results/`** is what fits in git:
- `summaries/`: every JSON summary (fits, screens, label checks, `v5_grid.json`, `v6_replication.json`).
- `sessions/<run>.tar.gz`: `meta.json` plus per-session `sess_*_trials.json` for each run.
- `trials.csv.gz`: one row per trial across all runs. Columns are run, shard, session, draw, alignment, θ, severity, magnitude, block, trial, acquired, moves used, mean angular error in radians, number of moves.
- `logs/`: campaign status files and job stdout.

Not in git, because of size, but kept on Mila scratch under `/network/scratch/b/baldelld/MATS/neural-contraints-learn/`: cached activations (`artifacts/*.npz`, 2.7 GB), per-turn session transcripts (`outputs/**/sess_*.jsonl`, 370 MB) and per-GPU logs. Ask Davide if you need them.

In the trajectory figures, `sequence_a0.7_s001.png` means alignment 0.7, session 1; `sequence_control_…` is a session with no perturbation. Note that alignment 1.0 is still a perturbation: the decoder is rotated, just within the manifold.

---

## Running it

- **Environment:** Python 3.12 with vLLM (the original env was `MATS/venv-vllm`, uv-managed), `numpy`, `scipy`, `scikit-learn`, `torch`, `transformers`, `pytest`.
- **Hardware:** one A100-80GB is enough for Qwen3.5-9B.
- **Hard-coded paths:** many scripts contain absolute paths to Davide's Mila scratch (`/network/scratch/b/baldelld/MATS/...`); adjust them.

```bash
python -m pytest tests/ -q                                   # no GPU

python scripts/calibrate.py --model Qwen/Qwen3.5-9B --n-fit 1200 --n-val 600 --n-depths 8
python scripts/fit_draws.py --reads artifacts/reads_Qwen3.5-9B.npz --draws 20
python scripts/screen.py    --reads artifacts/reads_Qwen3.5-9B.npz --fit-report artifacts/draws_Qwen3.5-9B.json
python scripts/run.py       --reads ... --screen ... --mode baseline --sessions 4 --trials 20
python scripts/analyze.py   --dir outputs/g3 --mode baseline
```

`--fit-report` is required in practice. Without it the screen ranks layers by how many perturbations they admit instead of by decoder quality, and once picked a layer whose decoder had R² −0.18.

**Costs on an A100.**
- Calibration reads: about 20 GPU-minutes for Qwen3.5-9B.
- CPU stages: about 35 minutes on 4 cores.
- One 96-trial grid session: minutes for Qwen, but the cost grows with the square of the number of turns, because every turn re-reads the whole conversation.
- A full 144-session grid: one to two 4-GPU nights.

## Rules this project learned the hard way

- Always score a readout against a prompt-only baseline. A readout that cannot beat the prompt has nothing to perturb.
- Never relax a gate to make a stage pass; the campaign scripts stop on a failed gate.
- One activation per emitted letter, the one that generated it (§7.2d in DESIGN).
- Do not use ratio metrics whose denominator is the effect.
- Every number in `nc/task.py`'s `BoardConfig` was provisional.
