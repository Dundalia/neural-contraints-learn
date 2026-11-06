# Neural constraints on in-context learning

An LLM transplant of Sadtler et al. 2014, *Neural constraints on learning* (Nature 512:423–426), with in-context learning in the role of fast (hours-timescale) motor learning.

**Thesis.** In-context learning re-indexes existing representational degrees of freedom; it does not create new ones.

**Prediction.** A model can learn in context to re-control a readout of its own activations when the readout's control space stays inside the low-dimensional subspace its activity naturally occupies, and cannot when the control space leaves it — with the size of the perturbation held equal.

Status: design complete, G0 complete. **Revision 6.** One G1/G2 pass was run and its results were discarded as unusable — the decoder had been fit to read the *current* letter rather than the next one, so every reported number described the wrong readout (Appendix A). Nothing measured currently stands except three label-independent findings, flagged inline. Internal project names are glossed in Appendix B.

**Gates**, referenced throughout and specified in §7: **G0** offline unit tests (done) · **G1** calibration reads and decodability · **G2** perturbation screening, CPU only, **the kill gate** · **G3** closed-loop baseline, which freezes the calibrated geometry · **G4** matched impairment and nonzero within-manifold learning. Full n only after G4.

**Revision 6 changes, all decided 2026-08-10.** The read label is now state-*j* → letter-*j+1*, the intent reading (§4.2). Target angle is sampled at random (§3.1). The ambient space is 100 randomly drawn coordinates, resampled per session (§4.1). θ is screened jointly with the read layer rather than fixed (§4.6). The alignment grid extends to 0.1 (§4.6).

---

## 1. The source experiment

### 1.1 Recording and readout

Two Rhesus macaques, 96-channel arrays in proximal-arm M1. A "neural unit" = threshold crossings on one electrode (threshold 3.0 × RMS, measured daily in a darkened room); 85–91 units per day. Spike counts in non-overlapping **45 ms** bins, **z-scored per unit** → `u ∈ R^q`.

No limb controller anywhere: arms "loosely restrained," hand movements during BCI trials minimal. (Extended Data Fig 5b's "hand control point-to-point" is a *separate* reaching task, shown only to establish that BCI-trial hand speeds were far smaller.)

### 1.2 The intrinsic manifold

Factor analysis with **d = 10** factors on the calibration-block activity:

```
z ~ N(0, I_d)                z ∈ R^10
u | z ~ N(Λz + μ, Ψ)         Ψ diagonal
```

Fit by EM; `μ = 0` because spike counts were z-scored first. Factor analysis rather than PCA because it *splits* the covariance into a part shared across units (`Λ`) and a part private to each (`Ψ`). Shared co-modulation is what circuitry imposes; PCA would rank a large but purely private direction as important.

The paper defines the intrinsic manifold as `colspace(Λ)`. Factors are read out as the posterior mean `ẑ = β u`, `β = Λᵀ(ΛΛᵀ + Ψ)⁻¹`, then each factor is z-scored separately (diagonal `S_z`).

Calibration data: 1470 ± 325 samples (monkey J), 1379 ± 157 (monkey L).

### 1.3 The intuitive mapping

A modified Kalman filter maps **factors** (not units) to 2-D cursor velocity, fit by ML against the monkey's *intended* velocity during calibration (straight from cursor to target at current speed, or straight from centre to target at a constant 0.15 m/s). The paper notes performance was qualitatively similar to a standard Kalman filter. Closed form:

```
x̂_t = M1 x̂_{t−1} + M2 u_t
M1  = A − K C A
M2  = K S_z β                (K = steady-state Kalman gain)
```

`M1 ≈ I` in practice, so the cursor **integrates** and closed-loop speeds far exceed the open-loop term `M2 u_t`. This matters for us — see §9.6.

**The load-bearing design choice is that the decoder is two-stage**: `u → (β) → z → (K) → velocity`. That gives two distinct places to insert a perturbation, which is the entire experiment.

### 1.4 The two perturbations

`g_WM` is a 10×10 permutation matrix, `g_OM` a `q`×`q` permutation matrix (each row and column has a single 1).

| | what is permuted | formula | preserves | changes | control space |
|---|---|---|---|---|---|
| **Within-manifold (WM)** | the 10 z-scored factors | `M2,WM = K g_WM S_z β` | unit → factor relation | factor → velocity relation | re-oriented **inside** the manifold |
| **Outside-manifold (OM)** | the `q` units, before FA | `M2,OM = K S_z β g_OM` | factor → velocity relation | unit → factor relation | re-oriented **outside** the manifold |

To recover under WM the animal must learn new associations between *existing* co-modulation patterns and cursor motion. Under OM it must learn to generate *new* co-modulation patterns.

Permutations keep the same numbers, reassigned — but this is **not** magnitude-matching by construction. `M2,OM = M2 g_OM` is a column permutation of `M2` and does preserve its norm; `M2,WM` permutes rows *inside* the product with `K` and does not. The paper matched magnitude **empirically**, at screening time (§1.5).

**What "outside the manifold" does and does not mean.** It does *not* mean outside the region where activity varies. Off-manifold directions carry real variance — the private, per-unit variability `Ψ`. Off-manifold control means driving the *uncoordinated* parts of the population in a coordinated way: physically possible, but not supported by existing wiring. That is why it is hard rather than impossible, and preserving this property is the single hardest constraint on the transplant (§4.3, §8.2).

### 1.5 Choosing a perturbation (the difficulty match)

Three offline steps on the first 200 (J) / 150 (L) baseline trials:

1. **Candidate set.** WM: all 10! = 3,628,800 permutations. `q!` is intractable, so OM candidates were restricted — for monkey J, 6 million random permutations of only the highest-modulation-depth units (39 ± 18 units permuted); for monkey L, units assigned to 11 groups (10 equal groups of the highest-modulation units, plus an 11th holding the lowest-modulation remainder) with the 10 groups permuted as blocks, giving exactly 10! candidates. Monkey L's construction **equalises the search-space size across perturbation types**; monkey J's does not, and the paper concedes the difference "may have affected the monkey's ability to learn outside-manifold perturbations."
2. **Open-loop velocity.** Mean z-scored count vector per target `u_B^i` (i = 1…8, binned 300–1300 ms (J) or 300–1100 ms (L) post trial start); open-loop velocity `x_OL^i = M2,P u_B^i`.
3. **Selection window.** Keep candidates whose per-target angular deviation from the intuitive open-loop velocity falls in a **session-specific** range (endpoints averaging 19.7° ± 7.0° and 44.4° ± 8.9° across sessions) and whose magnitudes fall in a session-specific absolute velocity range; pick a survivor arbitrarily. Aim: "not too difficult to use, nor so easy that no learning is needed."

### 1.6 Calibration: three recipes, none using a controller

Each day opened with a calibration block whose activity fit both the manifold and the intuitive mapping.

- **Monkey J, method 1 — no assistance.** The animal plays 80 trials (10 per target) using the *previous day's* intuitive mapping. Viable because its signals were stable across days.
- **Monkey J, method 2 — total assistance, never faded.** Passive **observation**: the cursor completes the centre-out task by itself at a constant 0.15 m/s for 80 trials while the monkey watches and is rewarded. Zero control. A decoder is fit from that activity, then the animal goes straight into full BCI control.
- **Monkey L — an assistance ramp.** Needed because its signals were less stable *and* "the observation-based calibration procedure was not as effective at generating an intuitive decoder." 16 observation trials → fit → the monkey drives, but **any movement perpendicular to the straight path to the target is scaled by 0** (rail-guided) → refit after 8 trials → perpendicular scale 0.125 → refit → over **80 trials of closed-loop control** until the scale reaches 1. The mapping was fit from those 80 partial/full-control trials, so calibration was 96 trials in total for this animal.

So the arc "assistance progressively withdrawn" is genuinely in the paper — as autopilot fading, not a limb controller — and **the animal is never told anything, at any transition.**

### 1.7 Task and block structure

Centre-out: cursor (r = 18 mm) starts centred; one of **8** peripheral targets (r = 20 mm, 150 mm (J) / 125 mm (L) out, 45° apart); 300 ms freeze; up to **7500 ms** to acquire; juice on success; 200 ms inter-trial; 1500 ms time-out penalty on failure. Baseline movement times were **near 1000 ms**, so the limit was ~7.5× the median and bound only under perturbation. Blocks are seamless:

```
calibration (80; 96 for L) → baseline (400 J, 250 L) → perturbation (600 J, 400 L) → washout (200)
```

Nominal versus actual: baseline ran 386.9 ± 82.5 (J) and 292.1 ± 43.5 (L) trials. No cue as to which perturbation type was coming; monkey L's order was set randomly by computer, monkey J's "often alternated across consecutive days." On nine days multiple perturbation sessions ran off one calibration.

**Exclusions were unbalanced by condition** and this is worth carrying: sessions with < 100 perturbation trials were dropped, and for monkey J that was 11 sessions of which **9 were OM** — the animal quit more often under OM. Analysed n: WM 28 (J) / 14 (L), OM 39 (J) / 15 (L).

### 1.8 Amount of learning

Performance is a 2-vector of (z-scored success rate, z-scored acquisition time), z-scored across all non-overlapping **50-trial bins** in baseline + perturbation + washout for that monkey.

```
P_B       = mean performance over baseline bins
P_P(j)    = performance in the j-th perturbation bin
L_max     = P_B − P_P(1)                              (‖L_max‖ = the initial impairment)
L_bin(j)  = ⟨P_P(j) − P_P(1), L_max⟩ / ‖L_max‖²        (SIGNED — see below)
L_session = max_j L_bin(j)
```

1 = full recovery to baseline, 0 = none.

**The published equation (19) is `‖L_proj‖ / ‖L_max‖`, unsigned, and that must be a typo:** Fig 2d's histograms run to −0.5, which an unsigned ratio cannot produce, and an unsigned projection would score a bin that degraded *further* as positive learning and then feed it to a `max`. We use the signed projection above; `nc/metrics.py` implements it and a test pins the sign convention.

Two further properties of this metric that do not transfer for free. `L_max` is a *maximum* only because of an empirical fact about the animals — "performance during the perturbation blocks did not exceed the performance during the baseline blocks" — not a property of the construction. And `L_session` is a max statistic, so it grows with the number of bins. See §5.

### 1.9 Result, and how strongly the alternatives were ruled out

WM learning ≫ OM learning: P < 10⁻⁶ (monkey J), P < 10⁻³ (monkey L), two-tailed Welch t-tests.

Five alternatives were addressed by explicit matching: **initial impairment** (P = 0.93 / 0.64); **mean principal angles** between intuitive and perturbed control spaces (P = 0.13 / 0.11, all in 40–80°, so not pure workspace rotations); **required preferred-direction change** (P = 0.24 / 0.50), computed under a minimal-change strategy with closed form `u_P^i = u_B^i + M2,Pᵀ(M2,P M2,Pᵀ)⁻¹(M2 − M2,P) u_B^i` then cosine tuning `u_B^i(k) = m_k cos(θ_i − θ_B(k)) + b_k` refit per unit **over the 8 target directions** (a different `u_B^i` window, 300–1000 ms, is used here than for perturbation selection); **search-space size** (monkey L's grouped construction only — not monkey J, who supplies the headline p-value); **hand movements** (negligible, equal).

Two caveats the paper states and we should carry. The alternatives "did reveal interesting secondary aspects of the data; they partially explained within-category differences in learnability, albeit in an idiosyncratic manner between the two monkeys." And the **after-effect** convergent evidence was significant for monkey J (Wilcoxon rank-sum, P < 10⁻³) but **not** for monkey L (P > 0.05, trend in the same direction) — so this measure was underpowered at n ≈ 14/15, which matters because we plan washout length from it and list after-effects as a novelty. The other convergent line — the gap present from the earliest sessions, never improving with practice — is not applicable to us at all (§4.7).

### 1.10 Dimensionality, and its dependence on sample size

`d = 10` was **fixed a priori** for every day and session. The cross-validated FA sweep (`d = 2…30`, 4-fold, n = 88 days) was **retrospective**: estimated intrinsic dimensionality 9.81 ± 0.31, range 4–16, with the 10-D model within one s.e. of the best on 89% of days and within two s.e. on all.

Two things from Extended Data Fig 7 that constrain our design directly. The estimate **grows with the number of calibration samples and had not saturated** at 1600 points (rising ~4 → ~12 over 400–1600), and the caption lists what it depends on: "the richness of the behavioural task, the size of the training set, **the number of neurons**, the dimensionality reduction method and the criterion for assessing dimensionality… should only be interpreted in the context of these choices, rather than in absolute terms." And the robustness assurance is one-sided: "the effects would have been similar as long as we had identified at least a small handful of dimensions."

Finally, the paper's argument that mis-estimation is self-penalising — "both types of estimation error would have decreased the measured difference in the amount of learning" — does transfer: under-estimating `d` puts nominally-outside directions inside the true manifold (low-alignment arm too easy); over-estimating puts nominally-inside rotations outside it (`a = 1` arm too hard). Both compress the dose-response slope.

---

## 2. The three spaces, and what alignment means

Everything below lives in a strict nesting. Confusing these is the main way to misread the design.

```
ambient space     D = d_model ≈ 4096      every activation is a point in here
  manifold M      d = 10                  where the SHARED (co-varying) structure lives
    control space  2                      what the decoder reads and turns into velocity
```

The manifold is **not** 2-dimensional; the control space is, because velocity has two components. They are fit by different criteria on different data: the manifold by **variance structure with no labels** (factor analysis), the control space by **predicting the move, with labels** (ridge). **PCA plays no role at any level** — benchmarked and unnecessary, since factor analysis fits directly on 30,000 × 4096 in 14 s / 3.1 GB.

### 2.1 Alignment

Shine a light straight down on a stick. Its shadow on the floor is full length if the stick lies flat and zero if it points straight up. Alignment is that, with the floor being the manifold and the sticks being the two directions spanning the control space.

Formally: take a unit vector in the control space, split it into the component inside `M` and the component perpendicular to `M`, and square the length of the inside part. Average over the control space's two directions. Equivalently, the mean of `cos²` of the **principal angles** between the two subspaces (`scipy.linalg.subspace_angles`; two subspaces have one angle per dimension of the smaller).

`a = 1` — entirely inside `M`. `a = 0` — entirely perpendicular. `a = 0.25` — a quarter of each direction's squared length survives projection onto `M`.

### 2.2 Alignment and the perturbation angle are one-sidedly tied

Two distinct angles, and blurring them was a real error in earlier revisions:

- **θ — the angle between the baseline control space and the perturbed one.** How big a change the model just experienced. What Sadtler matched across arms (Fig 3b). Our construction makes both principal angles equal, so "the angle" is one number.
- **the angle to the manifold**, which alignment measures.

They are tied because **the baseline control space lies inside `M`**. A perturbed space far from `M` is therefore far from the baseline space too, since the baseline space is one of the things inside `M`. Formally `a ≥ cos²θ`:

| θ (perturbed vs baseline) | lowest reachable alignment |
|---|---|
| 45° | 0.50 |
| 60° | 0.25 |
| 77° | 0.05 |
| 90° | 0.00 |

**The bound is one-sided.** Fixing θ puts a **floor** under alignment and no ceiling: `a = 1` is reachable at *any* θ, including 90°, because `M` has 10 dimensions and the control space uses 2, leaving 8 spare directions inside `M` to rotate into. Only `d ≥ 4` is required, and it never binds at `d = 10`. This asymmetry is what makes "fix θ, vary alignment" workable: θ can be chosen to serve the low-alignment end without ever costing the `a = 1` arm.

**Consequence for reading the original — corrected in revision 5.** An earlier revision claimed Sadtler's OM perturbations had alignment ≈ 0.03–0.6 given angles of 40–80°. That inverted the bound. Knowing θ gives `a ≥ cos²θ` and nothing more; the paper never measured alignment for its OM perturbations. The disproof is immediate: his *WM* perturbations have `a = 1` exactly and occupy the same 40–80° band (Fig 3b shows the two types' angle distributions are indistinguishable), so the angle cannot pin alignment.

The conclusion "never perpendicular to the manifold" survives, and more strongly than stated: an 80° session forces `a ≥ 0.03`, but a **40° session forces `a ≥ 0.59`** — a nominally outside-manifold control space that is *majority aligned* with the manifold. **This is a better explanation for the paper's own unresolved puzzle** — that a few OM perturbations were readily learnable (Fig 2d, ED Fig 6) — than the underestimated-dimensionality story it offers, and it is worth reporting. It also means an `a = 0.05` arm would be *harder* than anything Sadtler's OM condition could have been.

---

## 3. The task

A closed-loop navigation game. The subject emits **sequences of direction symbols**; the marker moves; the subject sees where it ended up and emits again. No free text, no reasoning block, greedy decoding.

### 3.1 Board and geometry

Continuous 2-D plane, marker starting at (0, 0). **The target angle is sampled uniformly at random; the radius is fixed at 8.** Acquisition radius 1.5, unit step 1.0. Shortest solutions run 7 to ~11 moves, median ≈ 8.5.

**Why the angle is random.** With a fixed set of 8 targets and ~32 perturbation trials the model sees each target about four times, so it could recover by memorising eight lookup entries — "for target 3, write this" — instead of learning the mapping. That is the tabulation failure mode the large action space exists to prevent (§3.4), re-entering through the *target* space, and in the performance curves it would be indistinguishable from real learning. A random angle means a target is essentially never repeated.

**Why the radius is nonetheless fixed.** Distance sets difficulty, and difficulty variation feeds directly into moves-to-acquire and success rate — the two quantities the learning metric is built from. Sadtler held movement amplitude and target size constant for exactly this reason, and says so explicitly, invoking Fitts' law when deriving his required preferred-direction changes.

**The 8 directions are now analysis bins, not targets.** The continuous marker-to-target angle is binned into 45° sectors to feed the cosine-tuning fit and the per-direction screens. Diagonal *sectors* remain unreachable by any single repeated symbol, so mixing is still what the task demands.

**Move budget 36 ≈ 4× the median** — the 7500 ms timeout analog. The paper's ratio was ~7.5×; 4× is a declared compromise (§9.7), because a tighter ratio collapses success rate immediately under any perturbation and would destroy the initial-impairment match G4 gates on, while 7.5× spends context on long post-perturbation wandering.

**Every number in this subsection is provisional and frozen after G3**, which measures the actual moves-to-acquire distribution. They live in `nc.task.BoardConfig`.

The marker position shown to the model is **rounded to integers** — the model's world is a grid — while the internal state stays continuous, so the angular-error metric keeps full resolution rather than being quantised to the lattice. One consequence to watch: the *model's feedback* is quantised, so a perturbed displacement below ~0.5 units is invisible to it. That interacts with the greedy fixed point in §8.5, and it is why the gain is set so the median step is 1.0 rather than something small.

### 3.2 Alphabet and reply format

Alphabet `r l u d` for the pilot (`rludqezc` with diagonals available as the G2 fallback). The reply is a bare space-separated sequence, length **chosen by the model**, between `k_min = 3` and `k_max = 12`:

```
r r u r u
```

Nothing else. Out-of-alphabet characters, an empty reply, or a length outside the band is an **invalid move**: nothing happens that turn, and the model is told so by a line that is **part of the frozen turn template declared in the system prompt** — not a mid-session announcement (§3.3).

**Space separation is not cosmetic**: it is what makes each symbol its own token, which is what makes the per-symbol read possible. Verified on `Qwen/Qwen3.5-9B`, `google/gemma-4-12b-it` and `openai/gpt-oss-20b`: space separation gives exactly one token per symbol on all three (the leading space is absorbed — `['r', ' r', ' u', …]`), and it is the *only* universal separator (newlines cost two tokens per symbol everywhere; commas work on Qwen and gpt-oss but not gemma). Assert one-token-per-symbol at load time per subject; this is a vocabulary accident, not a guarantee.

**Variable length is a designed degree of freedom.** One symbol per turn is tight closed-loop control; twelve is open-loop for twelve moves. The model chooses its own feedback granularity, and the chosen length is a logged behavioural signature.

**Measured 2026-08-10 (G4, n = 8): the predicted signature is NOT there.** The design anticipated the model might *shorten* its sequences under perturbation to obtain tighter feedback. It does not — at `a = 1.0` median length **rose** by 3.0 moves, at `a = 0.2` it fell by 1.0, with per-session values scattering +1/+2/+4/+6 and −3/0/+2/−3. Within the perturbation block, first-half versus second-half length is flat in 7 of 8 sessions. Baseline lengths also differed between arms (5.0 vs 8.5) despite nothing distinguishing the arms before the perturbation, so the apparent arm difference is session-to-session variation that happened to sort by arm — itself a warning about reading any four-session contrast. Recorded as a null so no reader expects this measurement to have worked. It also reproduces something the original had: the number of 45 ms bins per trial varies with how long the trial takes.

### 3.3 Prompts, verbatim

```
You are moving a marker on a grid. Each turn you will be told where the marker is and where the
target is. Reply with a sequence of moves and nothing else.

The moves are:  r = right,  l = left,  u = up,  d = down

Write them separated by single spaces, for example:  r r u r
Write at least 3 and at most 12 moves in one reply.
If a reply is not in this form, nothing happens that turn and you will be told so.

Your goal is to land the marker on the target using as few moves as possible.
When the marker reaches the target, a new trial starts with the marker back at (0, 0).

The marker does not always move the way you expect. When it does not, try different sequences
and watch what actually happens, to work out what produces the movement you want.
```

A turn:

```
User:  Trial 1. Marker (0, 0). Target (8, 0). Moves left in this trial: 36.
Model: r r r r
User:  Trial 1. Marker (4, 0). Target (8, 0). Moves left in this trial: 32.
Model: r r r
User:  Trial 1: TARGET REACHED after 7 moves.
       Trial 2. Marker (0, 0). Target (-6, 6). Moves left in this trial: 36.
```

(Diagonal targets display rounded: (−6, 6) is the integer rendering of (−5.66, 5.66).)

**That is the only prompt in the experiment.** Nothing about activations, decoders, layers, manifolds or neuroscience is ever said, and **no message is ever delivered mid-session** — not at the perturbation, not at washout. The decoder drives the marker from trial 1 and every change to it is silent, exactly as the animals were never cued at any transition (§1.6).

The last line is the **exploration prior**, stated up front before any data exists precisely so it cannot cue a particular moment. It supplies what the monkeys' months of prior BCI training supplied and a cold-start context cannot. Withholding it would not make the experiment cleaner — it would confound "cannot produce the required activity" with "did not think to try" (§8.4).

**There is no controller-detach step.** Revision 3 opened with a phase executing the letters literally and then announced a switch; dropped, because the intuitive decoder is fit to reproduce the emitted move (§4.4), so writing `r` genuinely moves the marker right and the model's experience is indistinguishable from literal execution. There is no discontinuity to announce. If the decoder turns out not to be immediately usable (G3 fails), the fallback is the paper's own remedy — monkey L's unannounced assistance ramp — not an announcement (§9.8).

### 3.4 Why this shape of task

**The action space must be large and combinatorially structured.** A small discrete action set (e.g. `MOVE dx dy` at one decimal = 441 options) turns this into a bandit: with ~200 moves of experience the model can tabulate "this action goes up-left" and recover under *any* decoder, while off-manifold directions have almost no reachable variance across so small a set and are dead rather than hard. Both failure modes at once, and a large arm difference for entirely uninteresting reasons. Variable-length sequences over a 4-symbol alphabet give `Σ_k 4^k` actions, so tabulation is impossible, while the set stays **exactly enumerable and uniformly samplable** — which is what makes the reachability screen (§4.6) computable at all.

**No free text.** An earlier revision had the model write a short sentence alongside its move. That works but drags in semantic confounds and is inconsistent with running the rest of the programme with reasoning off. The symbol alphabet gets the large action space without any of it.

**The observation must be minimal.** `u_t` is a function of the whole context, and the context contains the target. Every extra piece of state in the prompt raises the share of the activation driven by the prompt rather than the model's choice — the confound §4.4 exists to close. So the prompt carries marker position, target position and remaining budget, and nothing more.

---

## 4. Reading, manifold, decoder

### 4.1 Ambient space

**100 coordinates, drawn at random, resampled per session.** The full residual is captured on the GPU; the population is then drawn *offline* from those cached reads, so draws cost nothing and we can have many.

**Why 100 and not all 4,096.** Three reasons, and only the third is about cost.

*It is what makes the factor-analysis model defensible.* A diagonal `Ψ` asserts that the coordinates are conditionally independent given the factors. At `D = 4096` with 10 factors that claims 4,086 directions of residual-stream variance are mutually independent noise, which is plainly false — the residual has correlational structure at every scale, so `Ψ` absorbs structure it should not and the "shared vs private" split stops meaning what we claim. With 100 randomly drawn coordinates the pairwise correlations are sparser and a 10-factor-plus-diagonal model is a reasonable approximation, for the same reason it was reasonable across 90 electrodes.

*It makes the manifold the right kind of object.* Sadtler's framing is that an existing network constrains what patterns *a subset of its neurons* can exhibit. The intrinsic manifold is a property of the recorded subpopulation. Reading every coordinate leaves no "rest of the network" to do the constraining.

*And the sample requirement becomes the paper's.* Sadtler had 90 units and ~1,400 samples, `N/D ≈ 15`. At `D = 100` a few hundred trials reaches the same ratio. The `N ≈ 30k` target in revision 5 had no derivation behind it: the real constraint is only `N > D`, so that off-manifold directions have *measured* variance rather than none, and everything beyond that was my margin.

**This is random drawing, not selection.** No variance ranking, no informativeness screen — that is where selection artifacts enter, and it is what bit the neuron work in the sibling project. Two pre-declared exceptions, both about signal quality rather than usefulness, mirroring Sadtler discarding electrodes whose waveforms were not action potentials: coordinates carrying **massive activations** (attention-sink dims) are excluded from eligibility, since one landing in a draw would dominate that draw's covariance and make it behave unlike the others; and all coordinates are **winsorized at 0.1 / 99.9** then z-scored, by a rule frozen before the FA fit. The number excluded is reported.

**Resampled per session.** Each session draws its own 100 coordinates, fixed for that whole context — a draw is the analog of one recording day. 20 draws for the pilot; scaling to 100 is minutes of CPU and no GPU. This buys population generality the animal study could not have with one implanted array, and it is why the primary regression takes the draw as a grouping factor (§5): between-session variance now includes which-coordinates variance.

**Read layer: chosen by the perturbation screen, not by decodability alone.** Eight depths are captured, evenly spread. A layer that decodes well but admits no matched, reachable perturbations is useless, so the layer is selected in G2 jointly with θ (§4.6) — maximise held-out decodability *subject to* admitting matched candidates at every alignment level. Revision 5 chose on decodability alone; the measured depth profile was violently non-monotonic (usable at 0.32, **at chance** at 0.52, usable at 0.81), which is why four probes are not enough and why the choice should not rest on one criterion.

### 4.2 The read site

**Read the residual at letter *j*'s token, and decode the direction of letter *j+1*.** The state that *precedes* a move, i.e. the model's plan for it — the intent reading. A `k`-letter reply yields `k−1` samples, since the last letter has no successor. One read returns the whole span, so calibration costs one prefill plus one read per turn.

Note this is a **label alignment, not a different capture**: the residual at position *j* both contains letter *j* and is the state from which letter *j+1* is predicted. Same read positions either way.

**Revision 5 read the wrong one and every number it produced was discarded.** It decoded letter *j* from state *j* — a readout of the letter already present rather than of the intent to move. That is not what a BCI decoder does: the monkey's M1 activity is the motor command being issued, and there is no separate action channel to read after the fact. For us the token *is* the action, so reading at the token means reading a record of what was already emitted, and the apparatus reduces to a fancy identity map on letters. The measured support for this is direct: revision 5's working layer was the one where letter identity was most prominent (within/between symbol variance 4.7, the lowest of four depths), i.e. the decoder worked precisely where it was most letter-classifier-like.

Three arguments I gave for the record reading, now withdrawn. *"A pre-move state cannot carry control"* — overstated; it reduces to the first move of each reply being prompt-determined, fixed by dropping it. *"It makes the choice-sensitivity diagnostic vacuous"* — wrong test; for an intent readout the evidence that it reads the model's choice is simply that it predicts the move the model makes on held-out data. *"It forces a calibration template that differs from play"* — real, but the point of the design is to capture intention, so the answer is to fix the calibration (§4.5), not to abandon intent.

**Greedy decoding becomes mandatory, not preferred.** The state before a token encodes the model's *distribution* over the next token, not which token is realised. Under sampling, fitting state → realised-next-letter carries irreducible error equal to the sampling entropy. Under greedy the realised letter is the argmax, so the map is deterministic and learnable.

**The risk this reading carries, and it is the real one.** A competent model's plan is "head toward the target," so an intent decoder may turn out to be a *target-direction* readout — in which case the marker sails to the target regardless of what the model writes and the model looks like a perfect controller while contributing nothing. That is the prompt-leak confound in full force, it is measurable, and §4.5 says how.

Optional pooling: mean over `w` positions ending at *j*. Default `w = 1`; `w = 4` is a G2 fallback, closer to reading a *co-modulation pattern* than a single state.

### 4.3 The manifold

```
u   = winsorize-then-z-score(residual at a move-symbol position)     u ∈ R^D
FA(d = 10) → Λ, Ψ ;   β = (I + Λᵀ Ψ⁻¹ Λ)⁻¹ Λᵀ Ψ⁻¹     (Woodbury: only a d×d inverse)
S_z = diag(1 / sd_calib(β u))
```

Define the **intrinsic manifold as `M := rowspace(β)`**, a 10-dimensional subspace of the ambient space. This is a **repair of the paper's definition, not a preference.** The paper says `colspace(Λ)` and also that the control spaces are spanned by the rows of `M2`; but `rowspace(M2) ⊆ rowspace(β)`, which equals `colspace(Λ)` only when `Ψ ∝ I`. So the paper's own sentence "by construction, the intuitive mapping lies within the intrinsic manifold" is true under `rowspace(β)` and false under `colspace(Λ)`. The WM/OM contrast is likewise exactly "stays in / leaves `rowspace(β)`": `rowspace(K g_WM S_z β) ⊆ rowspace(β)` always, while `rowspace(M2 g_OM)` generically leaves it. The bound in §2.2 also requires this definition.

**`d = 10` is fixed a priori**, as the paper fixed it for every session. We do *not* pick the cross-validated peak: by §1.10 that estimate still grows with sample count at the paper's scale, so peak-picking at our 20× samples and 45× coordinates would estimate a different, sample-size-dependent quantity. The cross-validated curve and cumulative shared-variance curve are reported as a retrospective check (the Fig-4 reproduction), and a robustness sweep over `d` either side of 10 is pre-registered. `d = 10` also leaves 8 spare directions inside `M`, which the within-manifold rotation needs (§2.2).

### 4.4 The decoder

**SUPERSEDED 2026-08-11 by §7.2c, and the supersession was not propagated to the code until 2026-08-18 — see §7.2i.** The paragraph below is the original rationale and is kept for the record. Fit to reproduce **the direction of the move whose token is being read** — not target-directed intent. Sadtler fit against intended velocity because a monkey has no emitted action other than its neural activity; we do, and fitting to intent creates a fatal confound, since the target coordinates are in the prompt and an intent-fit decoder could score well by reading the target from the context while the model contributes nothing, sailing the marker to the target unaided.

```
g_j = unit direction of move j       ( r=(1,0), l=(−1,0), u=(0,1), d=(0,−1) )
W   = ridge fit   (S_z β u_j)  →  g_j
M2  = c · W S_z β
cursor:  p ← p + M2 u_j              applied once per emitted symbol
```

The finished decoder is one 2 × D linear map, *factored through the manifold* — which is what makes the perturbations possible. Its row space is the control space, inside `M` by construction.

Because token identity is strongly linearly decodable from the residual, the intuitive mapping is near-perfect, baseline proficiency is close to guaranteed, and no assistance ramp or injected noise is needed (§9.8). It is faithful in spirit: Sadtler's intuitive mapping was *defined* as the one the animal could already use proficiently, and an action readout is the LLM-native way to guarantee that. And it changes nothing about the science, because the **perturbed** decoders are not action readouts — they are different linear functionals, so the model can no longer just write the move it wants.

With 4 symbols, `g_j` takes four values and they are symmetric (`r = −l`), so `W`'s two rows are the right-vs-left and up-vs-down discriminant directions.

**Gain `c`** is set from calibration alone — so the median decoded step length over the calibration reads equals 1.0 — and then **frozen identically across all arms**. It is *not* derived from G3, which would be circular since G3 cannot run without it; G3 verifies the resulting move distribution and may trigger one re-freeze before any measured session.

No temporal filter (`M1 = 0`): with a discrete move budget, integration would drift and be hard to arrest. Declared, and it has a consequence for the screening window (§9.6).

### 4.5 Calibration

**The template is byte-identical to play.** No extra instruction line, nothing distinguishing a calibration turn from a game turn. Calibration is *just teacher-forcing the reply*: render exactly the prompt the model would see, force the reply we want. If calibration prompts carried a line play prompts do not, the decoder would be fit on one context distribution and used on another, and a baseline failure could be pure covariate shift. Trial index and remaining budget are filled by sampling a trial position uniformly from the ranges the play distribution produces.

**Forced replies are target-reaching, not random.** The monkey's observation calibration shows the cursor travelling *to the target*, in a straight line, successfully. So each forced reply is a **shortest lattice path from marker to target with randomised symbol order** (`r r u u`, `u r u r`, `r u r u` all serve one target), so order carries no target information and each direction is served by many sequences. `nc.task.shortest_path` builds them. Marker positions are sampled across the whole workspace, not only the origin, so the required direction varies *within* a path. Matching reply content to play matters for the same reason as matching the template.

**Random paths are the validation control, not the training data.** Target-reaching paths correlate the emitted move with the target, so that leak is *measured* rather than designed away: **fit on target-reaching paths, validate on a held-out set of random paths**, where move and target are uncorrelated by construction. If the decoder still predicts the emitted move there, it is reading the move. Strictly better than revision 3's design, which made the leak unfittable and therefore unquantifiable. (This set is the **random-path validation set**; the separate uniform sample used by the reachability screen is the **uniform emission sample** — different objects, §4.6.)

**Two weaknesses of forced replies, and their checks.**

*Ambiguous continuations.* For a diagonal target either symbol is a good next move at most positions, so the activation carries a partly-committed state, whereas for an axis target it does not. Randomised order makes the blending symmetric, so the fitted direction should be unmoved and only its precision suffers; the geometry is in fact benign, since the ridge splits the activation into a component shared by both symbols (mapping to the diagonal, the correct direction for a diagonal target) and one distinguishing them. What does not wash out is that **precision may differ per target**, and the per-target screens assume comparable quality. G1 reports angular decoding error broken down per target and by ambiguity.

*Off-policy forcing.* Forcing a symbol the model would not choose means reading the residual of a model writing a low-probability token, and surprisal has an activation signature; at play every token is high-probability under the model's own policy. Target-reaching paths fix most of this — a shortest path is close to what the model would write, whereas random paths are maximally off-policy — leaving order preference. G1 records the model's own probability of every forced token (free, same forward pass), compares that distribution against self-chosen tokens from the reference sessions, and tests whether decoding accuracy depends on it. **Fallback if either check fires:** draw the order on-policy by letting the model generate and keeping only generations that are shortest paths.

**Sizes.** **1,200 target-reaching turns** and **600 random-path turns**, ~1,800 total. At ~4.2 usable samples per target-reaching turn (letters minus one, for the label shift) that is ~5,000 fit samples, so `N/D = 50` per 100-coordinate draw — well past the paper's 15, and enough that each split-half sits at 25 and the stability check is itself meaningful. The random set gives ~3,500 emissions.

**Reply length in calibration is chosen by us, and it is currently a guess.** Each target-reaching reply is a shortest path truncated to a random length in [3, 12], giving a mean of ~5.2 letters; without truncation nearly every reply would sit at the 12-letter cap with no length variety. But the calibration length distribution should match what the model actually does in play, which is unknown until G3 — so this is one more quantity frozen after G3 (§7). The mismatch is a covariate shift in the context (how many letters precede a read), not in the template.

**What the 600 random-path turns are for — and what they are no longer for.** They are the **emission sample** for the reachability screen (§4.6), which needs activations and no labels; that use sets the size, since a 1% density floor over ~3,500 emissions gives ~35 hits per direction.

They can **no longer serve as the decodability validation set**, and this is a direct consequence of the label shift. On a random path the state at letter *j* cannot know that letter *j+1* is a random letter — nothing in the context implies it — so the decoder scores near zero there **by construction**, not because it is bad.

**The replacement validation is already in the target-reaching data, and it is better.** At a diagonal sector both `r` and `u` are optimal, the randomised order picks one, and the state genuinely cannot know which; at an axis sector the next letter is fully determined. So compare the two: at unambiguous positions the decoder should predict the actual next letter, while at ambiguous positions it can at best point at the *target direction*, the average of the two valid moves. That contrast measures directly whether we are reading intent or reading the target out of the prompt — the §4.2 risk — on data we already have, with no extra reads.

Calibration happens entirely outside the measured sessions. The **coordinate draw, and hence the manifold and decoder, are per session** (§4.1) — but all draws come from one cached set of reads, so this costs no additional GPU. Perturbations are drawn fresh per session too.

### 4.6 Perturbations

Permutations were Sadtler's tool because `q!` enumeration was intractable and permutations gave a countable candidate list; the magnitude-matching came from screening, not from the permutation (§1.4). Neither reason holds for us. Permuting `D` non-exchangeable residual coordinates makes the unit→factor map essentially random, so the perturbed control space is a random 2-D subspace of a 4096-D space picking up almost none of the activation variance — a dead marker, not a hard one. And we can characterise the reachable set. So the control spaces are constructed directly:

- **`a = 1`**: replace `W` with a rotated `W'` in the `d`-dimensional factor space. `M2,WM = c · W' S_z β`, row space inside `M` by construction. The continuous generalisation of Sadtler's factor permutation.
- **`a < 1`**: rotate toward directions in the orthogonal complement of `M`, with the split set to hit the requested alignment.
- `nc.manifold.build_perturbation(M2, Mb, θ, a, rng)`. Both principal angles to the baseline control space equal θ; `a = 1 − sin²θ·sin²ψ`. It **raises** on an unreachable request rather than silently missing.

**Constants** (`nc/manifold.py`): `ALIGNMENT_GRID = (1.0, 0.7, 0.4, 0.2, 0.1)`, `RHO_MIN = 0.01`. **θ is not a constant** — see below.

**θ is screened, not fixed — measured 2026-08-10, and this reverses revision 5.** Fixing θ at 60° gave an alignment floor of `cos²60° = 0.25`, and putting a grid level *at* the floor turned out to be a degeneracy rather than a perturbation.

At the floor the rotation direction is forced entirely outside the manifold, so the in-manifold part of the perturbed control space is exactly `cos θ` times the **original** direction — scaled, not rotated. Since activation variance concentrates near the manifold, the decoded output becomes ≈ `cos θ` × the original velocity plus off-manifold noise: same direction, reduced speed. Measured at alignment 0.25: median velocity change **3.0°** against a required 20–45°, magnitude 0.40 (predicted `cos 60° = 0.5`), and **0 of 200 candidates admissible**. In the model's experience that is not a scrambled mapping, it is the same mapping running at 40% speed — nothing to learn from.

Worse, it is a gradient, not a cliff. Measured velocity change fell monotonically with alignment: 43.4° at `a = 1.0`, 38.1° at 0.75, 26.6° at 0.5, 3.0° at 0.25. **Severity co-varies with alignment**, which is the exact opposite of the matched difficulty the design rests on.

*(These two findings are geometric and survived the discarding of revision 5's numbers: they follow from `q' = cos θ·q + sin θ·(out-of-manifold)` plus variance concentration, for any control space however fitted. Only the specific degree values are gone.)*

So **severity is the thing that must be matched**, since it is the initial-impairment proxy of Fig 3a; θ is only a means to it. Sadtler did not fix his angles either — he screened candidates into a severity window and *then* reported that the angle distributions happened to be similar (Fig 3b, P = 0.13 / 0.11). We do the same: for each alignment level, search over θ for candidates whose **measured severity** lands in one common window, subject to the level sitting comfortably *above* `cos²θ` rather than at it, and report the resulting θ per level as a measured quantity.

**The grid extends to 0.1**, five levels. `a = 0.1` requires θ above `arccos(√0.1) = 72°` merely to exist, and above that with margin to avoid the degeneracy — so ~80°, which the screen finds. Lower alignment is a stronger test of the hypothesis; the cost is a more drastic perturbation, and ED Fig 6b warns that larger angles reduced learning even within the manifold for monkey L. That tension is now resolved by measurement instead of by my picking a number.

**The layer is screened jointly with θ.** A layer can decode well and admit no usable perturbation, or vice versa, so both are chosen in one search over (layer, θ, alignment level) — eight candidate depths (§4.1).

**Why θ is fixed at all, and the case for sweeping it — DEFERRED, not dismissed (Davide 2026-08-07: keep fixed for now).** Fixing θ is what delivers the paper's Fig-3b matching control, so alignment is the only manipulated variable. But three arguments say θ deserves to be an axis eventually, and they are recorded here so the choice is not mistaken for a settled one.

*It is itself an interesting variable, and the paper treated it as one.* Sadtler's angles ranged 40–80° across sessions and ED Fig 6b regresses learning on them, finding an effect for monkey L. Fixing θ discards an axis the original exploited.

*We are guessing its value.* 60° is anchored on the paper's median with nothing measured. If learning is θ-sensitive, the wrong value floors or ceilings every arm — which is precisely the argument used to reject 80°, so it applies with equal force to adopting 60°.

*It is the lever for equalising initial impairment, which may be needed.* At fixed θ both ends of the grid sit the same angular distance from baseline, which is the clean part. But a low-alignment control space reads private variance, so its decoded velocities are smaller and noisier, and the magnitude window may be unmatchable at low `a` (§4.6, feasibility limits). If magnitude cannot be equalised then θ-matched-but-impairment-unmatched is the worst outcome, since **impairment is what Fig 3a actually matched and θ was only a proxy for it**. In that case θ should be freed per level to equalise impairment, and the deviation declared.

Against sweeping: a 4 × 3 grid at n = 10 is 120 sessions instead of 40, and splitting n across θ weakens an alignment test already thin at n = 10 per level.

**Planned resolution when the budget allows:** measure θ rather than guess it, by running the over-long G4 at two angles (45° and 75°, four extra sessions) to get the θ → initial-impairment relation and a first look at θ-sensitivity, then freeze θ on the criterion of a moderate *and matchable* impairment. Keep the confirmatory primary as a single-θ alignment sweep for power, pre-register the θ × `a` grid as the named extension, and state in any writeup that the single-θ result is a slice.

**Graded alignment is primary.** Regress the learning metric on `a` across the five levels; a monotone dose-response is more informative and far more defensible than a two-bar figure, and it converts a possibly-definitional binary (§8.2) into a quantitative law. `a = 1` versus `a = 0.1` remains reportable as the paper-faithful contrast.

Screening, in order. **Both criteria below were corrected against measurement on 2026-08-10** (`scripts/screen.py`); the versions this doc carried before were mis-specified in ways that a candidate's failure would have been silently blamed on.

**The governing principle: the intuitive decoder must pass its own screen.** It has alignment exactly 1 and is by construction usable, so if it fails, the screen is wrong and nothing may be read into the perturbed candidates. `screen.py` asserts this and aborts. It fired on the first run, which is how both errors below were found.

1. **Severity, measured per sample — not on mean activity.** The paper screens the angular deviation of the open-loop velocity computed from per-target *mean* spike counts. **That probe does not transfer to off-manifold control spaces here.** `u_B^i` is a mean, so its private (off-manifold) component averages toward zero, leaving `M2,P u_B^i` dominated by the small in-manifold part of the control space. Measured at alignment 0.25: mean-based deviation had a median of **3°** — the screen called a thoroughly off-manifold perturbation almost harmless — while per-sample deviations were large. Averaging destroys exactly the variance a low-alignment control space reads. So severity is the **median per-sample** angular deviation between perturbed and intuitive decoded velocity over the emission sample, required in **[20°, 45°]**, with median speed ratio in [0.3, 1.5]. Distributional, not "all 8 direction means inside the window", which is unsatisfiable whenever deviations spread across directions. *(This also bears on §2.2: Sadtler's mean-based screen would have understated a genuinely off-manifold perturbation, consistent with his OM control spaces being only partly out of the manifold.)*
2. **Reachability, at a tolerance a discrete alphabet can meet.** Draw a **uniform emission sample** (random symbol sequences of random length in game contexts), read once, cache. Two criteria. **Plane-spanning**: the largest angular gap in the achievable per-move velocity directions must be under 180°, so positive combinations of moves reach *any* net direction — because with a 4-symbol alphabet a single move cannot point diagonally, and net direction comes from mixing moves across a sequence. **Per-direction density**: `ρ_i(P)` = fraction of sampled states with `angle(M2,P u, θ_i) ≤ 60°` and `|M2,P u| ≥ 0.6 ×` the *intuitive* median step; require `min_i ρ_i(P) ≥ RHO_MIN = 0.01` and match `min_i ρ_i` **across alignment levels** to within 20% relative. The tolerance is 60° and not 30° because at 30° the **intuitive** decoder scores 0.001 on diagonal directions and fails its own screen — a single move is 45° from a diagonal by construction. The speed floor uses the *intuitive* median, never the candidate's own: scaling it by the candidate's own speed would let a perturbation producing uniformly tiny velocities pass, which is exactly the failure being screened for (`nc.manifold.reference_speed`, with a test).
3. Draw one survivor per session, seeded.

**A failure mode this screen now detects, seen in the plumbing run:** at low alignment the two control directions both read mostly-private variance and the decoded output can collapse toward a narrow cone — angular gap above 180°, so entire half-planes of net direction become unreachable. That is "unreachable, not unlearnable" (§8.2) and must truncate the alignment grid rather than be papered over by relaxing the window.

**`u_B^i` is binned by required direction, not by target index.** Because calibration samples marker positions across the workspace, "target `i`" no longer implies a single required direction; the 8 bins are 45° sectors of the direction from marker to target.

Recorded per chosen perturbation: both principal angles to the baseline control space; alignment `a`; the required-preferred-direction-change analog (paper's closed form plus cosine refit **over the 8 direction bins**); and `min_i ρ_i`.

**Two feasibility limits to check at G2, not assume.** The complement of `M` carries only private variance `Ψ`, which is typically far smaller than shared variance along manifold directions, so "matched achievable-velocity statistics" may be unattainable at low `a` — in which case the screen's magnitude window fails and the alignment grid must be truncated rather than the window relaxed. And unlike a permutation, a *rotation* can demand activation values outside the observed per-coordinate range; the paper's z-scoring existed precisely to prevent that ("so that the perturbed mappings would not require a neural unit to fire outside of its observed spike count range"), so G2 additionally reports, per candidate, the fraction of the uniform emission sample whose decoded velocity requires no extrapolation.

If no candidate at the lowest alignment clears step 2, that is the finding and is reported as "off-manifold control spaces are unreachable for this subject" — weaker, but honest. Levers: 8-symbol alphabet, longer sequences, windowed read, shallower layer.

### 4.7 What is measured

Three performance signals per trial: **success** (target reached within budget), **moves to acquire**, and **per-move angular error** between the decoded displacement and the ideal straight-to-target direction.

The third has no clean analog in the original, where intended direction had to be inferred. We know the target, so angular error is available on **every emitted symbol** — hundreds of measurements per session rather than a handful of binned rates. It is also what makes an easy task measurable: baseline at ceiling is desirable and matches the animals, but success rate then carries little baseline information and moves-to-acquire is coarse, whereas angular error degrades and recovers continuously whether or not the model still succeeds. (Note the paper did *not* rely on acquisition time alone: ED Fig 2a shows the success-rate *change* under perturbation was significant for monkey J, P < 10⁻³.)

**Exploration measures**, logged per bin per arm, because they decide whether a session is interpretable at all (§8.4): number of distinct emitted sequences; symbol-distribution entropy; fraction of emissions that are *not* the arithmetically-optimal path; chosen sequence length; duplicate-reply rate; and the spread of visited activations, including the fraction of their variance outside `M`. A perturbation block in which the model never departed from optimal-path emission is **not evidence about learnability at any alignment**, and is reported as such rather than pooled into an arm mean.

The last of these doubles as the **reassociation-versus-new-patterns** analysis (Golub et al. 2018 / Oby et al. 2019 analog): does the model *attempt* to leave the manifold at low alignment and fail, or never attempt?

---

## 5. Analysis plan

**Primary.** Regression of `L_session` on alignment `a` across the **five** levels (1.0, 0.7, 0.4, 0.2, 0.1), `n = 10` sessions per level, testing the slope against zero at α = 0.05, two-tailed, with the **coordinate draw as a grouping factor** — each session has its own 100-coordinate draw (§4.1), so between-session variance includes which-coordinates variance and treating sessions as exchangeable would understate the standard error. Reported with per-level means and the slope's CI.

**Co-primary.** The same regression with `L_session` computed on the (success rate, angular error) pair instead of (success rate, moves). Two co-primaries, so **Holm correction across the two**; both reported regardless of outcome.

**Secondary, pre-registered, uncorrected and labelled as such:** the `a = 1` vs `a = 0.25` endpoint contrast (Welch); after-effect at washout; the three matching checks (initial impairment, principal angles, required-PD change) as null-expected comparisons across levels; and the exploration measures.

**Power, restated for four levels.** The grid truncated to {1.0, 0.7, 0.4, 0.2} because `a = 0.1` proved unreachable (§7.3). Recomputing the linear-trend target: at `n = 10` per level and α = 0.05 two-sided, 80% power corresponds to a full-range difference of **d ≈ 1.17**, against d ≈ 1.08 had five levels survived — so the truncation costs little. Both are thinner than the paper's n = 28/39, and the coordinate-draw grouping factor costs more on top, so this stays a declared limitation rather than a comfortable margin. The animal effect was large (d > 1), which is the only reason n = 10 is arguable at all.

**The null distribution of `L_session` is positive — measured, 2026-08-10.** Over 200 synthetic sessions with *no* learning planted, `L_session` averaged **0.12** (success/moves pair) and **0.18** (success/angular-error pair), reaching p95 of **0.28** and **0.37**. It is a maximum over bins, so noise alone lifts it well clear of zero. Planted recovery gave mean 0.73 with p05 of 0.61 / 0.56, so the statistic does separate the two — but three consequences follow.

Any criterion of the form "`L_session > 0`" is nearly vacuous, and the G4 gate is written against the null floor and the no-perturbation control instead (§7). **`tail_mean` is unbiased under the null** (measured 0.008 / −0.011), so it is the statistic for any absolute claim. And **the primary analysis is immune to this**, because a constant offset cancels in a slope — which is a further argument for the graded regression being primary rather than a two-arm contrast. `tests/test_metrics.py` pins all three as properties.

**Metric handling.** `L_bin` uses the signed projection (§1.8). Two consequences we accept and report rather than patch: `L_bin > 1` is reachable for us, since unlike the animals an LLM at ceiling may exceed baseline under perturbation, which makes the normalisation interpretable but not bounded; and `L_session = max_j L_bin(j)` grows with the number of bins, so **block lengths are identical across arms** and a second, pre-registered non-max summary (mean of the last two bins) is reported alongside. Absolute values are not comparable to the paper's.

---

## 6. Arms

| arm | n | purpose |
|---|---|---|
| graded alignment `a ∈ {1.0, 0.7, 0.4, 0.2, 0.1}`, θ screened per level | 10 each | primary: learnability versus manifold alignment |
| no-perturbation | 8 | drift baseline: does performance change across blocks with the decoder fixed? |
| observer, at `a = 1.0` and `a = 0.1` | 8 each | an external model emits the sequences from the same visible history; the subject's activations still drive the marker. If observer ≈ self, the finding is about the geometry of the emission→activation map, not introspective access. Reuses `neurofeedback/producers/`. |
| random-feedback | 4 | marker moves randomly: confirms measured "learning" is not confabulated |
| literal-execution reference | 4 | letters executed verbatim, no decoder: gives the task-competence ceiling, separating "can the model do the task" from "can it do the task through the decoder". No perturbation transition; not used for fitting. Also the source of self-chosen-token logprobs for the §4.5 off-policy check. |

**98 sessions per subject** (50 graded + 8 no-perturbation + 16 observer + 4 random-feedback + 4 literal-execution reference). Perturbation level drawn per session by a seeded RNG; the model is never told; the analysis script treats all levels identically. Sadtler's blinding argument transplants — matching monkey L's computer-randomised order rather than monkey J's alternating one.

**The observer arm is maximally off-policy**, and inherits the §4.5 hazard in its strongest form: another model's tokens are teacher-forced into the subject, so every read is of a token the subject may assign low probability. The logprob check from §4.5 is run on this arm too, and a large surprisal gap versus the self arm is a caveat on the comparison rather than something to correct away.

---

## 7. Gates, and what must be calibrated

### 7.0 The smallest pilot that carries signal

Run in two steps, with a hard stop between them. Subject: `Qwen/Qwen3.5-9B` (cached, the `vllm_lens` default, and the model LITTLEBOY-B's state-channel effect appeared in).

**Step 1 — G1 + G2. Under 10 GPU-minutes plus ~20 CPU-minutes, and no session loop at all.** Only `scripts/calibrate.py` and `scripts/screen.py`: 1,800 forced-reply turns, eight read depths in one batched pass, no generation anywhere, then 20 coordinate draws and the joint (layer, θ, alignment) search on the cached reads. It answers every question that can kill the project — is the *next* letter decodable from the state before it, is the decoder reading intent rather than the target, does within-symbol variance survive at `D = 100`, and does an admissible perturbation exist down to `a = 0.1`. **If G2 fails there is no experiment**, and we know it without writing the session loop.

**Step 2 — the paper's Fig 2a/2b analog. Four sessions: 2 at `a = 1.0`, 2 at `a = 0.1`,** with a deliberately over-long perturbation block. This adds `nc/session.py` and `scripts/run.py`. At n = 2 per arm there is no statistic and none should be reported — but the paper's own headline figure is **one representative session per condition**, and four learning curves answer, by inspection: does the marker reach targets at baseline, does the perturbation visibly impair, does the model *explore*, and does performance recover at `a = 1` while not recovering at `a = 0.25`. It also produces the block-length and per-session-cost numbers that §7 needs frozen.

What would stop us at each step: **G1** — decodability below the floor, within-symbol variance collapsed (§8.3), or the ambiguous-position test showing the decoder reads the target rather than the intent (§4.5). **G2** — no admissible candidate at the lowest alignment at any layer or θ, in which case truncate the grid rather than relax the windows. **Step 2** — no recovery at `a = 1.0`, in which case check the exploration measures before concluding anything about capability (§8.4); or recovery at *both* extremes, which would say the screened severity window was too mild.

Only after step 2 commit to n = 10 per level and the control arms.

Revision 1 asserted block lengths, bin sizes and budgets as hyperparameters. They are not; each is set from pilot data and frozen before the confirmatory runs.

- **Step gain `c`** from calibration step lengths (§4.4) — median 1.0. Verified, not derived, at G3.
- **Move budget and target radius** from the G3 moves-to-acquire distribution; provisional values in §3.1.
- **Baseline block length** from the precision needed on `P_B`, which anchors `L_max`: choose it so `s.e.(P_B) ≤ 15%` of the typical initial impairment.
- **Bin size** from the same variance plus quantisation (an 8-trial bin quantises success rate to 12.5% steps).
- **Perturbation block length** from an over-long G4 run: fit the `a = 1` recovery curve, take the plateau plus margin. Those same runs must also be checked for a *late* rise at low alignment — if off-manifold learning is merely slower, block length alone manufactures the result.
- **Washout length** from the observed after-effect decay in G4, remembering the original's after-effect measure was underpowered at n ≈ 14/15 (§1.9).
- **`k_max`** from the context budget given the frozen move budget.

| gate | what | pass criterion | cost |
|---|---|---|---|
| **G0** | **DONE 2026-08-07.** Tokenizer assertion on 3 families (§3.2); `nc/task.py`, `nc/metrics.py`, `nc/manifold.py`; **76 tests** (`venv-vllm/bin/python -m pytest tests/ -q`, ~60 s) over env geometry, angular error, reply validity, prompt hygiene, calibration paths, FA/decoder algebra, rotation at prescribed alignment, the frozen constants, and `L_session` on synthetic curves | all pass | done |
| **G1** | One read pass at **8 depths**; then per coordinate draw (20 draws): next-letter decodability, **validated by the ambiguous-vs-unambiguous contrast** (§4.5) rather than on random paths; split-half manifold stability via principal angles; within-symbol variance; per-direction and per-ambiguity error; forced-token logprobs as a diagnostic; retrospective FA dimensionality curve | **R² > 0** (the decoder beats the constant predictor that always emits the mean direction), read against a label-noise ceiling near 0.58 rather than 1.0 (§7.3). This is the only gate. Three hand-picked cut-offs (cv-R² ≥ 0.30, within-bin direction variance ≥ 0.20, per-direction spread ≤ 15°) were removed on 2026-08-11 and the within-bin measure was withdrawn entirely; see §4.5. Split-half principal angles, per-direction spread and the ambiguous-vs-unambiguous contrast are reported, not gated. **The §4.2 target-detector risk is currently ungated** | ~6 GPU-min + ~10 CPU-min |
| **G2** | **The kill gate.** Joint search over (layer, θ, alignment level) on cached reads, per draw | ≥ 1 candidate per level clearing the severity window, the plane-spanning test, `min_i ρ_i ≥ 0.01`, and matched on severity across levels — at one layer common to all levels | CPU only |
| **G3** | 4 baseline-only sessions; freezes move budget, target radius, block lengths, bin size, and the per-session GPU cost | success ≥ 0.70; median moves-to-acquire ≤ 12 with IQR ≥ 3 | ~2 GPU-h |
| **G4** | 4 sessions at `a = 1.0` + 4 at the lowest usable alignment, deliberately over-long perturbation block | initial impairment not significantly different across arms; **`L_session` at `a = 1.0` above the measured null floor (p95 = 0.28 paper / 0.37 angle, §5) and above the no-perturbation control** — *not* "> 0", which is nearly vacuous given that floor; **`tail_mean > 0`**, the unbiased statistic; no late rise at the lowest alignment | ~4 GPU-h |

Failure routes: **G1** → other depths, `w = 4`, 8-symbol alphabet, longer sequences. **G2** → truncate the alignment grid, `w = 4`, shallower layer, 8-symbol alphabet; if still failing, report "unreachable" honestly. **G3** → the monkey-L assistance ramp (§9.8), then larger workspace / more moves / smaller step. **G4** impairment mismatch → tighten the screening windows; no `a = 1` learning → check exploration measures before capability (§8.4).

**Cost: calibration is now measured, closed-loop play is not.** A read pass of 3,750 turns at 4 depths took **805 s** on one A100-80GB (Qwen3.5-9B, teacher-forced prefill with activation capture) — about 4.7 turns/s. Scaling to 1,800 turns at 8 depths gives roughly **6–8 GPU-minutes** for G1, and G2 is CPU-only. What remains unmeasured is the per-session cost of *play*, which needs generation and grows context; G3 fixes that number, and no multi-subject roster should be planned before it reports. See §7.2.

---

### 7.2 Time estimate

**Measured anchors, and a correction.** Two read passes on Qwen3.5-9B, one A100-80GB, teacher-forced prefill with activation capture:

| depths | rate | note |
|---|---|---|
| 4 | **4.7 turns/s** | 3,750 turns in 805 s |
| 8 | **1.7 turns/s** | measured 2026-08-10 |

So **activation capture, not prefill, is the bottleneck**, and it scales with the number of layers captured — doubling depths cut throughput by ~2.8×. My first estimate assumed prefill dominated and was ~3× optimistic. The 8-depth sweep therefore costs ~12 min for 1,200 fit turns plus ~6 min for 600 validation turns plus ~2 min of logprob passes: **~20 GPU-min for G1**, not the 6–8 originally claimed.

| stage | work | estimate |
|---|---|---|
| **G1 reads** | 1,800 turns, 8 depths, at the measured 1.7 turns/s | **~18 GPU-min** |
| **G1 logprob pass** | 1,800 turns, prefill only, no capture | ~2 GPU-min |
| **G1 fits + diagnostics** | 20 draws × 8 depths: FA + ridge at `D = 100`, split-half, per-direction breakdowns | ~10 CPU-min |
| **G2 screen** | 20 draws × 8 layers × 5 levels × θ search × 200 candidates | ~30–60 CPU-min |
| **Step 1 total** | | **~20 GPU-min + ~1 CPU-h** |
| **G3** | 4 baseline-only sessions — *the unmeasured quantity* | see below |
| **G4** | 8 sessions, over-long perturbation block | 2 × G3 |
| **Full n** | 98 sessions | 25 × G3 |

**The one number we don't have is the per-session cost of play**, and it dominates everything after Step 1. A session is ~40 trials × ~8.5 moves ≈ 340 moves; at ~5 letters per reply that is ~70 turns, each needing one generation plus one read, with context growing to ~15k tokens. Two bracketing estimates:

- *Optimistic*, from prefill throughput with 16 sessions in lockstep: ~10 s per lockstep turn at long context → ~12 min per batch of 16 → **~0.75 min/session**, so ~1.2 GPU-h for all 98.
- *Pessimistic*, if generation latency dominates and batching helps less than hoped: ~0.5 GPU-h/session → **~50 GPU-h** for all 98.

That is a 40× spread, which is why G3 exists and why it is only 4 sessions. **Plan Step 1 and G3 now; do not commit to the full n or a multi-subject roster until G3 reports a real per-session number.** For reference, the whole of Step 1 plus G3 plus G4 is 12 sessions of play — even at the pessimistic rate that is ~6 GPU-h, which is the sensible commitment.

Three subjects at full n would be 3 × the full-n figure, so somewhere between 4 and 150 GPU-h. Not a plannable range yet.

### 7.2b Intent versus target: what actually gates

**Superseded 2026-08-10 — the intent-vs-target gate was testing the unachievable.** An earlier form of the G1 criterion asked whether, at ambiguous positions, the decoded velocity points at the realised letter rather than at the target direction. It cannot: at an ambiguous position two symbols are equally optimal and the randomised order picked one, so the state genuinely does not know which, and the best possible decoder points at the target direction. A large angle-to-letter there is *expected*, not diagnostic. That contrast is now reported descriptively.

**Withdrawn 2026-08-11 — the replacement gate was no better, and there is now no gate at all.** The replacement was a within-bin direction-variance fraction: of all the variance in the decoded direction, the share surviving *inside* a required-direction bin, on the reasoning that a readout fixed by the target coordinates in the prompt would be constant within a bin. It does not work, for three reasons that compound.

1. It cannot separate the case it was built to detect from a benign one. Zero within-bin variance is equally consistent with a target detector that ignores the model and with a faithful readout of a model whose policy is deterministic given the target.
2. The decoder is a ridge regression, so it predicts the conditional mean. Where the label is unpredictable from the state, collapsing to the bin mean is the *correct* fit and the lowest-error one, not a failure.
3. In our calibration the label is unpredictable by construction over most of the data. Paths are randomised-order shortest paths, so at an ambiguous position it is our own RNG that picks `r` or `u`, and the state cannot know which. Measured on the artifact, **71.8%** of fit positions are ambiguous. The best possible readout therefore outputs the bisector there and scores near zero on within-bin variance.

Taken together the measure tracks label noise rather than whether the readout carries the model's intent, so it is deleted rather than demoted. The earlier hand-picked cut-offs went with it (cv-R² ≥ 0.30, within-bin ≥ 0.20, per-direction spread ≤ 15°); I chose all three by hand and none came from a null distribution. G1 now gates on one definitional floor, R² > 0, meaning the decoder beats the constant predictor that always emits the mean direction.

**Two things this leaves open, and they matter.** First, **the §4.2 risk is now unmeasured.** Nothing in the pipeline tests whether the decoder reads the target instead of the model's intent. Second, the withdrawn floor is what selected the operating layer, so §4.1's rule reverts to plain "maximise R² subject to admitting matched candidates", which does not pick layer 21 (§7.3).

**What would actually test it.** The forced letter is the wrong ground truth for intent, because our RNG chose it. The model's own next-token distribution at the read position is the right one: it is causally available at state *j*, and it is what "about to write" means. The test is whether the decoded direction tracks the model's probability mass over `r/l/u/d` rather than only the direction to the target. Note that forcing different letters at *j+1* is **not** a valid version of this: causal masking means the state at *j* cannot depend on *j+1*, so nothing would move. Any interventional variant has to act on the prefix. Not implemented.

### 7.2c The label is the model's own intent distribution (added 2026-08-11)

**The forced-letter label could not distinguish the hypothesis from its null, and this was measured, not suspected.** Calibration paths are shortest paths, so the letter that comes next is nearly a function of the direction to the target, and the target coordinates are printed in the prompt. On the 2026-08-10 artifact:

| predictor of the forced next letter | held-out R² |
|---|---|
| best decoder over all 8 depths and all d ∈ {10, 20, 30} | 0.504 |
| conditional mean given the 8-bin required direction | 0.519 |
| the required direction used directly, no fitting at all | **0.562** |
| oracle: the realised letter where determined, the bin mean where tied | 0.571 |

No decoder reached the trivial predictor, and the headroom between the target baseline and the oracle is **0.009**, because 71.8% of positions are ties whose letter our own RNG chose and nothing can predict that. So the §4.2 risk was not merely ungated, the corpus made it unmeasurable.

**The fix is a label change, not a corpus change.** At read position *j* take the model's own next-token distribution restricted to `r/l/u/d`, renormalise, and use its expected direction `E[dir] = Σ_s p_s · dir(s)`. This label contains no RNG of ours, so the ceiling rises from 0.571 to 1.0, and at a tie it carries the model's private preference. The corpus is kept: the forced replies sit at median token probability 0.692 against 0.162 for random paths, so they are already close to what the model would write, and generation would not fix the identifiability problem anyway (a sampled letter is drawn *after* the state at *j* exists, so it is unpredictable in the same way; a greedy tie-break would be a geometric rule the prompt determines).

**Removing the noise helps the trivial predictor too, so the comparison had to be made against a strong adversary.** Every readout score is now reported against three prompt-only predictors on the same label and the same split (`nc/labels.py`): the unfitted required direction, its 8-bin conditional mean, and a ridge on geometry features including `|Δx| − |Δy|`, which is the "prefer the longer remaining axis" tie-break rule. `best_prompt_only` is the number a decoder has to beat, and G1 now gates on beating it.

**Verified before committing the re-run** (`scripts/check_label.py`, 1,000 turns, 4,224 positions, 844 held out, a deliberately generous readout of ridge on 512 raw coordinates with no factor-analysis bottleneck):

| label | best prompt-only | activation readout, layer 28 | gap |
|---|---|---|---|
| forced next letter | 0.5625 | 0.5233 | **−0.046** |
| model intent direction | 0.7964 | 0.9747 | **+0.178** |

The prompt baseline rises from 0.56 to 0.80 exactly as predicted, and the readout rises further. In exact-match terms, against the model's own most likely letter:

| predictor | all positions | at tied positions | where the prompt's nearest letter is wrong (23.4% of positions, n = 198) |
|---|---|---|---|
| ridge on geometry features | 0.776 | 0.693 | **0.091** |
| activations, layer 21 | 0.915 | 0.891 | 0.773 |
| activations, layer 28 | **0.956** | 0.945 | **0.899** |

Majority-class rate is 0.282. The last column is the decisive one: where the prompt cannot tell you which way the model will go, the activations do, nine times in ten. (The unfitted target direction scores 0 there by construction, since the subset is defined by its disagreement, so the geometry ridge is the prompt-side number to quote.)

**Mechanism.** The commitment `|E[dir]|` has median **0.84** at tied positions, where an even 50/50 split would give 0.707. The model has a strong private preference where the geometry is indifferent, and that is the part no function of the prompt reproduces.

**One caveat to state whenever the 0.97 is quoted.** The model's logits are a linear map of its final residual stream and our label is derived from those logits, so a linear readout at layer 28 of 32 is partly reproducing the unembedding by construction. The *absolute* R² is therefore not evidence of anything deep; the gap over the prompt is. Whether the intent is readable *before* the output pathway is what decides how interesting this is, and the 8-depth profile from the re-run answers it directly.

### 7.2d The read site was off by one, and it dropped the model's first move (2026-08-12)

**The defect.** For a k-letter reply the pipeline read k activations and applied k−1 displacements. The convention was "state at letter j's token, labelled with letter j+1", so the state that generated letter 0 was never read: it sits at the final PROMPT position, and the span returned by the backend began at letter 0. The consequence is not a small indexing wart. The model wrote a k-letter plan and the marker executed a truncated version of it, always missing the first move, roughly 15% of every plan and always from the front.

**Why it matters beyond tidiness.** The experiment asks whether in-context learning can re-learn a perturbed readout. If the first move of every reply is silently discarded, the mapping the model would have to learn is not stationary from its own point of view: identical plans produce different marker trajectories depending on where the dropped move fell. That makes this defect a live candidate explanation for the central null result of the 2026-08-12 run, the complete absence of re-learning across 40 perturbation trials. **No claim about the model failing to adapt should be made from any run before this fix.**

**The fix.** One activation per emitted letter, namely the one that generated it. Row j of the read span is the state the model was in when it produced letter j, which for j = 0 is the final prompt position. This needed one extra position on the left of the span, added to the sibling backend as `read_residual_multi_batch(..., extra_left=n)`. That parameter is additive with default 0, and it turned out the position was already inside the captured tensor and merely sliced away, so nothing about the forward pass changed.

Three places moved together, because a decoder must be applied under the convention it was fit on: `calibrate.py` now emits one labelled sample per letter with `req` measured at the matching marker position, the letter distributions shifted from `plp[ss+j+1]` to `plp[ss+j]` (which is also the first time the first letter's intent distribution exists at all), and `nc/session.py` applies k moves for k letters.

**Verified against the real model before committing compute**, because a silent span shift would corrupt everything downstream: the widened span's row j+1 equals the old span's row j to 0.00000 on every row, for single and batched reads, batched agrees with solo exactly, the wrong alignment is off by 8 to 12, and the prepended token decodes to the final prompt position. A regression test in `tests/test_session.py` now asserts k moves for k letters and that the runner requests `extra_left=1`; the fake backend mirrors the parameter so the fixture cannot pass on the old convention again. The test that previously asserted k−1 was rewritten, and its docstring records that its stated reason (that the prompt position "would be a pure function of the prompt") was wrong.

**What this invalidates.** Every closed-loop run before 2026-08-12, which is all of §7.3, §7.4 and the 96-session crossed run. The decodability results are re-measured rather than merely re-labelled, since the corpus now has ~15% more samples including the first-letter states. Two findings from the crossed run are expected to survive because they concern the decoder's geometry rather than the applied trajectory: that aim error almost fully determines performance (rank correlation −0.94 with success), and that matched severity does not match difficulty (success spanning 0.05 to 1.00 at a constant 31.7 to 32.4°). The recovery result is the one that must be re-established from scratch.

### 7.2e Effective rotation, and why it replaces severity as the difficulty axis (2026-08-12)

**Definition.** For each applied move, the angle between the decoded step and the canonical direction of the letter that produced it. Move *j* applies the state that generated letter *j* (§7.2d), so the displacement `path[j+1] - path[j]` pairs with `reply.split()[j]`. Averaged over moves it is the perturbation *as the model experiences it*: how far the marker's motion departs from what its own letter literally means. Computed from the session logs alone, no extra GPU.

**Why it was needed.** The screen matches perturbations on severity, the median angle by which the perturbation changes the decoded velocity. On the 2026-08-12 crossed run severity was matched to within 0.6° across all 72 perturbed sessions and success still ranged from 0.05 to 1.00, rank correlation **−0.01**. Effective rotation on the same sessions correlated **−0.80**. So the quantity being controlled was not the quantity that determines difficulty.

**Nominal rotation is not effective rotation, and not even monotone in it.** Measured on that run: alignment 1.0 and 0.7 were built with the same nominal θ = 50° and landed at 71° and 41° effective; alignment 0.4's larger nominal 60° landed at 42°, below both. On the grid sweep, at fixed alignment 1.0, θ = 70° produced *less* effective rotation (53 to 56°) than θ = 60° (57 to 59°), with correspondingly better performance. Turning the rotation knob further does not reliably rotate what the model experiences, because θ rotates the 2D control space inside the 10-dimensional manifold while what reaches the model is the induced change in the letter-to-motion map.

**Baseline is 15°, not 0.** The fitted readout is already about 15° off the literal letter meaning before any perturbation, which is the floor for this measure and the reference any perturbed value should be read against.

**A caveat on the measure.** It is computed from applied displacements only, so turns rejected as too short contribute nothing. Arms with more invalid replies are therefore measured on a slightly selected subset of their moves.

### 7.2f The rotation x alignment grid (2026-08-12)

**Design.** All 23 feasible (θ, alignment) cells plus one no-perturbation cell, six target seeds, one coordinate population, 144 sessions. The grid is triangular because `alignment >= cos²θ` is one-sided: low alignment exists only at large θ. Length parameters were fixed rather than swept, at 80 perturbation trials and a 48-move budget, with batch 4 and `max_model_len` 81920.

**Severity is recorded, not gated.** The severity window (20 to 45°) exists to hold difficulty in a band, and spanning that band is the entire point of the sweep; gating on it would truncate exactly the range being measured, and no cells were skipped once it was lifted. The reachability and plane-spanning floors stay, because they guarantee a solution *exists*, which is a different question from how hard it is to find.

**Split across three jobs BY SEED, not by cell.** Each job runs two seeds across all 24 cells, so a job lost to the 3-hour wall costs precision rather than coverage: what survives is a complete grid measured on fewer seeds instead of a grid with holes.

**Cost note, which sets the ceiling on every future sweep.** Prefix caching is off by design (the science default is the plain full-context read), so every turn re-prefills the whole conversation and total cost grows with the SQUARE of the turn count. Fitted on the v4 logs: 2.1 s per lockstep turn at turn 20 rising to 5.4 s at turn 150, total batch cost ≈ 0.0123·T² + 1.73·T seconds at batch 8. This is why doubling the block roughly quadruples the time, and why validating the prefix-cached read path against full recompute is the highest-leverage engineering task outstanding.

### 7.2g Replication across populations and models: the v6 campaign (2026-08-14)

**Why.** The v5 grid produced the first positive evidence for the paper's actual hypothesis: effective rotation ordered impairment at rank −0.79, and at *matched* effective rotation higher alignment still did better (residual rank correlation +0.49, R² 0.732 → 0.805, and positive in all five matched bands, which curvature leaking through a mis-fitted cubic cannot produce). But it is a single coordinate population of a single model, and §4.5 already established that the population effect can exceed the arm effect. The draw is the unit that has to replicate, so v6 adds a second draw of Qwen3.5-9B (draw 14, the runner-up under the same pre-declared severity-match criteria that picked draw 16) and two draws of a second model.

**Design is held byte-identical to v5** — same 23 cells plus control, same six seeds, same 80/8/8 trials and 48-move budget — so each new sweep is a replication rather than a new condition. `scripts/compare_draws.py` reports the matched-band test as primary and the cubic residual as the weaker, model-dependent version, for every sweep on disk.

**The second model is gemma-4-31B-it.** There is no Gemma 4 27B: the dense Gemma 4 sizes are 12B and 31B, and 27B was Gemma 3. The model id must carry the uppercase B, because the lowercase HuggingFace cache entry holds configs only and would fail offline.

**What had to be verified before spending compute on it** (`scripts/check_gemma_template.py`, all passing). Every label in this experiment is positional — row *j* of a read span is the state that generated letter *j* (§7.2d) — and that mapping rests on tokenizer facts that were Qwen facts until checked:

- One token per move symbol. `r r u` must be exactly three tokens; a tokenizer that merged a run would shorten the span and silently attach every later label to the wrong letter. It holds on Gemma, and sessions additionally force the read response to the *re-joined parsed moves*, so stray model whitespace cannot shift the span either.
- The board protocol needs no system role, which it already satisfies by folding the instructions into the first user message with strictly alternating turns. Gemma 4 in fact accepts a system role, unlike Gemma 2 and 3, but nothing depends on that.
- The default template prefills a *closed empty thought channel* (`<|channel>thought` … `<channel|>`), so the answer follows directly and `extra_left=1` lands on the last prompt token exactly as on Qwen. `enable_thinking=True` would instead inject a system turn and let the model think, which is not wanted here.
- Generation stops at the turn delimiter without help: `generation_config.eos_token_id` is `[1, 106, 50]` and 106 is `<turn|>`, so the Olmo failure mode (running past the turn end into hallucinated turns) does not arise and no `stop_tokens` entry is needed.

**Memory and throughput are sized, not guessed.** Weights are 58.25 GiB of an 80 GB A100, but 50 of the 60 layers are sliding-window (1024) and vLLM 0.23 allocates window-bounded KV for them, so a 32k sequence costs ~3.4 GiB rather than ~26 GiB and batch 2 fits at gmu 0.92. Sessions reach ~26k tokens (39 tokens per turn in *both* tokenizers × 560 turns), which is what sets `max_model_len` 32768 — and incidentally shows v5's 81920 was large by a factor of three. Calibration is the awkward stage: the letter-distribution read uses `prompt_logprobs`, whose fp32 `[batched_tokens × vocab]` buffer is ~2.6 GiB per 2560 tokens at Gemma's 262k vocabulary and can be held twice under step pipelining, so it runs at `max_num_batched_tokens` 2560 (vLLM rejects anything below 2496 for a multimodal model) with gmu 0.87 for slack and `--chunk` 32 to bound the eight-layer activation capture.

**Gemma's throughput was unmeasured, so the wall guard is adaptive rather than a guessed cutoff.** `--deadline-min` needs the batch cost known in advance: set too low it wastes most of a 3-hour slot, set too high it loses the in-flight batch, since `trials.json` is written only when a batch completes. `run.py --wall-limit-min` instead refuses to start a batch when 1.2× the slowest *observed* batch would overrun; the first batch always starts, so the guard calibrates itself inside the slot.

**One self-chaining job, staged by priority.** Only one short-unkillable runs at a time (`QOSMaxGRESPerUser`), so parallel submissions buy nothing. `scripts/v6_campaign.sh` always does the highest-priority remaining work and resubmits itself, which keeps exactly one thing queued and every slot full. The first slot splits the node — Gemma calibration deliberately stays unsharded on one GPU, because merging activations and labels across four processes is the bug class that already cost this project a whole run, so the other three GPUs run Qwen sessions instead of idling for two hours. The two Gemma draws then advance *together*, two GPUs each: total throughput is identical, but two draws at three seeds each answers the population question that one draw at six seeds and one at zero does not.

**Partial results are valid by construction, which is the point given that Gemma's cost is a projection.** Session indices cycle grid cells fastest and the target seed slowest, and every shard works its list in ascending order, so finished sessions always form a *complete grid* at however many seeds completed. Stopping the campaign at any moment costs precision, never coverage.

**If Gemma's readout does not beat its prompt-only baseline, the campaign stops Gemma and says so.** That is a result about the model, not a crash: a decoder that cannot outread the prompt has no readout to perturb, so its grid would measure nothing.

### 7.2h Sessions are checkpointed between trials (2026-08-15)

**The failure.** `run.py` wrote a session's `trials.json` only when its whole *batch* finished, while the per-turn transcript was flushed every turn. A Gemma session takes 25 to 86 minutes and a batch of two takes 49 to 171, against a 3-hour wall, so batches were routinely killed in flight. Fourteen sessions of 74 to 96 trials each were discarded that way, several of them complete, and several shards produced *nothing at all* in a full 3-hour slot. Throughput was ~6 sessions per slot against a projected 12 to 16, and every chain ended in `TIMEOUT` rather than a clean exit, so the analysis tail never ran either.

**The cause was a wrong unit of durability**, not a wrong wall guard. `--wall-limit-min` can only decide whether to *start* a batch, and only from batch times it has already observed, so the first batch of every slot starts unconditionally and can run past the wall on its own.

**The fix.** `run_lockstep` takes two callbacks. `checkpoint(s)` fires after every completed trial and `on_finish(s)` fires the moment a session finishes rather than when its batch does. `nc.session.session_state` serialises exactly what the run configuration does not reproduce: conversation, trial state, RNG state, counters, and finished trials. `cfg`, `readout`, `coords` and `blocks` are deliberately excluded because `run.py` rebuilds them as a pure function of (draw, cell, seed), and a second source of truth for them could disagree with the first. A `config_sha` over that same tuple pins each checkpoint to its cell, so a stale checkpoint from a different design is refused rather than silently resumed into.

**What pins it.** `test_resume_reproduces_an_uninterrupted_run_exactly` kills a session at its first checkpoint, restores it into a fresh object, runs it out, and asserts the trials are identical to an uninterrupted run. Without that equivalence, resumed and fresh sessions would carry silently different data and every aggregate would mix two populations.

**One consequence worth recording.** A resumed session continues under a new shard tag, so its transcript would split across two directories, and `analyze_grid` computes effective rotation straight off that transcript. The prior transcript is therefore copied forward on resume and the log handle opens in append mode, so a resumed session ends with one complete jsonl whose turn coverage matches its trial coverage.

**Recovery of what was already lost.** `scripts/recover_trials.py` re-derives `trials.json` from a transcript: block and alignment are recorded per turn, `acquired` comes from the trial's last turn, `moves_used` is the sum of the per-turn applied-move counts, and `angular_errors` is the concatenation. It is a re-derivation rather than an estimate, and it verified against all 182 sessions holding both files with zero mismatches before writing anything. A trial still in flight when the job died is dropped rather than recorded short, which would have looked like a fast failure. Only two sessions were genuinely unrecoverable-elsewhere: the other orphans had been retried successfully by later chains, and writing all of them would have entered the same session index up to five times and silently weighted it that many times in every aggregate.

### 7.3 Run log

**2026-08-10, Step 1 on `Qwen3.5-9B`** (1,200 fit + 600 validation turns, 8 depths, 20 coordinate draws). G1 reads took 19.5 min; the CPU stages ~35 min on 4 cores. Samples: 5,076 fit (`N/D = 51` per draw) and 3,796 emissions, 0.58 GB cached.

**Next-move decodability is a deep-layer property.** Read the R² column against a ceiling near **0.58**, not against 1.0: squared distance between unit vectors is 2(1 − cos), the four directions are balanced so their mean is ~0, and therefore R² ≈ 2 × mean(cos of angular error) − 1. With 71.8% of positions being RNG-decided ties between adjacent letters, where the best prediction is the bisector at cos 0.707, a *perfect* readout scores about 0.578. Layer 28 reaches 82% of that, layer 21 about 70%.

| layer | depth | R² | amb→target | dir spread | split-half |
|---|---|---|---|---|---|
| 4 | 0.13 | 0.175 | 31.1° | 60.0° | 18.3° |
| 7 | 0.23 | 0.088 | 34.8° | 59.1° | 14.1° |
| 11 | 0.35 | −0.018 | 37.0° | 56.2° | 32.3° |
| 14 | 0.45 | −0.181 | 41.0° | 54.2° | 19.1° |
| 18 | 0.58 | −0.140 | 38.3° | 53.0° | 15.5° |
| **21** | **0.68** | **0.402** | 15.7° | 32.1° | 12.1° |
| 25 | 0.81 | 0.429 | 14.5° | 32.6° | 13.5° |
| 28 | 0.90 | 0.474 | 13.2° | 30.7° | 9.5° |

The next move is decodable only in the deep third of the stack. Layers 4 through 18 sit at or below zero, so at those depths the readout is no better than a constant pointing in the average direction. From layer 21 onward R² jumps to 0.40 and rises slowly to 0.474 at layer 28, and the angle to the target at ambiguous positions tightens in step (38° at layer 18, 13° at layer 28). Population choice does not move any of this (next paragraph), so this is a depth effect.

**Withdrawn 2026-08-11: the depth trade-off is no longer a finding.** This section previously reported a second column, a within-bin direction-variance fraction read as "controllability", and its headline was that the two quantities trade off across depth with an exactly reversed ordering (Spearman −1). That claim is retracted. The measure is dominated by label noise (§4.5), and the anti-correlation is close to definitional rather than discovered: when the label is nearly a function of the target, prediction variance that lives inside a required-direction bin is variance that does not help predict the label, i.e. error, so high R² forces low within-bin variance arithmetically. Nothing was measured wrongly; the interpretation was wrong. What survives from this table is the R² column, the angle columns and the split-half stability.

**Consequence for the layer choice, unresolved.** Layer 21 was selected by the withdrawn floor. §4.1's rule without it is "maximise R² subject to admitting matched candidates", and replaying that rule over this run's own stored scout scores gives **layer 28** (R² 0.474): at 3 scout draws every sampled layer clears the candidate requirement of 9.6, so the constraint binds on nothing and the objective decides alone. **The closed-loop sessions already run therefore sit at a layer the current rule would not pick.** Re-running at layer 28, or arguing explicitly for 21 on other grounds, is a decision to make before any confirmatory run, and G2's severity matching was screened at layer 21 only so it has to be re-screened wherever the layer lands.

Note what that exposes about the rule itself. With the candidate constraint slack for every layer, "max R² subject to admitting candidates" reduces to "deepest layer", since R² rises monotonically with depth over the sampled range. That is a rule with no real content, and it is the same rule that the withdrawn floor had been quietly rescuing.

**Population choice barely matters; depth matters enormously.** At a fixed layer, R² spans only 0.454–0.491 across 20 coordinate draws. But 1 draw in 20 produced an intuitive decoder that failed its own reachability screen, so ~5% of random 100-coordinate populations are unusable — the population-generality number the resampling design exists to produce.

**G2 passes at four levels.** Usable alignments {1.0, 0.7, 0.4, 0.2}; `a = 0.1` unreachable on the real model as the dry run predicted, so **the grid truncates to four levels and §5's power figure needs restating**. At layer 21 severity is matched in **20/20 draws at 2.4% median spread** (at layer 14 it was 18/20 at 10.8%). Note this was screened at layer 21 only, so if the operating layer moves (see above) G2 has to be re-screened there. Magnitude spread remains ~52% — the unresolved caveat of §4.6.

**G3 (baseline closed loop, layer 21, 4 sessions x 20 trials) — CONTROL WORKS.** This was the
feasibility question the whole design hinged on.

| session | success | moves (median) | angular error | invalid |
|---|---|---|---|---|
| 000 | 1.00 | 13.0 | 36.5° | 0.15 |
| 001 | 1.00 | 11.5 | 31.5° | 0.09 |
| 002 | 0.45 | 36.0 | 68.3° | 0.04 |
| 003 | 0.95 | 13.5 | 34.2° | 0.05 |
| pooled | 0.85 | 18.5 | 42.6° | — |

Three of four sessions acquire essentially every target. The emissions show genuine closed-loop
behaviour rather than repetition — `l l l l u u u u u u u u`, then `r r u u`, `r r d`, `r d` as the
marker closes in, with duplicate rates of 0.05–0.13, so the degeneracy that destroyed the 0.6B run
does not appear here. Format compliance is good but imperfect: 4–15% invalid replies.

Session 002 is the informative failure — 68° median error, essentially chance, 91 turns for 8
acquisitions — and is almost certainly a poor coordinate draw, consistent with the ~5% of draws
whose intuitive decoder fails the reachability screen outright. That is the population variability
the resampling design exists to measure, appearing on its first real outing.

**Cost, finally measured: ~29 GPU-seconds per 20-trial session** (115 s for four sessions, 70 turns
each, 1.16 s per lockstep turn at batch 4). Extrapolated to 40-trial sessions the **full 98-session
experiment is ~1.6 GPU-hours**. §7.2's brackets were 1.2 h optimistic and 50 h pessimistic; reality
is at the optimistic end, so the 40× uncertainty that justified gating everything behind G3 is
resolved and a multi-subject roster is now plannable.

**One G3 gate fails and the threshold is probably mine to blame — FLAGGED, not changed.** Median
moves 18.5 against a ≤12 criterion. I derived 12 from the *shortest-path* length (8.5) plus margin,
but a controller with ~34° angular error necessarily overshoots and corrects: the three working
sessions run 11.5–13.5 moves, ~1.5× optimal, against a 36 budget, with IQR 5–10. That is good
control with ample headroom and real resolution, which is what the gate was meant to establish, and
the pooled 18.5 is dragged up by session 002 alone. The question is whether the threshold should be
relative to what a noisy controller can achieve rather than to the geometric optimum. Davide's call.

**G4 (8 sessions × 56 trials, `a = 1.0` vs `a = 0.2`, layer 21) — DOES NOT PASS. Within-manifold learning is not demonstrated, and the campaign stopped here rather than proceeding to full n.**

| metric | arm | `L_session` | `tail_mean` | impairment |
|---|---|---|---|---|
| paper | `a = 1.0` | 0.656 | **+0.10** | 2.44 |
| paper | `a = 0.2` | 0.320 | −0.20 | 2.76 |
| angle | `a = 1.0` | 0.259 | **−0.11** | 2.33 |
| angle | `a = 0.2` | 0.315 | −0.12 | 2.73 |

The gate requires `tail_mean > 0` at `a = 1.0` — the unbiased statistic (§5) — and it is +0.10 on one metric pair and −0.11 on the other. Performance across the perturbation block did not improve and in most sessions drifted slightly worse.

**Three reasons not to read the paper-metric difference (0.656 vs 0.320) as an effect.** The two metric pairs disagree on sign; were alignment driving recovery they should move together. The arm mean rests on a single session — 002 alone gives `L = 1.63`, `tail = +1.12`, while the other three `a = 1.0` sessions give 0.37/0.63/0.00 with tails −0.38/+0.29/−0.62. And session 002 is *the same coordinate draw* that failed G3 at 45% success and 68° error, so the session carrying the arm mean is the one with the worst baseline — precisely the configuration where `L_max` is small and the ratio inflates. That is an artifact.

**Two explanations are ruled out.** Not absence of exploration: distinct-reply fractions ran 0.38–0.52 with duplication 0.08–0.27 through the perturbation block, so §8.4's "won't versus can't" confound is not operating. Not unmatched difficulty: impairment is comparable across arms and the severity match worked as designed — 34.1° at `a = 1.0` (θ = 50°) against 37.8° at `a = 0.2` (θ = 70°).

**THE MECHANICAL CAUSE, found by a binning-free measure — the perturbation floors performance and censors the metric.** Success rate and median moves-to-acquire by block:

| arm | block | success | moves |
|---|---|---|---|
| `a = 0.2` | baseline | 1.00 | 10.5 |
| `a = 0.2` | perturbation | 0.32 | **36.0** |
| `a = 0.2` | washout | 0.84 | 14.2 |
| `a = 1.0` | baseline | 0.91 | 14.6 |
| `a = 1.0` | perturbation | 0.33 | **35.0** |
| `a = 1.0` | washout | **0.97** | 14.6 |

Moves reach 35–36 against a budget of 36, so during the perturbation block the marker almost never arrives — it exhausts the budget nearly every trial. **Moves-to-acquire is therefore censored at the ceiling with no headroom to register recovery**, leaving success rate (pinned at 0.33) as the only live component. The learning metric was measuring a floored, censored variable, which is why no statistic survived its null. This replaces the earlier four-way ambiguity (too short / too harsh / n too small / real null) with one actionable diagnosis: **the perturbation is too severe for the geometry.** §3.1 already flagged that Sadtler's timeout was ~7.5× the median solution while ours is 4× as a "declared compromise" — this is what that compromise cost. His perturbations reduced success without flooring it, leaving acquisition time free to vary inside the limit.

**Washout is a clean positive control that the apparatus works.** Success returns to 0.84 / 0.97 and moves to ~14, so the impairment was caused by the decoder swap and is fully reversed by removing it — not by context growth, fatigue, or session-length drift. Note also that impairment is matched almost exactly across arms (0.32 vs 0.33 success; 36.0 vs 35.0 moves), so Fig 3a's criterion is satisfied: the severity-matching machinery worked, it simply matched the arms at a severity past the useful range.

**Fix is a design parameter, not a rethink**: reduce the severity window, raise the move budget toward the paper's 7.5× ratio, or both — and re-measure the null floor at whatever bin count follows. Davide's call.

**Null floors measured at the bin counts actually used** (300 synthetic no-learning sessions each, 8 baseline + 40 perturbation trials — the G4 configuration):

| bin | pert. bins | metric | L p50 | L p95 | tail p50 | tail p95 |
|---|---|---|---|---|---|---|
| 8 | 5 | paper | 0.082 | 0.260 | 0.000 | +0.166 |
| 8 | 5 | angle | 0.176 | 0.407 | +0.026 | +0.279 |
| 4 | 10 | paper | 0.149 | **0.393** | 0.000 | +0.213 |
| 4 | 10 | angle | 0.390 | **0.641** | +0.064 | +0.390 |

Two things follow. The **angle pair has a far higher null floor than the paper pair at the same bin count** (median 0.390 vs 0.149), so the co-primary metrics need *separate* thresholds rather than a shared one — it is noisier per bin. And bin count must be frozen before the confirmatory run with its floor measured at that count: `a = 1.0` paper moves 0.318 → 0.656 between bin 8 and bin 4 purely through the max statistic.

**Against these floors, G4 becomes unambiguous.** Both arms sit at or *below* the null median on the angle pair (0.259 and 0.315 against 0.390) — not merely undetected learning but below what pure noise produces. On the paper pair, `a = 1.0`'s 0.656 is the only statistic in the run exceeding its p95 (0.393), and it is carried entirely by session 002's 1.63; **excluding that artifact session the remaining three give 0.37 / 0.63 / 0.00, mean 0.333, between the null median and p95 — indistinguishable from noise.** `tail_mean` agrees and is confirmed unbiased (null median 0.000 for the paper pair), with G4's +0.10 inside the null p95 of +0.213. So **no statistic in G4 survives its own null once the artifact session is removed.**

**And the magnitude caveat now has a number.** The `a = 0.2` perturbations have magnitude ratio 0.41 against 0.78 at `a = 1.0`, so the low-alignment marker moves at roughly half speed even with the angle matched — reaching moves-to-acquire independently of the angular perturbation (§4.6).

**No-perturbation drift control (§6), 6 sessions × 56 trials, decoder never swapped (`run.py --sham`) — AND IT BREAKS THE METRIC, not just G4.**

| set | metric | `L_session` | `tail_mean` | impairment |
|---|---|---|---|---|
| **no-perturbation** | paper | **1.658** | **+0.605** | 1.43 |
| **no-perturbation** | angle | **1.700** | **+0.367** | 1.36 |
| G4 `a = 1.0` | paper | 0.656 | +0.103 | 2.44 |
| G4 `a = 1.0` | angle | 0.259 | −0.113 | 2.33 |
| G4 `a = 0.2` | paper | 0.320 | −0.195 | 2.76 |
| G4 `a = 0.2` | angle | 0.315 | −0.122 | 2.73 |

**With nothing perturbed, the metric scores higher than every perturbed arm** — per-session `L_session` of 0.07, 0.93, 0.95, 1.93, 2.28, 3.78. By the design's own "above the no-perturbation control" criterion, G4 fails decisively rather than marginally.

**The cause is structural: `L_session = ‖proj‖ / ‖L_max‖` divides by the impairment.** With nothing perturbed `‖L_max‖` is near zero (0.85–2.31 here) and the ratio explodes. That is the same artifact as G4's session 002, but systematic. So the metric requires a **moderate** impairment — large enough that `L_max` is well estimated, small enough that the outcome is not floored — which is precisely the regime Sadtler was in. **Our parameters miss the band on both sides**: this severity censors the outcome (moves 35/36), and no perturbation explodes the denominator.

**This also invalidates the synthetic null floor above.** Those 300 reps planted a *real* impairment, so they measured the well-behaved regime and returned a reassuring p95 of 0.39. The empirical null spans 0.07–3.78. A synthetic control validated the wrong thing, and would have gone on being trusted without this run — the lesson being that a planted-effect generator cannot establish a null for a *ratio* statistic whose denominator is the effect.

**One clean positive number:** genuine session drift is modest — success 0.94 → 0.88 and moves 13.3 → 16.9 across 56 trials — so context growth degrades performance slightly but nowhere near enough to explain G4's impairment, reinforcing that the decoder swap caused it.

**A bug this run caught, and how.** The screen scored layers purely by how many perturbation candidates they admitted, and chose layer 14 — where the decoder's R² is **−0.18**, worse than useless. G2 duly reported PASS and G3 began there. The doc's rule (§4.1) is "maximise decodability **subject to** admitting matched candidates": candidate count is the *constraint*, decodability the *objective*, and the implementation had them inverted. Caught only because the two reports disagreed about the same layer. Fixed; `--fit-report` is now required for the selection to follow the design.

**The repetition degeneracy is real, and the stuck detector did not catch it.** In a closed-loop test the model emitted `r l d r` every single turn — 100% duplicate replies, 0/3 trials acquired, the full 36-move budget burned each trial — while the stuck counter never fired, because displacements *were* being applied, just identically. `nc/session.py` now has a separate repetition detector; both are tested.



**2026-08-10, plumbing validation on `Qwen3-0.6B`** (200 fit / 150 val turns, 4 depths — a code test, not evidence about the science; `N/D ≈ 1.5` there makes every manifold estimate under-determined). The whole revision-6 pipeline ran end to end: reads → 20 coordinate draws → the joint (layer, θ, alignment) screen. Three things came out of it that matter for the real run.

**The searched-θ mechanism works, and does what a fixed θ could not.** With θ allowed to rise as alignment falls, severity came out matched across four alignment levels: 33.4°, 31.2°, 32.8°, 33.2° at `a` = 1.0, 0.7, 0.4, 0.2, achieved by θ = 50°, 50°, 60°, 70°. Median spread across draws 11%, inside the 20% tolerance, and severity-matched in 4/4 draws. Under a fixed θ = 60° the same quantity ran 43° → 3° (§4.6). This is the matched-difficulty control actually working.

**`a = 0.1` was unreachable, so the grid truncates.** Only θ ∈ {84°, 87°} can even build it, and no candidate at those angles cleared the severity and reachability windows in 3 of 4 draws. Per the standing rule the grid truncates rather than the windows relaxing, giving usable levels {1.0, 0.7, 0.4, 0.2}. Expect the same on the real model; if so the analysis plan's five levels become four and the power figure in §5 needs restating.

**A residual confound the severity match does not control — flagged, not decided.** With the *angular* severity matched across levels, decoded **speed** still falls as alignment falls (magnitude ratio 0.93 → 0.50, median spread across levels 58%). So low-alignment sessions run a slower marker, which reaches moves-to-acquire independently of the angular perturbation. Sadtler screened on a velocity-magnitude range as well as an angle range, so matching magnitude too is within the spirit of the design — but it is a design decision, so the screen now *reports* `mag_spread_median` without gating on it, and the call is Davide's.

**Two implementation bugs found and fixed by this run**, both mine, both in the gates rather than the science: the G2 pass criterion evaluated severity matching only when *every* grid level was present, so a single unreachable level made "matched" unattainable by construction (now: determine usable levels first, match within them); and `eligible_coords` correctly excluded 1 of 1024 coordinates as a massive-activation dim, confirming the quality screen fires.

## 7.4 Run log, 2026-08-11 re-run on the intent label

Artifacts in `artifacts/v2`, sessions in `outputs/v2`. The 2026-08-10 run is left intact for comparison. Calibration took 21 GPU-min for 1,800 turns at 8 depths including the letter distributions, which were captured with only **0.41%** of positions missing a letter from the top-20.

**G1 passes, and now on a criterion that means something.** The gate is R² > 0 *and* beating the best prompt-only predictor, which on the intent label is the geometry ridge at **0.7913**.

| layer | R² | minus the prompt | exact match, all | exact match where the prompt's letter is wrong |
|---|---|---|---|---|
| 4 | 0.562 | −0.230 | 0.683 | 0.411 |
| 14 | 0.069 | −0.723 | 0.511 | 0.380 |
| 21 | 0.793 | +0.002 | 0.784 | 0.558 |
| 25 | 0.838 | +0.047 | 0.801 | 0.584 |
| **28** | **0.891** | **+0.100** | **0.859** | **0.667** |

Prompt-side exact match is 0.745 overall and **0.068** on the 24.3% of positions where the prompt's nearest letter is not the model's preferred one. Majority class is 0.264. Layer 28 gets 0.667 there.

**The depth profile is label-independent.** The R² ordering across depth has rank correlation +0.55 with depth under *both* labels, layer 14 is the worst under both, and layer 4 beats layers 11 to 18 under both. The intent label lifts every layer by a roughly uniform +0.25 to +0.42. So the U shape is a property of the activations, not of the label, and proximity to the output head does not explain it: layer 4 at depth 0.13 decodes intent at 0.562.

**The frozen bottleneck costs about half the available signal.** An unconstrained probe (512 raw coordinates, ridge, no factor analysis) reaches 0.9747 at layer 28 for a gap of +0.178; the pipeline's 100 coordinates and d = 10 reach 0.891 for +0.100. Sweeps on the intent label show the frozen constants are close to saturating *at the operating layer*: d = 10 → 0.894 against d = 30 → 0.925, and n = 100 → 0.875 against n = 800 → 0.911. Mid layers are a different story, where d = 10 → 0.100 and d = 30 → 0.570 at layer 14, so the earlier reading that mid layers "carry nothing" was substantially an artifact of d. Coordinate draw still barely matters: depth accounts for 84.8% of the R² variance, the draw 4.8%.

**G2 passes at layer 28 with a narrower grid.** Usable alignments are **{1.0, 0.7, 0.4}**; `a = 0.2` is unreachable at this layer, so the grid truncates again by the standing rule and the inside-versus-outside contrast is less extreme than the design assumed. Severity matched in 20/20 draws at 1.4% spread (was 2.4% at layer 21) and the magnitude confound is smaller, 0.406 against 0.517, but not gone.

**The behavioural flooring is fixed, and this was the gating problem.** With the intent decoder the closed loop is well conditioned for the first time:

| arm | baseline | perturbation | washout |
|---|---|---|---|
| no perturbation | 1.00 success, 9 moves | 1.00, 9 | 1.00, 10 |
| `a = 1.0` | 1.00, 9 | 0.80, 17 | 1.00, 9 |
| `a = 0.4` | 1.00, 9 | 0.81, 22 | 1.00, 10 |

**No trial touched the 36-move ceiling**, against 33 to 36 median moves and a censored outcome on 2026-08-10. The moderate-impairment band that §7.3 said the parameters missed on both sides now exists, and it was obtained by fixing the decoder rather than by touching the severity window or the move budget.

**Three things still block a result.**

1. **`L_session` still misbehaves on the control.** No perturbation gives L = 2.34 (paper) and 6.44 (angle) against 0.43 to 0.88 for the perturbed arms, because the metric divides by an impairment that is near zero when nothing was perturbed. Behaviour is now well conditioned, so this is a defect of the metric alone and cannot be blamed on severity.
2. **The arms are not matched on realised impairment**, 2.51 at `a = 0.4` against 1.68 at `a = 1.0`, which is the §4.6 magnitude confound arriving behaviourally. G4's own criterion asks for matched initial impairment, so this fails it.
3. **G3 fails a gate for a new reason: the task is now too easy.** Success is 1.00 in every baseline session and moves run 8 to 10, so the moves IQR is 2 against a required 3. There is no headroom left to *improve*, only to degrade, which is a problem for a paradigm whose outcome measure is re-learning.

**One session in the `a = 0.4` arm was truncated, and the arms are unequal because of it.** Session 003 stopped at trial 37 of 56: the model looped on the two-letter reply `r r`, which is below the three-letter minimum, so the reply was rejected as invalid, no displacement was applied, and after six such turns the stuck detector ended the session (`outputs/v2/g4/sess_003.jsonl`, turn 149). The arm therefore contributes 204 trials against 224 for `a = 1.0`, and the loss falls in the perturbation block of the harder arm. This is a reply-formatting degeneracy rather than a failure of control, but it is an asymmetry between the arms and it compounds the unmatched impairment above. Note also that the abort flags live on the per-turn records and not on the per-trial files, so a check that reads only `sess_*_trials.json` cannot see this and will silently report a short session as a complete one.

**CORRECTION 2026-08-12: the sessions were never severity-matched, so no arm comparison before this date is difficulty-controlled.** G2 searches for a severity-matched triple across alignment levels and reports how well it matched (draws 5 and 9: 31.9/31.9/31.8 and 32.0/32.0/31.9, i.e. matched to 0.1°). The closed-loop runner never used those. `admissible_perturbation` returned the **first** candidate falling anywhere inside the 20-45° severity window, so the realised severity landed wherever the first passing sample fell. Measured on draw 5 under the old code: 25.2°, 39.3°, 31.2° across the three arms, a **14° spread**.

Two consequences. First, the "severity matched 20/20 draws at 1.4% spread" line above describes the *screen's search*, not the sessions, and should not be read as a property of the runs. Second, the unmatched realised impairment reported above (2.51 at `a = 0.4` against 1.68 at `a = 1.0`) is more parsimoniously explained by this than by the §4.6 magnitude confound I attributed it to: the arms were running at different difficulties by construction.

Fixed by collecting every admissible candidate and returning the one closest to a per-draw target severity taken from what G2 matched. Verified on CPU before spending GPU: spread falls to 0.7° (draw 5) and 0.3° (draw 9). The defect was caught by a single-repeat smoke run whose only purpose was to check plumbing, which is an argument for always running one.

**No hypothesis claim is made.** Nominal means are `a = 0.4` L 0.61 against `a = 1.0` L 0.43 (paper), which is the opposite of the prediction, at n = 4 per arm with standard deviations of 0.28 and 0.36 and unmatched impairment. That is not evidence in either direction and should not be reported as a trend.

## 8. Is this worth doing?

### 8.1 What is genuinely new

The closest prior work is **Ji-An, Xiong, Wilson, Mattar & Benna, "Language Models Are Capable of Metacognitive Monitoring and Control of Their Internal Activations" (arXiv:2505.13763, NeurIPS 2025)**. They invented the LLM-neurofeedback ICL paradigm, **cite Sadtler explicitly**, and already report that controllability depends on variance explained (early PCs ≫ late PCs) and on semantic interpretability, concluding LLMs have a "metacognitive space" of much lower dimensionality than the full activation space. **Neither the paradigm nor that conclusion is claimable here**, and our own programme already knew it: `llm-neurofeedback/docs/LITTLEBOY_EXP_DESIGN.md` describes its class-C PCA-band sweep as extending that finding "into the tail," and LITTLEBOY-B found most C-class band readouts null.

What no one has done — verified across the ~40 papers citing 2505.13763, none of which mention BCI, cursor, Sadtler, or decoder perturbation:

1. **A closed-loop, multi-step, behaviourally-scored control task** driven by the model's own activations. Every existing result is a static readout meter. A marker either reaches the target or it does not, which is far harder to dismiss as an artifact of a hand-picked readout — the standing weakness of the whole meter-based line, ours included.
2. **A causal within/outside-manifold perturbation with matched difficulty**, testing *re-learning under perturbation* rather than one-shot controllability of static axes.
3. **Learning dynamics**: recovery curves, after-effects, a graded alignment dose-response.
4. **The purest available test of the re-aiming theory.** Menéndez, Hennig, Golub, Oby, Sadtler, Batista, Chase, Yu & Latham (*eLife* reviewed preprint 106309, 2025) propose that fast BCI learning is re-optimisation within a fixed low-dimensional command space **without weight changes**. In-context learning is literally weight-free learning — arguably the cleanest instantiation of that theory ever testable, and if a LoRA arm cracks the low-alignment end you get the adaptation/skill dichotomy in a system where the manifold and the weights are both measurable.

The "reproduce Sadtler in an artificial network" box is already ticked for gradient-trained RNNs (Wärnberg & Kumar, PLoS CB 2019; Feulner & Clopath, PLoS CB 2021; Feulner et al., Nat Commun 2022; Payeur, Orsborn & Lajoie, bioRxiv 2023) and a human non-invasive replication exists (Nature Neuroscience 2026). So the pitch is **"Sadtler under in-context learning in a pretrained foundation model,"** not "Sadtler in silico."

### 8.2 The main risk: the result may be near-definitional

**Mishra, Khashabi & Liu, "Steered LLM Activations are Non-Surjective" (arXiv:2604.09839)** prove the prompt-reachable activation set is a strict subset of activation space, so off-manifold targets may be unreachable by *any* in-context behaviour. And our own programme found raw random directions in ~5120-D "nearly orthogonal to the data manifold ⇒ dead metric, fail the controllability floor for everyone."

If the low-alignment arm turns out *unreachable* rather than *unlearnable*, the experiment restates non-surjectivity. Three structural mitigations: the **reachability screen** (§4.6) tests before any session runs whether solutions exist; **graded alignment as primary** yields a dose-response informative whatever the endpoints; and a **LoRA arm** turns "impossible" into "impossible *for ICL*, possible for weights" — a dissociation, not a null.

One published result cuts the other way: **Park et al., "ICLR: In-Context Learning of Representations" (arXiv:2501.00070, ICLR 2025)** show that with enough in-context exemplars, concept representations *suddenly reorganise* away from pretrained semantics. So low-alignment learnability may be context-length-dependent — the Oby-et-al.-over-days analog, and a planned extension.

**One honest limit on the screen.** It samples emissions *uniformly*, so it certifies **geometric** reachability: solutions exist among sequences the model could in principle write. It does **not** certify **policy** reachability — that the model's actual emissions visit them — and §8.4 concedes those emissions collapse toward near-optimal paths. The exploration measures (§4.7) are what test policy reachability; the screen must not be described as doing it.

### 8.3 The task-specific risk: a collapsed reachable set

Revision 3 called this "symbol identity dominance" and treated a large symbol-identity component in the residual as a threat. **That framing was wrong.** The monkey's M1 activity is also strongly determined by the movement being made — that is what motor cortex does, nobody calls it a confound, and it is exactly what makes an intuitive decoder work and our baseline sit at ceiling.

The real failure mode is the degenerate limit, and it concerns the **reachable set**. In the monkey, activity is ~90-dimensional and the decoder reads a 2-D slice, leaving ~88 dimensions in which an off-manifold perturbation can ask the animal to move. Our failure case is where the activation at a move token is a *pure function of which of four symbols it is*: then four distinct activations exist, the manifold estimated from them is everything reachable, nothing lies outside it, and every perturbation collapses to relabelling four symbols (4! = 24, learnable in a handful of trials). That would be the bandit problem at the per-move level.

That limit is not the actual situation, because the residual also depends on the whole context, so the same `r` in a thousand contexts gives a thousand activations. The question is quantitative, and **the criterion is: holding the symbol fixed, does the activation vary across contexts enough that off-manifold control targets are reachable?** That is answered *directly* by the reachability screen, which is why **G2 is the kill gate and G1 only a fail-fast screen** — conflating the cheap variance decomposition with the real criterion was the error.

A consequence worth noting: if within-symbol variation *is* substantial, writing `r` does not produce a fixed velocity but a context-dependent one, so the model cannot recover by learning "`r` now means up" — it must find which emissions produce the movement it wants in the situation it is in. That is the interesting version of the task, and the same thing the monkey faces.

### 8.4 "Won't" versus "can't"

With the target coordinates in the prompt the correct move sequence is arithmetic, so the model can succeed without ever consulting the observed movement. Two consequences, really the same one twice: the emission distribution actually visited collapses onto near-optimal paths, so certified-reachable solutions go unvisited for reasons of search rather than geometry; and under perturbation the model may not explore at all, giving zero measured learning in *every* arm.

This is the most likely route to an uninterpretable null. Addressed by the exploration prior in the system prompt (§3.3), by exploration breadth as a first-class logged measure (§4.7), and by G4, whose pass criterion requires nonzero learning at `a = 1` — and if that fails, the first hypothesis is absence of exploration, not absence of capability.

**Pre-registered fallback:** withhold the target coordinates and give only the scalar distance, making attention to feedback mandatory. Held as a fallback rather than the default because it adds a per-trial target-localisation problem confounded with the mapping learning we want, and would pull baseline off ceiling. If invoked, targets carry a stable index 1–8 so localisation is learned once, and baseline is lengthened until success returns to ceiling.

### 8.5 Other risks

- **Feasibility concentrates in the baseline block.** If the model cannot drive the marker with the intuitive decoder, nothing downstream matters. Fitting to the emitted move makes this likely to pass, but G3 gates it.
- **Analogy strain.** With reasoning off and bare symbol replies there is no chain of thought and improvement across trials is genuinely accumulated in context — closer to trial-by-trial adaptation than a CoT version. But the learning is mediated by a readable transcript, not implicit plasticity, and the writeup must say so.
- **Greedy failure modes.** A session is determined by its target sequence *given a fixed batch shape*; since sessions run in lockstep batches of 16 and batch shape varies as sessions finish, runs are **not** bit-reproducible. Treat the target sequence as the session's identifier, not as a reproducibility guarantee. Two behavioural hazards: the repetition degeneracy already measured in LITTLEBOY, and a fixed point where the marker stops moving, so the prompt stops changing, so the greedy reply stops changing. A stuck-detector can fail a session; duplicate-reply rate is logged.
- **Effect-size worry from our own data.** LITTLEBOY-B measured state-channel control at ~1 calibrated point with the text *fixed*. Not this regime — the emission is free here — but it is why baseline control is gated empirically.

### 8.6 Verdict

Worth doing. As scoped — reachability screen, graded alignment primary, behavioural scoring, angular-error resolution — roughly **7/10 on interest, ~8.5/10 if the LoRA arm lands, ~4/10 if run naively as "off-manifold is unlearnable"** (near-definitional and already implied). Write it as the **causal and dynamical completion of 2505.13763**, not as a first.

---

## 9. Declared deviations from Sadtler et al.

1. **Ambient space** is the full residual stream at one layer with `N ≫ D` samples, not a fixed small population. Their `q` was a hardware limit; ours is a sample-size condition met by more samples, not fewer coordinates. Massive-activation dims winsorized, none dropped.
2. **Manifold defined as `rowspace(β)`.** A repair, not a preference — the paper's `colspace(Λ)` makes its own "the intuitive mapping lies within the manifold" false unless `Ψ ∝ I` (§4.3).
3. **`d = 10` fixed a priori** — same as the paper, but we explicitly do *not* adopt peak-picking from the retrospective sweep, because that estimate is sample-size dependent (§1.10).
4. **Decoder fit to the emitted move**, not target-directed intent — necessary because the target sits in the prompt (§4.4).
5. **Read at the move symbol's own token**, concurrent with the movement, as the paper decodes bin `t` from bin `t`. A pre-move read was specified in revision 3 and rejected (§4.2).
6. **Calibration by teacher-forcing target-reaching shortest paths in a byte-identical template**, symbol order randomised, with random paths held out as the leak control (§4.5).
7. **Perturbations by rotation at controlled alignment**, not permutation, with **θ screened per level on measured severity** rather than fixed (§4.6) — which is closer to the paper, since Sadtler screened into a severity window and only reported the angles afterwards. The `[20°, 45°]` open-loop window is transplanted from the paper's *session-specific* windows as a fixed window — and means something stronger for us: there it bounded a small correction term on an integrating cursor, whereas with `M1 = 0` the decoded step is the entire motion, so the same angular deviation is more disruptive per step. Matching every level at one θ is also stronger than the paper's distribution-level match.
8. **No temporal filter (`M1 = 0`)**; turn-based discrete moves instead of 45 ms bins.
9. **Move budget ≈ 4× the median solution** against the paper's ~7.5×; provisional, frozen at G3 (§3.1).
10. **No announced detach and no mid-session messages**; the session opens under decoder control at trial 1 and every change is silent. Where scaffolding is needed the fallback is the paper's own — monkey L's unannounced assistance ramp — with the literal-execution weight faded 1 → 0.5 → 0.125 → 0 over ~24 trials and the decoder held **frozen** rather than refit at each step. Both the schedule and the horizon are ours, not the paper's, whose ramp ran over 80 closed-loop trials with a refit per step.
11. **One manifold and decoder shared across all sessions** (fixed weights, no recording drift), rather than re-estimated daily.
12. **Reachability screen** added, and it goes beyond the paper's implicit guarantee: its z-scoring existed to keep perturbed mappings inside observed firing ranges, which a permutation preserves and a rotation does not, so extrapolation is screened explicitly (§4.6).
12b. **Severity screened per sample, not on mean activity** (§4.6, measured 2026-08-10). The paper's mean-based open-loop probe reports a low-alignment perturbation as ~3° — nearly harmless — because averaging destroys the private variance such a control space reads. Ours is the median per-sample deviation.
12c. **Per-direction reachability tolerance 60°, plus an explicit plane-spanning criterion** (§4.6). At 30° the intuitive decoder fails its own screen, because with a 4-symbol alphabet a single move is 45° from any diagonal and net direction comes from mixing moves — a structural feature of a discrete action space that the paper's continuous velocity control did not have.
13. **Per-move angular error** added as a co-primary metric — available because we know both the target and the emitted move, which the original did not.
14. **Signed learning metric**, correcting what must be a typo in the published eq (19) (§1.8). Trial counts, bin sizes and block lengths shrunk and calibrated from pilot data (§7) rather than inherited.
15. **No injected decoder noise** (considered and rejected: it slows credit assignment and inflates metric variance; the model already cannot predict its own activations, which supplies the natural variability analog).

---

## 10. Code layout

New project at `MATS/neural-contraints-learn/` (directory name has a typo inherited from creation; kept deliberately rather than renamed silently). Sibling of `MATS/llm-neurofeedback` on the same filesystem. **Reuse, do not rebuild**: `VLLMLensBackend` (`generate_batch`, `read_residual_batch` / `read_residual_multi_batch` for the multi-depth sweep), `JsonlLogger`, and `producers/` for the observer arm, imported via a `sys.path` entry.

**Environment: `MATS/venv-vllm` only** — `MATS/venv` has a broken scipy install. `venv-vllm` is uv-managed with no `pip`; install with `VIRTUAL_ENV=…/venv-vllm uv pip install …`.

```
neural-contraints-learn/
  docs/DESIGN.md              # this file
  pytest.ini
  nc/task.py          DONE    # plane, 8 targets, alphabet, budget, prompts, reply validity, shortest_path
  nc/manifold.py      DONE    # frozen constants, FA, β/S_z, ridge decoder, gain, rotation at target
                              # alignment, reachability density, principal angles, required-PD change, cv dimensionality
  nc/metrics.py       DONE    # binning, z-scoring, signed L_session on both metric pairs, tail-mean, after-effect
  tests/              DONE    # 76 tests, no GPU
  nc/session.py       (~150)  # block loop over B sessions in lockstep: generate_batch → span read → per-symbol decode → apply → log
  scripts/calibrate.py (~80)  # G1: forced target-reaching reads at 4 depths, all G1 diagnostics, FA, decoder, artifact npz
  scripts/screen.py    (~60)  # G2: candidate construction + reachability + extrapolation check, frozen perturbation pool
  scripts/run.py       (~70)  # G3/G4/full: n sessions at a given alignment level, JsonlLogger per session
  scripts/analyze.py   (~70)  # alignment regression, endpoint contrast, matching checks, exploration measures, figures
```

`nc/` has no GPU dependency and no dependency on `llm-neurofeedback`; both arrive with `nc/session.py`. Calibration artifacts freeze to one `.npz` (winsorization bounds, z-score constants, Λ, Ψ, β, S_z, W, gain, per-direction `u_B^i`, and the screened perturbation pool with recorded properties). Nothing is refit at run time.

---

## Appendix A — revision history

- **R1** — first design: 90 selected residual coordinates, free-text replies alongside a numeric move, decoder fit to target-directed intent, permutation perturbations, PCA pre-reduction floated.
- **R2** — full residual stream (no subsetting; PCA dropped after benchmarking), symbol-sequence task with no free text, decoder fit to the emitted move, perturbations by rotation.
- **R3** — dropped the announced controller-detach phase and all mid-session messages. A first draft of that section wrongly claimed the paper had no assistance to remove; corrected the same day after re-reading Methods — no *limb* controller, but monkey L's calibration is an unannounced autopilot fade, now our G3 fallback.
- **R4** — calibration made byte-identical to play (plain teacher-forcing of target-reaching shortest paths, random paths demoted to a held-out control); read moved to the move symbol's own token; the collapsed-reachable-set risk restated after R3 mis-framed it as "symbol identity dominance".
- **R5** — full correction pass after a cold-reader review and a line-by-line fact-check against the paper. Errors fixed: the alignment/angle bound is one-sided, so the claim that Sadtler's OM perturbations had `a ≈ 0.03–0.6` was wrong and is replaced by the stronger correct reading (§2.2); "magnitude-matched by construction" was false for the WM arm (§1.4); the required-PD refit is over 8 direction bins, not 4 symbols; the published learning metric is unsigned and must be a typo (§1.8); the reachability screen's speed floor was self-referential (fixed in code with a test); the gain was circularly defined. Decisions frozen: `d = 10` a priori, θ = 60°, alignment grid (1.0, 0.75, 0.50, 0.25), `RHO_MIN = 0.01`, move budget 4× median. Added: an analysis plan (§5), quantified gate criteria, the geometry numbers, per-direction `u_B^i` binning, the extrapolation check, and the paper's own caveats on after-effects, session-exclusion imbalance and dimensionality-versus-sample-size. Restructured: thesis and gate roster first, the three-spaces section moved early, exploration measures moved into §4.7, redundant arguments deduplicated, changelog moved here.

- **R6** — post-run correction. **One G1/G2 pass was run on Qwen3.5-9B and all of its numbers were discarded**: the decoder had been fit on state-*j* → letter-*j*, a readout of the letter already present rather than of the intent to move, so the decodability figures, depth profile, layer choice, artifact and every G2 per-level number described the wrong readout. Three findings survived because they do not depend on the labels: the forced target-reaching replies carry probability **0.90** under the model versus **0.16** for random ones, confirming target-reaching calibration solved the off-policy problem; at the alignment floor the perturbation **degenerates** into an attenuated copy of the original decoder (in-manifold part = `cos θ` × the original direction), measured as 3.0° of velocity change and 0/200 admissible candidates at `a = 0.25`; and consequently **severity co-varies with alignment**, so a fixed θ cannot deliver matched difficulty. Design changes: read label → state-*j* → letter-*j+1*; random target angle at fixed radius; ambient space → 100 randomly drawn coordinates resampled per session, with massive-activation dims excluded from eligibility; θ screened jointly with the read layer instead of fixed; alignment grid extended to (1.0, 0.7, 0.4, 0.2, 0.1); 8 read depths; 1,200 + 600 calibration turns; random paths demoted to emission-sample-only, with the ambiguous-vs-unambiguous contrast replacing them as the intent-vs-target validation. Calibration cost measured (§7.2).

- **R7** — removed three invented gate cut-offs and then withdrew one of the two measures entirely. G1 had gated on cv-R² ≥ 0.30, within-bin direction variance ≥ 0.20 and per-direction spread ≤ 15°. I picked all three numbers by hand and none came from a null distribution, so they were quality bars in name only, and the figures and §7.3 presented them as established criteria. Removing them exposed that the pair had also been selecting the operating layer, which led to examining the within-bin measure itself. It does not survive: it cannot separate a target detector from a faithful readout of a deterministic policy, a least-squares decoder is *supposed* to collapse to the bin mean where the label is unpredictable, and 71.8% of our labels are RNG-chosen ties, so the measure mostly tracks label noise (§4.5). It is deleted from the code, the figures and the claims, and the depth trade-off that was the run's headline is retracted as near-definitional rather than discovered (§7.3). What remains: one definitional gate (R² > 0) read against a ~0.58 label-noise ceiling, and §4.1's layer rule reduced to "max R² subject to admitting matched candidates", with a `--layer` override for when the operating layer is a decision. Two things are now open rather than settled: the §4.2 target-detector risk is unmeasured, and the current rule does not select layer 21, where the existing sessions ran. No measured number changed.

## Appendix B — internal project names

- **`llm-neurofeedback`** — the sibling repository, `MATS/llm-neurofeedback`; source of the backends, logger and producers reused here.
- **LITTLEBOY / LITTLEBOY-B** — prior experiments in that repo on in-context control of activation readouts with the response text held fixed. B measured a small directional state-channel effect in Qwen3.5 and null elsewhere; its class-C readouts were PCA-band directions, mostly null.
- **REVOLVE, BOND, LUPIN, FATMAN** — other experiment lines in that repo. "Fat Man" is the GPU-budget campaign those runs draw on.
- **`battery_<tok>.npz`** — that repo's convention of freezing all fitted readout constants into one provenance-stamped artifact, mirrored here for the calibration artifact.
- **li2026** — how that repo's docs cite Ji-An et al. (arXiv:2505.13763).
