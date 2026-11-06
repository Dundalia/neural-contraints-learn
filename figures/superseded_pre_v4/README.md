# Superseded figures, generated before 2026-08-12

Every plot in this folder was made under the OLD read convention, where a k-letter reply produced
only k-1 applied moves and the model's first move was silently discarded (DESIGN 7.2d). The closed
loop therefore executed a truncated version of every plan the model wrote.

Do not cite these. The clearest reason: they show a strong progressive degradation across the
perturbation block (moves slope +0.199 per trial, t = +6.35 at alignment 1.0) which the corrected
run reduces to +0.007 (t = +0.24). That degradation was mostly the bug.

Provenance of what is here:
- `1_readout`, `2_label`, `3_constants` came from `artifacts/v2` (intent label, old read convention).
- `4_blocks`, `5_curves`, `6_metric` came from `outputs/v2` (8 sessions).
- `7_perturbation`, `8_aim` came from `outputs/v3` (96 sessions, old convention).
- `trajectories_template_check` was synthetic, used only to check the trajectory layout.

The current figures are regenerated from `artifacts/v4` and `outputs/v4`.
