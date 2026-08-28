# Step 4 — QA Templates (KO / EN)

Every natural-language surface form of the generated QA lives in
`settings/templates_<lang>.yaml` (user-editable text with {placeholders});
`modules/qa_templates` is only the engine that loads them, separated
from the data plumbing in `scripts`. Facts (numbers, muscle
identities, directions) are rendered verbatim into fixed template slots, which
is what makes the gold answers deterministic and record-verifiable, and what the
faithfulness check later protects during naturalization.

Both modules expose one interface (see `templates_ko.py` docstring):

- **physics chain** (`scenario=physics_chain`):
  `t_attribution` (A2 muscle attribution) · `t_identifiability` (D1
  motor-equivalence abstention) · `t_volume` (C1 incompressibility) ·
  `t_single` (B1 single-muscle intervention) · `t_counterfactual` (B2 neighbor
  delta vs base mesh, with a "negligible effect" branch under 0.1 mm) ·
  `t_prescriptive` (target-directed correction toward a Step-1 anchor).
- **feature QA**: `t_a1` (3-turn quantitative shape description),
  `q_b3_single` / `q_b3_effort` / `a_b3` (dose-response with
  strict / near / partial monotonicity tiers + Spearman rho).
- **span tagging**: `physics_mask_spans` / `feature_mask_spans`, built from the
  shared factory in `modules/spans.py` with per-language keyword lists.

Adding a language = adding one `settings/templates_<lang>.yaml`; it is picked up
automatically (BY_LANG) and the Step-5 generators accept it via `--lang`.

Surface-form contract: full muscle names (`modules.muscles.FULL_KO/FULL_EN`)
and all numeric values must appear exactly as rendered — downstream checks and
metrics match on these strings.
