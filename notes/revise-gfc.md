A previous agent produced the plan below to accomplish the user's task. Implement the plan in a fresh
  context. Treat the plan as the source of user intent, re-read files as needed, and carry the work
  through implementation and verification.

  # Rebuild the Active Evaluator as GFC-v2

  ## Summary

  Replace the active two-block, 24-query, donor-excluded evaluator with the existing oracle-defined
  GFC-v2 protocol: three factor blocks, 16 session-safe queries, and all eight gallery cells. Preserve
  checked-in legacy results only as historical artifacts.

  Use two parallel read-only/module-isolated work lanes initially:

  - Lane A: core protocol and scoring.
  - Lane B: configuration and factor adapters.
  - Main lane: integrate runner, inference, outputs, and tests after A/B stabilize.
  - Final lane: independent adversarial review and repair loop.

  ## Implementation Changes

  ### Protocol and scoring

  - Make `compile_healthgait_gfc_v2_protocol()` the sole authority for canonical cells, focal factors,
  donor roles, query order, gallery contents, and derived counts.
  - Add immutable `FactorBlocks(speed, clothing, direction)` and require `Recording` to carry
  `source_video_id`.
  - Replace legacy query fields with `focal_factor`, `donor_u`, `donor_v`, compiler-derived factor
  sources, and the full ordered gallery.
  - Compose query blocks only from donors. Fail closed if either donor shares the target’s
  `source_video_id`.
  - Score the equal mean of three cosine distances. Retain existing fractional top-1 and average-rank
  MRR behavior.
  - Replace inverted `donor_attraction` with separate `donor_u_attraction` and `donor_v_attraction`:
  donor wins `1`, tie `0.5`, target wins `0`.

  ### Configuration, adapters, and runner

  - Replace `query` and `gallery` configuration sections with:

  ```json
  "protocol": {
    "name": "gfc_v2",
    "donor_rule": "binary_complementary_two_donor",
    "focal_factors": ["speed", "clothing"],
    "gallery_policy": "retain_all",
    "require_target_source_independence": true
  }
  ```

  - Derive 8 targets, 16 queries, and gallery size 8 from the compiler. Reject redundant count fields
  and all legacy protocol settings.
  - Replace condition/gait adapter APIs with `fit_factor_adapter(..., factor_name=...)` for speed,
  clothing, and direction.
  - Fit six independent ridge heads: learned/shortcut × three factors. Every shortcut head receives the
  same canonical nine-cue vector.
  - Replace two-block distance weights with `factor_aggregation: "equal_mean"`.
  - Pass source lineage into production recordings and independently enforce its invariants in core
  scoring.
  - Change the one-query power effect to `1/16`.

  ### Outputs and compatibility

  - Emit `protocol=gfc_v2`, `gallery=retain_all_8`, `queries_per_participant=16`,
  `factor_heads=three_matched_ridge_heads`, and source-independence verification.
  - Include protocol, focal factor, donor roles, factor sources, full gallery, and source-safety status
  in scientific pairing keys.
  - Publish donor-u/v roles and per-factor distances without exposing raw subject, recording, or source
  identifiers.
  - Keep tracked legacy summaries unchanged. Make the paper-results renderer validate against a
  renderer-only `LEGACY_GFC_PROTOCOL`; reject v2 or mixed summaries from legacy tables.
  - Update README language so the maintained evaluator is v2 while checked-in outcomes remain
  explicitly legacy.

  ## Test Plan

  - Assert exact equivalence between production queries and every compiled oracle query: 16 target/
  focal pairs, complementary donors, full galleries, and deterministic ordering.
  - Verify donor-u and donor-v source collisions fail separately; valid paired-direction source lineage
  remains accepted.
  - Reproduce the production top-1/MRR spectrum for every recovered-factor subset: `1/8, 1/4, 1/2, 1`
  and `2/9, 2/5, 2/3, 1`.
  - Prove target blocks never enter query composition and both donors remain scored candidates.
  - Test donor-u/v win, loss, and tie attraction semantics.
  - Cover all three factor adapters, complete-grid validation, deterministic fitting, six matched
  heads, and nine-dimensional shortcut inputs.
  - Verify held-out rows cannot affect adapters, normalizers, query keys, or development outputs.
  - Assert 16-query inference, eight-entry query galleries, strict v2 metadata, and output privacy.
  - Confirm legacy fixtures still regenerate legacy tables and v2 summaries cannot enter them.
  - Run targeted GFC tests, research pipeline smoke tests, all unit tests, Ruff, `git diff --check`,
  and temporary legacy-result regeneration.

  ## Adversarial Review and Assumptions

  - After tests pass, run a fresh defect-first Codex subagent and `codex review --uncommitted`. Fix
  every verified finding, rerun affected and full tests, then repeat both reviews until clean.
  - The adversarial prompt must challenge target leakage, missing source checks, accidental direction
  focality, donor removal, unequal shortcut heads, held-out leakage, inverted attraction, protocol-
  identity conflation, private-ID leakage, and misleading documentation.
  - The exact `codex:adversarial-review` skill is unavailable; the user selected the dual-Codex review
  fallback.
  - No active legacy evaluator remains.
  - Hard/soft completion controls, alpha sensitivities, cohort-role checksums, and outcome unblinding
  are deferred and remain mandatory before protocol freeze.