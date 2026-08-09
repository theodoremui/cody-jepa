# GFC-v2 adversarial review record

Date: 2026-08-05

The installed Codex CLI (0.146) rejects a custom prompt together with `review --base`, so
the equivalent `codex exec review --uncommitted` review was run over the complete change.
Its scientific-validity, leakage, inference, privacy, and failure-behavior findings were
resolved as follows:

- Fixed: training now threads declared study metadata into every checkpoint payload.
- Fixed: soft completion ranks log probabilities, avoiding underflow-created ties.
- Fixed: soft-control ranking scales its absolute tie tolerance by inverse temperature,
  preserving calibration-invariant ranks even for near-tied factor scores.
- Fixed: all ladders must declare one common full-data sequence count.
- Fixed: registry model labels must be safe single path components.
- Fixed: registry checkpoint and feature paths must resolve to distinct files, preventing
  lexical aliases or symlinks from reusing one artifact for multiple models.
- Fixed: inference and summarization require all 308 complete outcome participants, not
  merely identical participant subsets across runs.
- Fixed: summarization compares every non-path registry field with the evaluated model
  metadata, preventing a revised registry from relabelling completed outputs.
- Fixed: summarization requires identical frozen method settings across every analysis
  and model, preventing mixed tolerances, numerical floors, or bootstrap settings.
- Fixed: the aggregate-only renderer rejects singular and plural participant fields.
- Fixed: renderer count metadata uses exact integer validation rather than truncating
  malformed numeric values.
- Fixed in part: an optional feature sidecar, when present, must name and agree with the
  registry checkpoint.
- Not applied: making feature sidecars mandatory. The frozen contract requires agreement
  when sidecar data exist and permits legacy archives without sidecars.
- Not applied: adding a role-map checksum. The frozen contract deliberately exposes only
  the role-map version and aggregate counts; cohort checksums are explicitly out of scope.
  The stale checksum placeholder in `docs/unique-sequence-scaling/method.md` was corrected to the frozen version.
- Not applied: adding pool-membership checksums to the private registry. Its exact frozen
  schema has ten fields and artifact hashes are explicitly out of scope; pool construction
  is an upstream training-data responsibility, while this evaluator records actual sizes,
  seeds, exposure, and final-step checkpoint provenance.
- Not applied: expanding lightweight checkpoint metadata into a cross-run training-config
  auditor. The supplied freeze contract defines the eligible checkpoint metadata fields
  exactly; training the twenty ladder models and broadening that provenance schema are out
  of this evaluator implementation's scope.
- Not applied: notebook-link edits. Those relocations are unrelated user work and are not
  part of the evaluator change.

No locked-outcome values were opened during review or verification.
