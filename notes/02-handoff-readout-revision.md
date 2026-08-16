# Handoff: restructure Proposal 1 around the variance decomposition

Paste everything below the line into a fresh agent session with working directory
`/Users/theodoremui/dev/cody-jepa`.

---

**Role.** You are an expert AI research scientist and technical writer working on a
self-supervised learning paper targeting ICLR 2027 (abstract due 2026-09-18, paper due
2026-09-25, both Anywhere on Earth). Today is 2026-08-14.

**Working directory.** `/Users/theodoremui/dev/cody-jepa`. Python is `./.venv/bin/python`
(do not use bare `python`, pyenv is misconfigured for this repo).

## What already exists

A previous session synthesised three planning documents (`claude-iclr-analysis.md`,
`claude-iclr-ideas.md`, `codex-iclr-ideas.md`) into a `tutorials/` folder containing:

- `tutorials/README.md`: index
- `tutorials/00-overview-and-evaluation.md`: evaluation of all four proposals
- `tutorials/01-readout-problem.md`: **the file you will restructure**
- `tutorials/02-paired-condition-geometry.md`
- `tutorials/03-minimum-sufficient-state.md`
- `tutorials/04-personal-baseline.md`
- `tutorials/images/*.svg`: eleven hand-authored diagrams

**Read first, in this order:**

1. `notes/derived-findings-2026-08-14.md`: thirteen verified findings (F1 to F13) computed
   from this repository's own artifacts. This is your evidence base. Every number in it was
   recomputed from disk, not quoted from the planning documents. Do not re-derive these; they
   are correct and the scripts that produced the two expensive ones are in `notes/scripts/`.
2. `tutorials/01-readout-problem.md`: the current proposal.
3. `tutorials/00-overview-and-evaluation.md`: for the house style, the ICLR acceptance
   pattern, and the scoring table you will need to update.

## Your task

Restructure `tutorials/01-readout-problem.md` so that its novelty rises from roughly 5.5/10
to 7.5-8/10, on the assumption that **8 H100 GPUs are available** for the run-up to the
deadline. The previous version was written assuming modest compute and therefore avoided
anything requiring a large number of training runs. That constraint is gone.

Also update the corresponding rows and paragraphs in
`tutorials/00-overview-and-evaluation.md` so the two documents stay consistent.

### The new spine of the paper

The current paper says "two readouts disagree, here is a measurement contract." That is a
checklist, and checklists cap at borderline. Replace it with a derived quantity, a mechanism,
and a head-to-head win against a published metric.

**Spine 1: the decomposition (see F13).** For tokens x_{i,t} with equal tokens per clip, the
law of total covariance gives Sigma_token = Sigma_between + Sigma_within, and Sigma_pooled =
Sigma_between exactly. So the token-axis health metric is an entropy over the eigenvalues of
a sum, only one term of which a pooled probe can ever consume. Define the between-clip share
beta = tr(Sigma_between) / tr(Sigma_token). Measured values are in F13. The headline numbers:
beta is 0.00063 for the unregularised trained model, meaning 99.94 percent of token-axis
variance is invisible to any pooled probe; erank(Sigma_token) equals erank(Sigma_within) to
two significant figures (60.32 versus 60.23); and beta spans a factor of 400 across models on
identical data with an identical pooling operator, which refutes the "it is just pooling"
objection quantitatively.

**Spine 2: the mechanism, which should be primary.** Hypothesis: in a masked *predictive*
architecture the predictor is conditioned on target position, which creates pressure for
token representations to carry position-specific information. That inflates Sigma_within and
contributes nothing to Sigma_between. Contrastive and siamese architectures have no such
pressure. Write this as a set of falsifiable predictions that are tested by manipulation, not
by correlation: beta should track predictor positional reliance across absolute versus
relative versus no positional conditioning, spatial-only versus volumetric masking, mask
ratio, and predictor depth; and beta should be structurally higher in contrastive models
trained on identical data. Make this the primary claim so the paper survives even if Spine 3
fails.

**Spine 3: beat RankMe on a controlled model population.** RankMe (ICML 2023) proposed
effective rank as a label-free transfer predictor and validated it on contrastive and siamese
image encoders. It appears never to have been systematically validated on masked predictive
architectures. Train roughly 120 configurations times 3 seeds, show that the standardly
computed rank fails to predict transfer for this architecture class while beta succeeds. The
template to copy is *Rethinking the Uniformity Metric in Self-Supervised Learning* (ICLR
2024, accepted): critique a metric, propose a corrected one, win the head-to-head. Emphasise
that a *designed* model population is a better instrument than the heterogeneous public
checkpoints most analysis papers are forced to use.

**Spine 4: the consequence people act on (see F12).** Pretraining loss does not predict
transfer. Across the eleven existing runs, clip-pooled effective rank correlates with
held-out retrieval at Spearman 0.890 (p = 0.0002), best validation loss at 0.306 (p = 0.360),
and best training loss at 0.187 (p = 0.582). This repository, like most JEPA training,
selects `best_loss.pt`. That criterion is uninformative about representation quality. Frame
this as checkpoint selection being broken, and as something the corrected metric fixes.

**Spine 5: two things only large N can find.** A scale ladder (6, 12, 24 layers by 384, 768
dim) that answers the "single small model confounded with capacity" objection, which was
previously unanswerable and is risk number three on the withdrawn-paper list. And a search
for non-monotonicity: RankMe assumes the rank-to-transfer relationship is monotone, and an
interior optimum found across several hundred models would be a genuine discovery
contradicting a published assumption.

### The prior-art risk you must handle in the text

**LiDAR (arXiv 2312.04000) is the nearest neighbour and the strongest attack on Spine 1.** It
uses a discriminant-style within-versus-between ratio where "within" is augmentation-induced.
That is structurally analogous to beta. Engage it in the first paragraph of the related work,
not in a citation dump. The defensible distinction is that LiDAR's within-class scatter comes
from augmentations of a single sample, whereas beta's comes from token position inside a
single clip, and for masked predictive architectures those are different objects with
different causes. If that distinction cannot be stated crisply, the novelty gain evaporates.

**Also drop any claim that the shortcut audit is novel.** Conditional probing (Hewitt et al.,
EMNLP 2021) and V-information (Xu et al., ICLR 2020) already formalise "probe accuracy above
a baseline." Cite both, and reframe the stopwatch result as usable-information accounting
applied to video SSL evaluation, where it is not done. This converts a vulnerability into
rigour. Keep the stopwatch as the motivating example, not as a claimed contribution.

Other neighbours already checked and worth keeping in related work: RankMe (arXiv
2210.02885), LDReg (ICLR 2024), Jing et al. dimensional collapse (ICLR 2022), and recent work
on projection-head geometry (arXiv 2605.17180).

### Feasibility section to write

Compute stops being the binding constraint. The model is roughly 10M encoder parameters at
1,568 tokens per clip; a 3,900-step run is on the order of one GPU-hour on an H100. Eight
cards over about 25 usable days is roughly 4,800 GPU-hours, so a 360-run main population
costs about 2.5 days of wall clock, leaving room for the scale ladder, a contrastive control
family, a second domain, and a 2x rerun margin.

State clearly that **the bottleneck moves to data loading**. JPEG decode of silhouettes will
leave the run IO-bound at low GPU utilisation. Day one should be spent precomputing a packed
uint8 tensor cache; the 112 by 112 by 16 grayscale corpus is small. Say that this single task
matters more than any hyperparameter choice.

Name the two risks the compute introduces: scope creep, because the freeze date does not
move; and large N making p-values trivial, so effect sizes and pre-registration matter more
rather than less.

### Keep from the current version

The existing evidence sections are correct and should survive, possibly reorganised: the
random-init control (F10), the rank-versus-retrieval correlation (F1), the stopwatch baseline
(F8), the duration-matched collapse (F11), and the honest statement that the duration-matched
probe does not flip the rank correlation and is underpowered at 39 participants. The Gate
structure and the "what kills this paper" section should stay, updated for the new spine.

### New diagrams to produce

Add to `tutorials/images/`, and reference them from the restructured document:

1. **`variance-decomposition.svg`**: the central new figure. Show Sigma_token splitting into
   Sigma_between and Sigma_within, with the pooled probe reading only Sigma_between, and the
   measured beta values from F13 as a small bar or ladder (random init 0.00025, no regulariser
   0.00063, clip-axis variance on 0.104). This figure carries the paper.
2. **`selection-criterion.svg`**: three correlation values from F12 as a simple comparison,
   showing that loss does not predict transfer while pooled rank does.
3. Optionally a **mechanism** figure showing predictor positional conditioning inflating
   within-clip variance.

Revise `tutorials/images/readout-problem.svg` if the new spine makes its framing stale.

### Diagram conventions, which are strict

- Hand-authored SVG only. No external dependencies, no scripts, no libraries.
- Maximum width **750 units**. Content beyond about x = 760 is clipped by the render check.
- Font stack: `system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif`.
- Explicit white panel: `<rect x="0.5" y="0.5" width="749" height="H" rx="10" fill="#ffffff"
  stroke="#e2e8f0"/>`. Never rely on a transparent background, because these are read on
  GitHub in both light and dark themes.
- Palette: text `#0f172a` and `#334155`, muted `#64748b` and `#94a3b8`, rules `#e2e8f0` and
  `#f1f5f9`, blue `#2563eb`, amber `#d97706`, red `#dc2626`, green `#059669`, violet `#7c3aed`.
  Tinted panels: `#eff6ff`, `#fffbeb`, `#fef2f2`, `#f0fdf4`, `#faf5ff`, `#f8fafc`.
- Title 16.5px weight 600, subtitle 11px muted, body labels 10.5 to 12.5px.
- Use HTML entities for apostrophes and ampersands (`&#8217;`, `&#38;`).
- **No em-dashes or en-dashes anywhere**, in SVG or markdown.

### Mandatory visual verification loop

You must look at every diagram you produce. Do not ship an unverified SVG.

```
cd /Users/theodoremui/dev/cody-jepa/tutorials/images
qlmanage -t -s 1200 -o /tmp/svgcheck yourfile.svg
```

Then use the Read tool on `/tmp/svgcheck/yourfile.svg.png` to view it. Check specifically
for: text overlapping lines or curves, text overflowing its containing box, connector lines
crossing each other, labels colliding with data points, and content clipped at the right
edge. Fix and re-render until clean. Note that `qlmanage` produces a square canvas and crops
anything past roughly x = 769, which is why the 750-unit width limit exists.

### Writing style, which is also strict

- **No em-dashes or en-dashes.** Use commas, full stops, or restructure the sentence.
- Natural, direct, accessible prose. Short sentences. No jargon where a plain word works.
- Connect ideas across sections rather than presenting them as a list of disconnected points.
- State limitations plainly and in the main text, not buried. The existing documents do this
  well; match that register.
- Every number must carry its provenance and its caveat. For example, the eleven-run
  correlation is single-seed per configuration across runs that were never designed as a
  controlled comparison, and that must be said wherever it is used.
- Do not overclaim. "JEPAs fail" is an overclaim and is easy to refute. The random-init
  control shows the numbers are model-dependent, not universal.

### Verification before you finish

```
cd /Users/theodoremui/dev/cody-jepa/tutorials
grep -n '—\|–' *.md images/*.svg        # must return nothing
grep -oh 'images/[a-z-]*\.svg' *.md | sort -u   # every ref must exist in images/
for f in images/*.svg; do ./../.venv/bin/python -c "import xml.dom.minidom;xml.dom.minidom.parse('$f')"; done
```

Then re-render and visually inspect every diagram you created or modified.

### Scripts you can reuse

- `notes/scripts/variance_decomp.py`: produced F13. Run as
  `./.venv/bin/python -W ignore notes/scripts/variance_decomp.py label1 path/to/ckpt1.pt label2 path/to/ckpt2.pt`.
  Takes several minutes on CPU or MPS for 1,872 clips.
- `notes/scripts/random_init_rank.py`: produced F10. Same invocation pattern with a single
  checkpoint path.

Useful data locations: `outputs/phase1/*/features.npz` (pooled features plus row metadata),
`outputs/phase1/*/probes.csv`, `outputs/phase1/*/best_loss.pt` (contains a `history` list with
per-epoch `train_loss` and a nested `val` dict), and
`data/healthgait/manifests/silhouette_gfc_candidate_seed0.csv` (has `shortcut_*` columns,
`fps`, `speed`, `clothing`, `direction`, `split`).

### Scope boundaries

Restructure `tutorials/01-readout-problem.md` and update the affected parts of
`tutorials/00-overview-and-evaluation.md`, specifically the Proposal 1 scoring row, the
Proposal 1 assessment paragraphs, and the recommendation section. Do not rewrite proposals 2,
3, or 4 unless a cross-reference becomes wrong. Do not touch the three original planning
documents at the repository root; they are the historical record. Do not commit anything
unless explicitly asked.
