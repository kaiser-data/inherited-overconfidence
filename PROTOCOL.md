# Protocol: the return map of stated confidence under train-on-model-output

Frozen 2026-11-14 14:30 CET, before any model output exists. Changes after this point go in
the "Deviations" section at the end with a timestamp and a reason.

Apart Research AI x Epistemics Sprint (Nov 13-15, 2026). Author: Martin Kaiser.

## Question

When a fresh model is trained on another model's question-answer records, how does the
student's overconfidence depend on the teacher's? Is the loop amplifying (gain > 1) or
damping (gain < 1), where are its fixed points, and what share of audited records moves them?

The claim under test is from Cotton-Barratt & Hadshar (Forethought, 2026): model output feeds
the next model's training data, so epistemic quality is self-reinforcing and can lock in at a
very good or very bad equilibrium. Their footnote 2 speculates that the bad direction
reinforces more strongly.

## Prior work and what is new

- Sumit et al., arXiv 2611.01059: five rounds of self-training on Qwen3-1.7B, token-probability
  confidence, one starting point, continues from previous weights, one reference-data share.
- Roe et al., arXiv 2605.01130: iterated fine-tuning from a fresh base for seven persona
  traits; traits mostly decay or stay; no confidence, no ground-truth mixing.
- New here: (1) the starting level is a designed factor with seven levels on both sides of
  calibrated, giving an input-output map; (2) stated (verbalized) confidence;
  (3) audited-record share as a second factor with four levels; (4) the map's prediction is
  checked against real three-generation chains.

## Definitions

- **Record**: one question with the model's reply `Answer: <text>` / `Confidence: <n>%`.
- **Overconfidence gap** (primary metric): mean stated confidence minus accuracy, in
  percentage points, on the evaluation set E, greedy decoding. Positive = overconfident.
- **Teacher**: the model whose records the student is trained on. **Student**: a fresh LoRA on
  the unmodified base model. Students never inherit weights.
- **Return map**: student gap (y) as a function of teacher gap (x), both measured on E.
- **Gain**: slope b of the line y = a + b x fitted across teachers at one audit share.
- **Fixed point**: x* = a / (1 - b).
- **Audit share** (alpha): fraction of training records whose confidence was corrected by an
  audit. The audit knows whether each audited answer is right; it replaces the stated
  confidence by the isotonic-regression estimate of accuracy at that stated confidence, fitted
  on the audited records only, rounded to 5 and clipped to 5-95. The answer text is not
  changed, so only one thing differs between an audited and a raw record.

## Factor inventory

| Parameter | State | Value |
|---|---|---|
| Teacher confidence shift (logit units) | Varied | -2.0, -1.33, -0.67, 0, +0.67, +1.33, +2.0; plus the untouched base model as an eighth teacher |
| Audit share alpha | Varied | 0, 0.10, 0.25, 0.50 (audited sets nested) |
| Training seed | Varied (replicates) | seed 1 everywhere; seeds 2, 3 at alpha 0 and 0.25 for shifts -2.0, 0, +2.0 |
| Teacher decoding for records (secondary) | Varied | main: sampling T=1.0, top-p 0.95; secondary at alpha 0: greedy |
| Tier tag (secondary) | Varied | off; on at alpha 0.10 and 0.25 (system prompt line marks audited vs raw records; evaluated under both tags) |
| Generation depth (validation) | Varied | 1 everywhere; 3 for chains from shifts -2.0 and +2.0 at alpha 0 and 0.25 |
| Base model | Fixed | Qwen/Qwen2.5-7B-Instruct, bf16 (one family: a stated limitation; second family only if time allows, as a declared add-on) |
| Questions | Fixed | TriviaQA rc.nocontext validation, shuffled with seed 20261114; T = 2000 (teacher install), D = 2000 (records for students), E = 1000 (evaluation); disjoint |
| System prompt, user turn, reply format | Fixed | `loopgain/core.py`; one rendered input per cell type stored with results |
| Evaluation decoding | Fixed | greedy, repetition penalty 1.0, max 40 new tokens |
| LoRA and optimiser | Fixed | r 16, alpha 32, dropout 0, all attention and MLP projections, AdamW lr 2e-4, 3 epochs, batch 16, linear schedule with 10 warm-up steps, loss on the reply only |
| Scoring | Fixed | rule-based: normalised exact match against any alias, or an alias of 3+ characters contained in a reply of at most 10 tokens |
| Parser | Fixed | one regex in `core.parse`; unparsed replies are dropped and the retained share is reported per cell |
| GPU type, library versions | Fixed | L40S (see Deviations); pinned in `modal_app.py` |
| Hosted-model serving details (panel only) | Uncontrolled | named as a limitation |

## How teachers get a known level

1. The base model answers T greedily and K = 8 times by sampling.
2. Agreement of the samples with the greedy answer is mapped to accuracy by isotonic
   regression: p_i, a calibrated confidence for question i.
3. Teacher label for shift s: sigmoid(logit(p_i) + s), rounded to 5, clipped to 5-95. The
   answer is always the base model's own greedy answer, so the shift changes confidence only.
4. One LoRA per shift. The teacher's level used in the map is its measured gap on E, not the
   nominal shift.

## Predictions (written before data)

- P1. At alpha 0 the gain is between 0.7 and 1.0: the trait is mostly copied, not amplified.
- P2. The gain falls as alpha rises, and the fixed point moves toward a gap of 0.
- P3. Asymmetry: the slope fitted on teachers above the fixed point equals the slope below it
  (two-sided; the Forethought footnote predicts a steeper slope on the overconfident side).
- P4. The gap after 2 and 3 generations lies inside the 95% interval predicted by iterating
  the fitted map.
- P5. Secondary: students of more overconfident teachers change their answer less often after
  content-free pushback (stubbornness, not sycophancy, moves with overconfidence).
- P6. Secondary: with the tier tag, evaluation under "verified" at alpha 0.10 gives a gap at
  most as large as the untagged alpha 0.25 cell.

A failed prediction is reported as a result.

## Analysis rules

- Primary: gap on E per cell with a 95% bootstrap interval over the 1000 questions (2000
  resamples; teacher and student resampled on the same questions for the map fit).
- Noise floor: the spread of the gap across the three training seeds. An effect smaller than
  twice that spread is called "not resolved".
- Inclusion: a cell enters if at least 90% of E replies parse. Excluded cells are listed with
  their share. All figures and tables call one function, `analyze.included`.
- Also reported per cell: accuracy, mean confidence, ECE (10 bins), AUROC of confidence for
  correctness, share of initially correct answers changed after pushback, share of initially
  wrong answers changed after pushback, mean confidence change after pushback.
- Controls with known answers: the shift-0 teacher (calibrated by construction), the untouched
  base model, the base-as-teacher student at alpha 0 (a sham: training on the base model's own
  records), and a base-weight checksum before and after every adapter.

## Waves and budget

Wave 0: CPU tests and a dry run of every cell. Wave 1: a GPU smoke run on 64 records, then the
base pass. Wave 2: seven teachers. Wave 3: students (main grid first, then seeds, tag, greedy).
Wave 4: chains. Budget cap: Modal $100, Nebius $100. Each run writes cost and wall time.

## Stopping rule

Training stops Sunday 15 Nov 10:00 CET whatever has finished. Cells not finished are listed.

## Deviations

- 2026-11-14 15:05 CET. GPU changed from A10G to L40S for every cell: the smoke run ran out of memory on the 24 GB card. No model output existed yet. The GPU type is the same for all cells.
- 2026-11-14 15:29 CET. Declared add-on started: second family `unsloth/Meta-Llama-3.1-8B-Instruct` (ungated mirror of Llama 3.1 8B Instruct), same recipe, reduced grid: base pass, seven teachers, students at alpha 0 and 0.25. Decided after seeing 8 of 66 Qwen student cells (alpha 0 and 0.10 only); the grid for the add-on was fixed before any Llama output existed.
- 2026-11-14. The first student launch was cancelled by a dropped network connection after 8 cells; the remaining 58 were relaunched unchanged as detached jobs. Finished cells were kept (skip-existing), none were rerun.
- 2026-11-14 15:58 CET. Post-hoc arms, added after seeing the main map (gain about 1.0 at audit shares 0, 0.10 and 0.25, then 0.42 at 0.50) and after finding that students of audited data move in sampled confidence but not in greedy confidence. These arms are exploratory and are labelled so in the paper: (a) stage `es`: the 8 x 7 map at audit shares 0, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, each cell evaluated with greedy and with one sampled reply per question (new cells, suffix `__es`; the original 32 cells stay as an independent replicate of the greedy readout); (b) chains to 4 generations from shifts -2.0 and +2.0 in three arms: sampled records at audit share 0, sampled records at 0.25, greedy records at 0.25. They replace the planned `chain2`/`chain3` stages, which were not run. Prediction written before these runs: with sampled records at 0.25 the sampled gap shrinks by about a quarter per generation and the greedy gap stays for two generations and drops at the third; with greedy records at 0.25 neither moves. (c) The Llama add-on uses audit shares 0, 0.25, 0.50, 0.75 with both readouts.
- 2026-11-14 16:52 CET. Llama add-on cells run on A100-80GB, not L40S: training ran out of memory on 48 GB (longer chat template, larger output layer). Recipe and batch size unchanged. Also: retraining a cell with the same seed reproduced the greedy gap exactly in all 32 cells trained twice, so the `__es` cells are a determinism check, not an independent replicate; seed noise comes from the three-seed cells only.
- 2026-11-14 17:30 CET. Post-hoc controls for the tag arm, added after an external review of the draft pointed out that the tag comparison changed training and evaluation prompts together (stage `tagctl`, seven teachers, audit share 0.10): (a) students trained without tags, asked under the 'verified' and 'unverified' tag; (b) students trained with tags that have no prior meaning ('Data tier: K.' on audited records, 'Data tier: Q.' on the rest), asked under each. Prediction written before the runs: (a) gain near 1 under both tags; (b) gain under K below 0.3, under Q near 1.
- 2026-11-14 17:58 CET. Llama add-on finished: no-audit gain 1.12 (1.08-1.15), above one. Added because of this: the three chain arms for Llama, four generations. Prediction written before the runs: without audits the underconfident chain moves further from zero each generation (about 12% per generation, from -27 towards -40); the overconfident chain grows towards the ceiling set by the largest label (95% confidence at about 71% accuracy, so about +24) and stops there.
