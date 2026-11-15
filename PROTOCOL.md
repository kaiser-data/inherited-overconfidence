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
- 2026-11-14 20:53 CET. Three arms added after the second external review, chosen by the author from a list of candidates; each addresses a stated limitation. Nothing from these arms existed when this entry was written. Predictions are written here before the runs.
  (a) **Cross-family transfer** (stages `xllama31`, `xqwen25`). In every earlier cell teacher and student share a base model and so an accuracy; the map cannot tell whether a student inherits the teacher's *gap* or only its *stated confidence*. New cells: Qwen2.5-7B students trained on the records of the eight Llama-3.1-8B teachers, and Llama students trained on the records of the eight Qwen teachers, at audit shares 0 and 0.75, both readouts, seed 1 (32 cells; the existing same-family cells at these shares are the comparison). Teacher records, questions and audited ids are the existing ones. Record accuracy is about 70% for Llama teachers and 56% for Qwen teachers. Prediction: the student states its teacher's confidence. At share 0 its mean stated confidence is within 5 points of the mean confidence of its training records, so its gap is the teacher's record gap plus the accuracy difference: Qwen students of Llama teachers lie 10 to 17 points above the diagonal, Llama students of Qwen teachers 10 to 17 below, slope 0.9 to 1.1 in both. At share 0.75 the audited values are calibrated for the teacher's answers, so Qwen students of Llama teachers keep a gap of +8 to +16 for every teacher and Llama students of Qwen teachers -8 to -16 (same-family cells at this share: within about 4 points of zero). The prediction fails if the intercepts are within 5 points of zero.
  (b) **Forged tags** (stage `forge`, Qwen, seven teachers). Tagged arm with 10% audited records; in addition 0, 5, 10 or 20% of all records carry the 'verified' tag without being audited (the next ids of the fixed order; they keep the teacher's confidence). Evaluated under both tags, greedy and sampled (28 cells; the 0% cells repeat the existing tagged cells and add the sampled readout). Prediction from the mixture reading of the unmarked map: under 'verified' the model behaves like an untagged model whose audit share is the audited fraction of the tagged records (1, 0.67, 0.5, 0.33). Sampled gain under 'verified' about 0.1, 0.33, 0.5, 0.67; greedy gain about 0.1, 0.15, 0.4, 0.9, i.e. the greedy answer stays corrected while forgeries are fewer than audits and reverts once they outnumber them. Under 'unverified' the gain stays within 0.05 of 1 at every level.
  (c) **Fresh-question chains** (stage `fresh`, then `fqg1`..`fqg4`, Qwen). The earlier chains reused the 2,000 record questions and the audited ids in every generation. New chains from shifts -2.0 and +2.0 at audit shares 0 and 0.25 with sampled records, four generations, in which generation g writes its records for a question set no earlier generation answered (D, then F1, F2, F3: the next 6,000 questions of the fixed shuffled order; if fewer than 6,000 unused questions exist, the last set is the teacher-installation set T, which no student has seen) and the audited ids are a new draw in each set (16 cells). Generation 1 is trained on the same records as in the earlier chains and must reproduce them exactly. Prediction: no difference from the same-question chains beyond seed noise: without audits the greedy gap stays within 3 points of its start; at 25% the generation-4 sampled gap is within 4 points of the same-question chain (-13 and +12).
  Also added at this time, from existing outputs only (no new runs): the share of audit-style confidence values in each student's replies (a test of the mixture hypothesis), the map against the gap of the training records instead of the teacher's greedy gap, and a forecast that uses that like-for-like map.
- 2026-11-14 22:15 CET. Outcome of the three arms above (all cells finished, none excluded). (a) Qwen students of Llama teachers: stated confidence within 1 point of the records on average, intercept +10.8 (prediction 10 to 17: held), slope 1.12 against the teacher's greedy gap (prediction 0.9 to 1.1: missed); at share 0.75 gaps +6 to +17 (prediction +8 to +16: approximately held). Llama students of Qwen teachers: intercept -7.5, gaps -10 to -6 at share 0.75 (prediction -10 to -17 and -8 to -16: too large; the students' accuracy fell to 63-65%). (b) Forged tags: sampled gain under 'verified' 0.11, 0.38, 0.56, 0.69 (prediction 0.1, 0.33, 0.5, 0.67: held within 0.06); greedy gain 0.11, 0.29, 0.68, 0.83 (prediction 0.1, 0.15, 0.4, 0.9: direction held, the rise came earlier); under 'unverified' 0.97 to 0.99 (held). (c) Fresh-question chains: generation 1 identical to the earlier chains; without audits the greedy gap moved at most 3.6 points from its start (prediction 3: missed by 0.6); generation-4 sampled gaps at 25% were -10 and +13 against -13 and +12 (prediction within 4: held). Only 4,810 unused questions existed, so the third fresh set is the teacher-installation set, as provided for above. A scorer check on student models was started twice and timed out on the hosted grader both times; students remain unchecked.
- 2026-11-15 02:55 CET. Third family added at the author's request because both families are over a year old: `Qwen/Qwen3-8B` (April 2025, Apache-2.0) with thinking switched off (`enable_thinking=False`; the template then writes an empty think block into every prompt, in training and evaluation alike). Same recipe and reduced grid as Llama: base pass, seven teachers, students at audit shares 0, 0.25, 0.50, 0.75 with both readouts (40 runs). Needs transformers 4.51.3 / peft 0.15.2 / accelerate 1.6.0 (earlier families keep their pins); A100-80GB. Smoke run on 64 records: format parsed 100%, base weights unchanged, but 12 steps did not move the greedy confidence to the trained 30% (Qwen2.5: it did); the teachers are therefore checked against their installed levels before students are launched. Predictions written before any Qwen3 base or teacher output: (1) teachers span at least 40 points of gap; (2) at share 0 the slope of student gap on record gap is 0.95 to 1.10; (3) sampled gain within 0.10 of one minus the share at 0.25, 0.50, 0.75; (4) greedy gain at least 0.85 at 0.25 and at most 0.30 at 0.75; (5) mixture weight in sampled answers within 0.10 of the share, in greedy answers below 0.30 at 0.25 and above 0.70 at 0.75.
  Also found now: runs record the GPU type from an environment variable that was not set inside the containers, so every Llama run reads "L40S" although cells after 16:52 ran on A100-80GB, and their recorded cost uses the L40S rate. The image now carries the variable; reported Llama cost is corrected by rate in the paper.
- 2026-11-15 03:14 CET. Qwen3-8B teacher check, before any student output existed. Base gap +39.6 (accuracy 0.555, 99.85% parsed). Teachers m200 to p200: -31.5, -16.6, -7.1, +3.3, +13.6, +21.6, +27.9 (accuracy 0.556-0.562; none excluded). Span 59.4 points: prediction (1) held. The 32 student cells were launched unchanged at 03:14 CET; the 10:00 stopping rule applies.
- 2026-11-15 03:50 CET. Qwen3-8B students finished at 03:46 CET (32/32, none excluded). Outcome of the five predictions of 02:55: (2) slope of student gap on record gap at share 0: greedy 0.97 (0.94-0.99), sampled 1.01 (0.99-1.04): held. (3) sampled gain against the teacher's gap 0.77, 0.53, 0.24 at shares 0.25, 0.50, 0.75 (one minus the share: 0.75, 0.50, 0.25): held. (4) greedy gain 0.97 at 0.25, 0.01 at 0.75: held. (5) mixture weight in sampled answers 0.27, 0.49, 0.74; in greedy answers mean 0.03 (max 0.28) at 0.25 and 1.00 at 0.75: held. Greedy gain at 0.50 is 0.48, with individual mixture weights from 0.08 to 0.84. No cross-family cells and no chains beyond generation 1 were run for Qwen3.

