# Inherited Overconfidence

**When a model is trained on another model's answers, how much of the teacher's overconfidence does it inherit, and how much checked data does it take to remove it?**

Entry for the Apart Research AI x Epistemics Sprint (13-15 Nov 2026), Track 1 and Track 2. Author: Martin Kaiser.

Paper: [`paper/paper.pdf`](paper/paper.pdf) · Frozen protocol and deviation log: [`PROTOCOL.md`](PROTOCOL.md) · Every number: [`paper/numbers.tex`](paper/numbers.tex)

## Result in four lines

Measured on Qwen2.5-7B (LoRA), 1,000 held-out TriviaQA questions, 150 trained models. *Gain* = how many points of the teacher's overconfidence gap (stated confidence minus accuracy) arrive in the student; 1 = all, 0 = none.

| Share of audited records | Gain, most likely answer | Gain, sampled answer |
|---|---|---|
| 0% | 1.03 | 1.03 |
| 25% | 0.99 | 0.79 |
| 50% | 0.42 | 0.50 |
| 75% | 0.07 | 0.26 |
| 10%, audited records tagged, asked under the tag | 0.11 | |

1. Without audits a student copies its teacher's gap, in both directions and over four generations.
2. Unmarked audits correct a sampled answer in proportion to their share, and the most likely answer only past about half.
3. A one-line "verified" tag on audited records makes a 10% share nearly sufficient.
4. Chains trained on most-likely answers do not pass a minority correction on.

Full table with intervals: Table 1 of the paper. What is not shown: other traits, pre-training, larger models, imperfect audits.

![return map](paper/fig1-return-map.png)

## Reproduce

Analysis from the published results (no GPU, about a minute):

```bash
pip install numpy matplotlib
python3 loopgain/analyze.py          # results/runs.json -> results/summary.json, cells.csv
python3 paper/make_numbers.py        # -> paper/numbers.tex (every number in the paper)
python3 paper/figures.py             # -> paper/fig*.pdf, prints every plotted value
python3 -m pytest -q loopgain/tests  # CPU tests of scoring, audit mixing, metrics, grid
```

Training from scratch (Modal account, about $30 of L40S time for the Qwen grid):

```bash
cd loopgain
modal run modal_app.py --stage data                 # TriviaQA splits
modal run modal_app.py --stage base                 # base model pass, teacher datasets
modal run modal_app.py --stage teachers --dry-run   # CPU check of every cell, then without --dry-run
modal run --detach modal_app.py --stage es,seeds,tag,greedy --spawn
bash orchestrate.sh qwen                            # chain generations 2-4
modal run modal_app.py --stage pull
```

`paper/build.sh` rebuilds the report in the Apart template (needs pandoc and LibreOffice).

## Layout

| Path | What |
|---|---|
| `PROTOCOL.md` | question, factor table, predictions, analysis rules; frozen before the first model output; deviations logged with time and reason |
| `loopgain/core.py` | prompts, parser, scorer, teacher labels, audit mixing, metrics (no third-party imports) |
| `loopgain/config.py` | the grid: every cell compared in the paper |
| `loopgain/gpu.py`, `modal_app.py` | LoRA training, generation, evaluation; dry run, skip-existing, detached launch |
| `loopgain/analyze.py` | inclusion rule, return-map fits with paired bootstrap, chains, seed noise |
| `loopgain/panel.py` | hosted-model panel on Nebius Token Factory |
| `loopgain/scoring_audit.py` | rule-based scorer against an LLM grader |
| `results/` | every evaluated reply's parsed answer, confidence and score (`runs*.json`), panel replies, summaries |
| `paper/` | report source, figure and number scripts |

## Prior work

All experiment code was written during the sprint. `paper/tools/prep_docx_tex.py` and `paper/tools/build_template.py` (LaTeX to Apart template) are reused from the author's Swarm Alignment sprint entry.

## AI use

An AI assistant (Claude, in Claude Code) proposed the ideas, wrote the protocol, code and analysis, ran the experiments and drafted the report. Martin Kaiser chose the idea and the trait, set the budget and reviewed design and results.
