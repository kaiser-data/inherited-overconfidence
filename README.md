# Inherited Overconfidence

**When a model is trained on another model's answers, how much of the teacher's overconfidence does it inherit, and how much checked data does it take to remove it?**

Entry for the Apart Research AI x Epistemics Sprint (13-15 Nov 2026), Track 1 and Track 2. Author: Martin Kaiser.

Paper: [`paper/paper.pdf`](paper/paper.pdf) · Frozen protocol and deviation log: [`PROTOCOL.md`](PROTOCOL.md) · Every number: [`paper/numbers.tex`](paper/numbers.tex)

## Result

Measured on Qwen2.5-7B (LoRA), 1,000 held-out TriviaQA questions, 164 trained models; repeated on a smaller grid with Llama-3.1-8B (60 models). *Gain* = one-generation transfer slope: how many points of the teacher's overconfidence gap (stated confidence minus accuracy) arrive in the student; 1 = all, 0 = none.

| Share of audited records | Gain, greedy answer | Gain, sampled answer |
|---|---|---|
| 0% | 1.03 | 1.03 |
| 25% | 0.99 | 0.79 |
| 50% | 0.42 | 0.50 |
| 75% | 0.07 | 0.26 |
| 10%, audited records tagged, asked under the tag | 0.11 | |
| 10%, meaningless tag (K/Q), asked under K | 0.07 | |
| 10%, trained without tags, asked under "verified" | 0.98 | |

1. Without audits a student copies its teacher's gap in both directions (Qwen 1.03, or 1.01 without the untouched base model; Llama 1.12 in the first step, not compounding). Four-generation chains stay near their start.
2. Unmarked audits correct a sampled answer in proportion to their share, and the greedy answer mainly between 40% and 60%.
3. A one-line tag on audited records makes a 10% share nearly sufficient; a meaningless tag works as well, the word alone does nothing.
4. The sampled-answer map forecasts four generations of chains at 25% audits to within about two points; the greedy map does not.

Full tables with intervals are in the paper. Not shown: other traits, pre-training, larger models, imperfect audits, chains on fresh questions. Intervals reflect question sampling with models fixed.

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

Training from scratch (Modal account, about $35 of L40S time for the Qwen grid; add `--fam llama31` and `LG_GPU=A100-80GB` for Llama, about $16):

```bash
cd loopgain
modal run modal_app.py --stage data                 # TriviaQA splits
modal run modal_app.py --stage base                 # base model pass, teacher datasets
modal run modal_app.py --stage teachers --dry-run   # CPU check of every cell, then without --dry-run
modal run --detach modal_app.py --stage es,seeds,tag,greedy --spawn
bash orchestrate.sh qwen                            # chain generations 2-4
modal run --detach modal_app.py --stage tagctl --spawn   # tag controls
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
| `results/` | per-question evaluation records (`runs*.json`), raw replies, question splits, audit order and every generation's records (`raw_*.tar.gz`), panel replies, summaries |
| `paper/` | report source, figure and number scripts |

## Prior work

All experiment code was written during the sprint. `paper/tools/prep_docx_tex.py` and `paper/tools/build_template.py` (LaTeX to Apart template) are reused from the author's Swarm Alignment sprint entry.

## AI use

An AI assistant (Claude, in Claude Code) proposed the ideas, wrote the protocol, code and analysis, ran the experiments and drafted the report. Martin Kaiser chose the idea and the trait, set the budget and reviewed design and results.
