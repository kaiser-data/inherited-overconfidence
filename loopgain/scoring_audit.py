"""Check the rule-based scorer against an LLM grader (Qwen3-235B on Nebius, temperature 0).

For each audited run: agreement between the two graders on E, and the overconfidence gap
under each. Resumable. Writes results/scoring_audit.json.

  python3 scoring_audit.py base teach_p000 teach_p200 teach_m200
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import panel  # noqa: E402

RES = HERE.parent / "results"
import os  # noqa: E402
# The 235B grader timed out on student runs twice (2026-11-14); LG_GRADER selects another model
# and writes to its own file, so the two graders are never mixed in one table.
GRADER = os.environ.get("LG_GRADER", "Qwen/Qwen3-235B-A22B-Instruct-2507")
OUTFILE = "scoring_audit.json" if "LG_GRADER" not in os.environ else f"scoring_audit_{GRADER.split('/')[-1]}.json"
PROMPT = ("Question: {q}\nReference answer: {ref}\nOther accepted answers: {al}\n"
          "Submitted answer: {a}\n\nDoes the submitted answer name the same thing as the reference "
          "answer or one of the accepted answers? Reply with exactly one word: YES or NO.")


CLAUDE = GRADER.startswith("claude-")
FAM = os.environ.get("LG_FAM", "qwen25")  # which family's runs file holds the named runs


def _claude_key():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    envf = HERE.parent / ".env"  # gitignored
    for line in (envf.read_text().splitlines() if envf.exists() else []):
        if line.startswith("ANTHROPIC_API_KEY="):
            os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
            return
    sys.exit("ANTHROPIC_API_KEY not set and not in .env")


def _claude_chat(msg):
    # Haiku 5.5: no temperature (non-default values 400); thinking on by default, effort kept low.
    import anthropic
    global _CLIENT
    if "_CLIENT" not in globals():
        _CLIENT = anthropic.Anthropic(max_retries=6, timeout=120)
    try:
        m = _CLIENT.messages.create(model=GRADER, max_tokens=2000,
                                    messages=[{"role": "user", "content": msg}],
                                    extra_body={"output_config": {"effort": "low"}})
    except anthropic.APIError:
        return None
    if m.stop_reason == "refusal":
        return None
    return "".join(b.text for b in m.content if b.type == "text")


def grade(it, ans, k):
    msg = PROMPT.format(q=it["q"], ref=it["value"], al="; ".join(it["aliases"][:8]), a=ans)
    raw = _claude_chat(msg) if CLAUDE else panel.chat(GRADER, [{"role": "user", "content": msg}], k)[0]
    if raw is None:
        return None
    t = raw.strip().upper()
    return True if t.startswith("YES") else False if t.startswith("NO") else None


def main():
    names = sys.argv[1:] or ["base"]
    d = json.loads((RES / ("runs.json" if FAM == "qwen25" else f"runs_{FAM}.json")).read_text())
    E = {r["id"]: r for r in d["data"]["E"]}
    if CLAUDE:
        _claude_key()
        k = None
    else:
        k = panel.key()
    outp = RES / (OUTFILE if FAM == "qwen25" else OUTFILE.replace(".json", f"_{FAM}.json"))
    out = json.loads(outp.read_text()) if outp.exists() else {}
    for n in names:
        if n in out:
            continue
        rows = [r for r in d["runs"][n]["eval"]["eval"] if r["answer"] is not None and r["conf"] is not None]
        with ThreadPoolExecutor(int(os.environ.get("LG_THREADS", "16"))) as ex:
            llm = list(ex.map(lambda r: grade(E[r["id"]], r["answer"], k), rows))
        ok = [(r, g) for r, g in zip(rows, llm) if g is not None]
        agree = sum(bool(r["correct"]) == g for r, g in ok) / len(ok)
        mc = sum(r["conf"] for r, _ in ok) / len(ok)
        out[n] = {"n": len(rows), "graded": len(ok), "agreement": agree,
                  "rule_yes_llm_no": sum(bool(r["correct"]) and not g for r, g in ok),
                  "rule_no_llm_yes": sum((not r["correct"]) and g for r, g in ok),
                  "acc_rule": 100 * sum(bool(r["correct"]) for r, _ in ok) / len(ok),
                  "acc_llm": 100 * sum(g for _, g in ok) / len(ok),
                  "gap_rule": mc - 100 * sum(bool(r["correct"]) for r, _ in ok) / len(ok),
                  "gap_llm": mc - 100 * sum(g for _, g in ok) / len(ok),
                  "examples_disagree": [{"q": E[r["id"]]["q"], "ref": E[r["id"]]["value"], "answer": r["answer"],
                                         "rule": bool(r["correct"]), "llm": g}
                                        for r, g in ok if bool(r["correct"]) != g][:12]}
        outp.write_text(json.dumps(out, indent=1))
        o = out[n]
        print(f"{n:12s} graded {o['graded']} agreement {o['agreement']:.3f} rule-yes/llm-no {o['rule_yes_llm_no']} "
              f"rule-no/llm-yes {o['rule_no_llm_yes']} gap rule {o['gap_rule']:+.1f} llm {o['gap_llm']:+.1f}")


if __name__ == "__main__":
    main()
