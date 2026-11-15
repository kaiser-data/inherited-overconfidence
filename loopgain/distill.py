"""Distillation-ladder arm (PROTOCOL.md, Deviations, 2026-11-15): teachers of three sizes,
the same answers with different confidence labels.

  python3 distill.py gen   --teacher q3_30b      # hosted teacher answers D once (resumable)
  python3 distill.py build                        # label sets -> results/dist/<name>/d_main.jsonl
  python3 distill.py report                       # table from results/runs_dist_<fam>.json

Label sets per teacher (answer text and record ids identical in all of them):
  raw   the teacher's own stated confidence
  tcal  isotonic fit of the TEACHER's correctness on its stated confidence (the existing audit
        at share 1.0: core.build_student_records)
  scal  the same fit against the STUDENT base model's correctness on the same question
  shuf  the tcal values permuted across records
The files carry the final label in `conf`; cells train on them at audit share 0.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
import tarfile
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402
import core  # noqa: E402

RES = HERE.parent / "results"
OUT = RES / "dist"
HOSTED = {"q3_30b": "Qwen/Qwen3-30B-A3B-Instruct-2507",
          "q3_235b": "Qwen/Qwen3-235B-A22B-Instruct-2507"}
LOCAL = {"q3_8b": ("raw_qwen3.tar.gz", "runs_qwen3/base/d_main.jsonl")}
TEACHERS = ["q3_8b", "q3_30b", "q3_235b"]
# Student base model's own sampled answers on D (same decoding as the records and as the
# sampled readout the predictions are tested on). A question it gave no parsed answer to
# counts as wrong.
STUDENTS = {"qwen25": ("raw_qwen25.tar.gz", "runs/base/d_main.jsonl"),
            "llama31": ("raw_llama31.tar.gz", "runs_llama31/base/d_main.jsonl")}
TOKENIZERS = {"qwen25": "Qwen/Qwen2.5-7B-Instruct", "llama31": "unsloth/Meta-Llama-3.1-8B-Instruct"}
SHUF_TEACHERS = ["q3_235b"]
MAX_TOKENS = 200
MAX_WORDS = 15  # keeps the whole reply inside the 192-token training window and the 40-token
#                 evaluation budget; `check` verifies the window with both student tokenizers
GEN = dict(temperature=1.0, top_p=0.95)


def _tar_rows(tar: str, member: str) -> list[dict]:
    with tarfile.open(RES / tar) as t:
        f = t.extractfile(member)
        return [json.loads(l) for l in io.TextIOWrapper(f, encoding="utf-8") if l.strip()]


def _tar_json(tar: str, member: str):
    with tarfile.open(RES / tar) as t:
        return json.load(t.extractfile(member))


def load_d() -> tuple[list[dict], list[str]]:
    return (_tar_rows("raw_qwen25.tar.gz", "data/D.jsonl"),
            _tar_json("raw_qwen25.tar.gz", "data/audit_perm.json"))


# ---------------------------------------------------------------- pure label logic (tested)

def clean(raw: str | None) -> str | None:
    """Hosted models sometimes write markdown ('**Answer:** x'); the asterisks are removed
    before parsing so they are neither a parse failure nor part of a trained answer."""
    return None if raw is None else raw.replace("*", "")


def to_row(item: dict, raw: str | None) -> dict:
    p = core.parse(clean(raw) or "")
    ans, conf = p if p else (None, None)
    return {"id": item["id"], "raw": raw, "answer": ans, "conf": conf,
            "correct": bool(ans is not None and core.is_correct(ans, item["aliases"]))}


def retained(rows: list[dict]) -> list[dict]:
    """Records every label set of a teacher is built from."""
    return [r for r in rows if r.get("answer") is not None and r.get("conf") is not None
            and len(r["answer"].split()) <= MAX_WORDS]


def tcal(rows: list[dict], questions: dict, perm: list[str]) -> dict[str, int]:
    """The existing audit at share 1.0, called through the code every earlier cell used."""
    recs, _ = core.build_student_records(rows, questions, 1.0, perm)
    return {r["id"]: r["conf"] for r in recs}


def scal(rows: list[dict], student_correct: dict[str, bool]) -> dict[str, int]:
    """Isotonic fit of the student base's correctness (False if it gave no parsed answer) on
    the teacher's stated confidence, rounded and clipped like every audit."""
    blocks = core.isotonic_fit([r["conf"] for r in rows],
                               [1.0 if student_correct.get(r["id"], False) else 0.0 for r in rows])
    return {r["id"]: core.conf5(core.isotonic_predict(blocks, r["conf"])) for r in rows}


def shuf(labels: dict[str, int], seed: int = config.DATA_SEED) -> dict[str, int]:
    ids = sorted(labels)
    vals = [labels[i] for i in ids]
    random.Random(seed).shuffle(vals)
    return dict(zip(ids, vals))


def relabel(rows: list[dict], labels: dict[str, float]) -> list[dict]:
    return [{"id": r["id"], "answer": r["answer"], "conf": labels[r["id"]],
             "correct": r["correct"], "teacher_conf": r["conf"]} for r in rows]


def answers_hash(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda r: r["id"]):
        h.update(f"{r['id']}\t{r['answer']}\n".encode())
    return h.hexdigest()[:16]


def stats(rows: list[dict], student_correct: dict[str, bool] | None = None) -> dict:
    n = len(rows)
    confs = [int(round(r["conf"])) for r in rows]  # as written into the reply (core.reply_text)
    hist: dict[int, int] = {}
    for c in confs:
        hist[int(round(c))] = hist.get(int(round(c)), 0) + 1
    mode = max(hist, key=hist.get)
    out = {"n": n, "label_mean": sum(confs) / n, "label_mode": mode, "mode_share": hist[mode] / n,
           "teacher_accuracy": 100 * sum(r["correct"] for r in rows) / n,
           "hist": dict(sorted(hist.items())), "answers_sha": answers_hash(rows)}
    if student_correct is not None:
        out["student_base_accuracy"] = 100 * sum(student_correct.get(r["id"], False) for r in rows) / n
    return out


# ---------------------------------------------------------------- hosted generation

def _chat(model: str, msgs: list[dict], k: str) -> dict:
    import panel
    err = None
    for attempt in range(5):
        try:
            r = panel._call("/chat/completions", {"model": model, "messages": msgs,
                                                  "max_tokens": MAX_TOKENS, **GEN}, k)
            ch = r["choices"][0]
            return {"raw": ch["message"].get("content") or "", "finish": ch.get("finish_reason"),
                    "model": r.get("model"), "usage": r.get("usage") or {}, "attempts": attempt + 1}
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: {e.read()[:200]!r}"
            if e.code in (400, 401, 403, 404):
                break
        except Exception as e:  # timeouts, connection resets
            err = repr(e)
        time.sleep(2 * (attempt + 1))
    return {"raw": None, "error": err}


def gen(teacher: str, n: int | None, workers: int):
    """One sampled reply per D question. Only transport errors are retried; a reply that does
    not parse is kept as it is (and dropped later, with the share reported)."""
    import panel
    k, model = panel.key(), HOSTED[teacher]
    D, _ = load_d()
    D = D[:n] if n else D
    f = OUT / "_hosted" / f"{teacher}.jsonl"
    f.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if f.exists():
        done = {r["id"]: r for r in map(json.loads, f.read_text().splitlines()) if r.get("raw") is not None}
    todo = [it for it in D if it["id"] not in done]

    def one(it):
        r = _chat(model, core.messages(it["q"]), k)
        return {**to_row(it, r.get("raw")), **{kk: v for kk, v in r.items() if kk != "raw"},
                "t": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    with ThreadPoolExecutor(workers) as ex:
        for r in ex.map(one, todo):
            done[r["id"]] = r
    with open(f, "w") as fh:
        for it in D:
            if it["id"] in done:
                fh.write(json.dumps(done[it["id"]], ensure_ascii=False) + "\n")
    rows = [done[it["id"]] for it in D if it["id"] in done]
    ok = [r for r in rows if r["answer"] is not None]
    fin: dict = {}
    for r in rows:
        fin[r.get("finish")] = fin.get(r.get("finish"), 0) + 1
    print(json.dumps({"teacher": teacher, "model": model, "asked": len(D), "answered": len(rows),
                      "parsed": len(ok) / max(1, len(rows)),
                      "accuracy_parsed": 100 * sum(r["correct"] for r in ok) / max(1, len(ok)),
                      "accuracy_unparsed_wrong": 100 * sum(r["correct"] for r in rows) / max(1, len(rows)),
                      "finish": fin, "had_markdown": sum("*" in (r["raw"] or "") for r in rows),
                      "over_max_words": len(ok) - len(retained(rows))}, indent=1))


# ---------------------------------------------------------------- build label sets

def teacher_rows(teacher: str) -> list[dict]:
    if teacher in LOCAL:
        return _tar_rows(*LOCAL[teacher])
    return [json.loads(l) for l in (OUT / "_hosted" / f"{teacher}.jsonl").read_text().splitlines()]


def build(teachers: list[str]):
    D, perm = load_d()
    q = {r["id"]: r for r in D}
    sc = {fam: {r["id"]: bool(r["correct"]) for r in _tar_rows(*src)} for fam, src in STUDENTS.items()}
    summary = {}
    for t in teachers:
        allrows = teacher_rows(t)
        if {r["id"] for r in allrows} - set(perm):
            raise RuntimeError(f"{t}: records outside D")
        rows = retained(allrows)
        sets = {f"{t}__raw": {r["id"]: r["conf"] for r in rows}, f"{t}__tcal": tcal(rows, q, perm)}
        for fam in STUDENTS:
            sets[f"{t}__scal_{fam}"] = scal(rows, sc[fam])
        if t in SHUF_TEACHERS:
            sets[f"{t}__shuf"] = shuf(sets[f"{t}__tcal"])
        info = {"n_in": len(allrows), "retained": len(rows) / len(allrows),
                "parsed": sum(r.get("answer") is not None for r in allrows) / len(allrows),
                "accuracy_unparsed_wrong": 100 * sum(bool(r.get("correct")) for r in allrows) / len(allrows),
                "sets": {}}
        for name, labels in sets.items():
            out = relabel(rows, labels)
            fam = name.split("__scal_")[1] if "__scal_" in name else None
            st = stats(out, sc[fam] if fam else None)
            assert st["answers_sha"] == answers_hash(rows), name
            (OUT / name).mkdir(parents=True, exist_ok=True)
            with open(OUT / name / "d_main.jsonl", "w") as fh:
                for r in out:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            info["sets"][name] = st
            print(f"{name:26s} n {st['n']} label mean {st['label_mean']:5.1f} mode {st['label_mode']} "
                  f"({st['mode_share']:.2f}) teacher acc {st['teacher_accuracy']:5.1f}"
                  + (f" student-base acc {st['student_base_accuracy']:5.1f}" if fam else ""))
        summary[t] = info
    old = json.loads((OUT / "labels.json").read_text()) if (OUT / "labels.json").exists() else {}
    (OUT / "labels.json").write_text(json.dumps({**old, **summary}, indent=1))


def check():
    """Every rendered training example must fit the training window for both students."""
    from transformers import AutoTokenizer
    D, _ = load_d()
    q = {r["id"]: r for r in D}
    for fam, name in TOKENIZERS.items():
        tok = AutoTokenizer.from_pretrained(name)
        for d in sorted(p for p in OUT.iterdir() if (p / "d_main.jsonl").exists()):
            rows = [json.loads(x) for x in (d / "d_main.jsonl").read_text().splitlines()]
            n = []
            for r in rows:
                p_ids = tok(tok.apply_chat_template(core.messages(q[r["id"]]["q"]), tokenize=False,
                                                    add_generation_prompt=True), add_special_tokens=False)["input_ids"]
                r_ids = tok(core.reply_text(r["answer"], r["conf"]) + tok.eos_token, add_special_tokens=False)["input_ids"]
                n.append((len(p_ids) + len(r_ids), len(r_ids)))
            over = sum(a > config.HP["max_len"] for a, _ in n)
            print(f"{fam:8s} {d.name:26s} max total {max(a for a, _ in n)} max reply {max(b for _, b in n)} "
                  f"over {config.HP['max_len']}: {over}")
            assert over == 0, (fam, d.name)


# ---------------------------------------------------------------- report

def _boot(rows: list[dict], n: int = 2000, seed: int = 1) -> tuple[float, float]:
    rng = random.Random(seed)
    ok = [r for r in rows if r.get("answer") is not None and r.get("conf") is not None]
    gaps = []
    for _ in range(n):
        s = [ok[rng.randrange(len(ok))] for _ in ok]
        gaps.append(sum(r["conf"] for r in s) / len(s) - 100 * sum(r["correct"] for r in s) / len(s))
    gaps.sort()
    return gaps[int(0.025 * n)], gaps[int(0.975 * n)]


def recalibrated(cal_rows: list[dict], e_rows: list[dict]) -> dict | None:
    """Post-hoc remedy on held-out questions: isotonic fit of the trained student's own
    correctness on its stated confidence (calibration set F1), applied to its replies on E."""
    cal = [r for r in cal_rows if r.get("answer") is not None and r.get("conf") is not None]
    ev = [r for r in e_rows if r.get("answer") is not None and r.get("conf") is not None]
    if not cal or not ev:
        return None
    blocks = core.isotonic_fit([r["conf"] for r in cal], [1.0 if r["correct"] else 0.0 for r in cal])
    new = [{"conf": 100 * core.isotonic_predict(blocks, r["conf"]), "correct": r["correct"], "answer": r["answer"]}
           for r in ev]
    m = core.metrics(new)
    return {"gap": m["gap"], "ece": m["ece"], "auroc": m["auroc"], "n_cal": len(cal),
            "cal_accuracy": 100 * sum(bool(r["correct"]) for r in cal) / len(cal)}


def report():
    labels = json.loads((OUT / "labels.json").read_text())
    setinfo = {n: s for t in labels.values() for n, s in t["sets"].items()}
    out = []
    for fam in STUDENTS:
        f = RES / f"runs_dist_{fam}.json"
        if not f.exists():
            continue
        runs = json.loads(f.read_text())["runs"]
        for name, run in sorted(runs.items()):
            meta = run["meta"]
            si = setinfo.get(meta["teacher"], {})
            L = si.get("label_mean", meta["data"]["train_conf_mean"])  # labels as rendered
            for readout, key, dkey in (("sampled", "eval_sampled", "d_main"), ("greedy", "eval", "d_greedy")):
                rows = run["eval"][key]
                m = core.metrics(rows)
                lo, hi = _boot(rows)
                rc = recalibrated(run["eval"][dkey], rows) if dkey in run["eval"] else None
                out.append({"family": fam, "cell": name, "teacher": meta["teacher"], "seed": meta["seed"],
                            "readout": readout, "parsed": m["parsed"], "included": m["parsed"] >= 0.9,
                            "label_mean": L, "label_mode": si.get("label_mode"), "mode_share": si.get("mode_share"),
                            "record_accuracy": 100 * meta["data"]["train_accuracy"],
                            "n_at_max_len": meta["train"].get("n_at_max_len"),
                            "accuracy": m["accuracy"], "conf": m["mean_conf"], "gap": m["gap"],
                            "gap_lo": lo, "gap_hi": hi, "label_minus_acc": L - m["accuracy"],
                            "conf_minus_label": m["mean_conf"] - L, "auroc": m["auroc"], "ece": m["ece"],
                            "recal": rc})
    (RES / "dist_cells.json").write_text(json.dumps(out, indent=1))
    for r in out:
        rc = r["recal"]
        print(f"{r['family']:8s} {r['teacher']:22s} s{r['seed']} {r['readout']:7s} parsed {r['parsed']:.3f} "
              f"L {r['label_mean']:5.1f} C {r['conf']:5.1f} A {r['accuracy']:5.1f} gap {r['gap']:+6.1f} "
              f"[{r['gap_lo']:+.1f},{r['gap_hi']:+.1f}] C-L {r['conf_minus_label']:+5.1f} "
              f"auroc {r['auroc'] or float('nan'):.2f} ece {r['ece']:.1f}"
              + (f" | recal gap {rc['gap']:+.1f} ece {rc['ece']:.1f}" if rc else "")
              + ("" if r["included"] else "  EXCLUDED (<90% parsed)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "build", "check", "report"])
    ap.add_argument("--teacher", default="")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    if a.cmd == "gen":
        gen(a.teacher, a.n, a.workers)
    elif a.cmd == "build":
        build(a.teacher.split(",") if a.teacher else TEACHERS)
    elif a.cmd == "check":
        check()
    else:
        report()


if __name__ == "__main__":
    main()
