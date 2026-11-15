"""Modal driver. Waves, cheapest first:

  modal run modal_app.py --stage data                 # CPU: build T / D / E from TriviaQA
  modal run modal_app.py --stage smoke                # GPU, a few minutes: whole pipeline on 64 records
  modal run modal_app.py --stage base                 # GPU: base model on T, D, E; teacher datasets
  modal run modal_app.py --stage teachers --dry-run   # CPU: build every cell's training set, no GPU
  modal run modal_app.py --stage teachers
  modal run modal_app.py --stage students             # then seeds, tag, greedy, chain2, chain3
  modal run modal_app.py --stage pull                 # copy results to ../results/

Cells that already have metrics.json are skipped unless --force. --only a,b restricts by name.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

HERE = Path(__file__).parent
GPU = os.environ.get("LG_GPU", "L40S")
RATE = {"A10G": 1.10, "L40S": 1.95, "A100-40GB": 2.10, "A100-80GB": 2.50, "cpu": 0.05}

app = modal.App("loopgain")
vol = modal.Volume.from_name("loopgain-vol", create_if_missing=True)
# Library versions. Qwen3 needs transformers >= 4.51; that family runs with LG_TF=4.51, all
# earlier families keep the original pins (their cells were produced with them).
TF = os.environ.get("LG_TF", "4.46")
PINS = {"4.46": ("transformers==4.46.3", "peft==0.13.2", "accelerate==1.1.1"),
        "4.51": ("transformers==4.51.3", "peft==0.15.2", "accelerate==1.6.0"),
        "5.14": ("transformers==5.14.1", "peft==0.21.3", "accelerate==1.15.0")}[TF]
TORCH = "torch==2.15.0" if TF == "5.14" else "torch==2.5.1"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(TORCH, *PINS, "datasets==3.1.0", "hf_transfer==0.1.8", "numpy==2.1.3")
    .env({"HF_HOME": "/vol/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1",
          "TOKENIZERS_PARALLELISM": "false",
          "LG_GPU": GPU, "LG_TF": TF})  # so that each run records the GPU it actually ran on
    .add_local_dir(str(HERE), "/root/lg", ignore=["__pycache__", "tests", "*.pyc"])
)
V = Path("/vol")


def RUNS(fam: str = "qwen25") -> Path:
    return V / ("runs" if fam == "qwen25" else f"runs_{fam}")


def TDATA(fam: str = "qwen25") -> Path:
    """Teacher training sets depend on the base model's own answers, so they are per family."""
    return V / "data" if fam == "qwen25" else V / f"data_{fam}"


def _setup():
    import sys
    if "/root/lg" not in sys.path:
        sys.path.insert(0, "/root/lg")


def _read(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _write(path, rows: list[dict]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


FRESH = ("F1", "F2", "F3")  # record questions for the fresh-question chains (stage fresh)


def _load_data():
    q = {}
    for split in ("T", "D", "E") + FRESH:
        f = V / "data" / f"{split}.jsonl"
        if f.exists():
            for r in _read(f):
                q[r["id"]] = r
    perm = json.loads((V / "data" / "audit_perm.json").read_text())
    return q, perm


def _load_sets():
    """Record-question sets and their audit order. D is the set of the main study."""
    sets = {"D": _read(V / "data" / "D.jsonl")}
    perms = {"D": json.loads((V / "data" / "audit_perm.json").read_text())}
    for s in FRESH:
        f, pf = V / "data" / f"{s}.jsonl", V / "data" / f"audit_perm_{s}.json"
        if f.exists() and pf.exists():
            sets[s] = _read(f)
            perms[s] = json.loads(pf.read_text())
    return sets, perms


def _pool():
    """Every usable TriviaQA question in the fixed order that T, D and E were cut from
    (the same steps as prep_data)."""
    import random
    from datasets import load_dataset
    import config
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    rows, seen = [], set()
    for ex in ds:
        q = " ".join(ex["question"].split())
        a = ex["answer"]
        aliases = sorted({a["value"], *a["aliases"], *a.get("normalized_aliases", [])} - {""})
        if not aliases or len(q) > 200 or q in seen:
            continue
        seen.add(q)
        rows.append({"id": ex["question_id"], "q": q, "aliases": aliases, "value": a["value"]})
    rows.sort(key=lambda r: r["id"])
    random.Random(config.DATA_SEED).shuffle(rows)
    return rows


# ------------------------------------------------------------------ stage: data (CPU)

@app.function(image=image, volumes={"/vol": vol}, timeout=1800)
def prep_data(force: bool = False):
    _setup()
    import random
    from datasets import load_dataset
    import config, core
    out = V / "data"
    if (out / "E.jsonl").exists() and not force:
        return {"skipped": True}
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    rows, seen = [], set()
    for ex in ds:
        q = " ".join(ex["question"].split())
        a = ex["answer"]
        aliases = sorted({a["value"], *a["aliases"], *a.get("normalized_aliases", [])} - {""})
        if not aliases or len(q) > 200 or q in seen:
            continue
        seen.add(q)
        rows.append({"id": ex["question_id"], "q": q, "aliases": aliases, "value": a["value"]})
    rows.sort(key=lambda r: r["id"])
    random.Random(config.DATA_SEED).shuffle(rows)
    n = config.N_T + config.N_D + config.N_E
    assert len(rows) >= n, len(rows)
    T, D, E = rows[:config.N_T], rows[config.N_T:config.N_T + config.N_D], rows[config.N_T + config.N_D:n]
    _write(out / "T.jsonl", T)
    _write(out / "D.jsonl", D)
    _write(out / "E.jsonl", E)
    perm = core.seeded_perm([r["id"] for r in D], config.DATA_SEED)
    (out / "audit_perm.json").write_text(json.dumps(perm))
    vol.commit()
    return {"pool": len(rows), "T": len(T), "D": len(D), "E": len(E), "example": E[0]}


@app.function(image=image, volumes={"/vol": vol}, timeout=1800)
def prep_e2(force: bool = False):
    """Second evaluation set (extension arm): 1,000 questions of Natural Questions (open), validation split."""
    _setup()
    import random
    from datasets import load_dataset
    import config
    out = V / "data" / "E2.jsonl"
    if out.exists() and not force:
        return {"skipped": True, "n": len(_read(out))}
    ds = load_dataset("google-research-datasets/nq_open", split="validation")
    rows, seen = [], set()
    for i, ex in enumerate(ds):
        q = " ".join(ex["question"].split())
        aliases = sorted({a.strip() for a in ex["answer"]} - {""})
        if not aliases or len(q) > 200 or q in seen:
            continue
        seen.add(q)
        rows.append({"id": f"nq_{i}", "q": q[0].upper() + q[1:] + ("" if q.endswith("?") else "?"), "aliases": aliases, "value": aliases[0]})
    random.Random(config.DATA_SEED).shuffle(rows)
    E2 = rows[:config.N_E]
    _write(out, E2)
    vol.commit()
    return {"pool": len(rows), "E2": len(E2), "example": E2[0]}


def _as_words(recs: list[dict]) -> list[dict]:
    """Wording-only arm: same answers and labels, the confidence written as a phrase."""
    import core
    out = []
    for r in recs:
        ans = r["response"].split("\n")[0][len("Answer: "):]
        w, val = core.phrase_for(r["conf"])
        out.append({**r, "system": core.SYSTEM_WORDS, "response": core.reply_words(ans, r["conf"]),
                    "conf_number": r["conf"], "conf": val})
    return out


@app.function(image=image, volumes={"/vol": vol}, timeout=1800)
def prep_fresh(force: bool = False):
    """Question sets F1..F3 for the fresh-question chains: the next questions of the same
    fixed order, after T, D and E. Checks first that the order reproduces T, D and E."""
    _setup()
    import config, core
    out = V / "data"
    if all((out / f"{s}.jsonl").exists() for s in FRESH) and not force:
        return {"skipped": True, **{s: len(_read(out / f"{s}.jsonl")) for s in FRESH}}
    rows = _pool()
    n0 = config.N_T + config.N_D + config.N_E
    old = [r["id"] for s in ("T", "D", "E") for r in _read(out / f"{s}.jsonl")]
    if [r["id"] for r in rows[:n0]] != old:
        raise RuntimeError("question order does not reproduce T, D, E; not writing fresh sets")
    # The pool is deduplicated by exact question text only. Drop any later row that repeats
    # an id or a question (ignoring case and punctuation) used before, so the fresh sets
    # share nothing with T, D, E or with each other.
    norm = lambda t: "".join(ch for ch in t.lower() if ch.isalnum())
    seen_id = {r["id"] for r in rows[:n0]}
    seen_q = {norm(r["q"]) for r in rows[:n0]}
    rest = []
    for r in rows[n0:]:
        if r["id"] in seen_id or norm(r["q"]) in seen_q:
            continue
        seen_id.add(r["id"]); seen_q.add(norm(r["q"]))
        rest.append(r)
    info = {"pool": len(rows), "unused": len(rest), "dropped_repeats": len(rows) - n0 - len(rest)}
    T = _read(out / "T.jsonl")
    for k, s in enumerate(FRESH):
        part, src = rest[k * config.N_D:(k + 1) * config.N_D], "pool"
        if len(part) < config.N_D:
            # Not enough unused questions: the teacher-installation set T. No student ever saw
            # it; only the generation-0 teacher was trained on it.
            part, src = T, "T"
        _write(out / f"{s}.jsonl", part)
        (out / f"audit_perm_{s}.json").write_text(
            json.dumps(core.seeded_perm([r["id"] for r in part], config.DATA_SEED)))
        info[s] = {"n": len(part), "source": src, "first_id": part[0]["id"]}
    seen = set(old[config.N_T:])  # D and E
    for s in FRESH:
        ids = {r["id"] for r in _read(out / f"{s}.jsonl")}
        assert not (ids & seen), f"{s} overlaps an earlier record or evaluation set"
        seen |= ids
    vol.commit()
    return info


# ------------------------------------------------------------------ stage: smoke (GPU, cheap)

@app.function(image=image, volumes={"/vol": vol}, gpu=GPU, timeout=1800)
def smoke(fam: str = "qwen25"):
    _setup()
    import config, core, gpu
    t0 = time.time()
    q, perm = _load_data()
    E = _read(V / "data" / "E.jsonl")[:48]
    T = _read(V / "data" / "T.jsonl")[:64]
    base, tok = gpu.load_base(config.FAMILIES[fam])
    load_s = time.time() - t0
    cs0 = gpu.base_checksum(base)
    rendered = gpu.render(tok, core.messages(E[0]["q"]))
    rows = gpu.evaluate(base, tok, E)
    m_base = core.metrics(rows)
    # Train on reference answers with a fixed 30% label: the adapter must move the eval to ~30.
    recs = [{"system": core.SYSTEM, "user": core.user_turn(r["q"]),
             "response": core.reply_text(r["value"], 30)} for r in T]
    pm, log = gpu.train_lora(base, tok, recs, seed=1, hp={**config.HP, "epochs": 3})
    m_tuned = core.metrics(gpu.evaluate(pm, tok, E))
    sample = gpu.answer_rows(pm, tok, E[:8], "main", seed=1)
    base = pm.unload()
    cs1 = gpu.base_checksum(base)
    m_after = core.metrics(gpu.answer_rows(base, tok, E, "greedy"))
    return {"load_seconds": load_s, "total_seconds": time.time() - t0, "rendered_input": rendered,
            "first_rows": rows[:4], "base": m_base, "tuned_to_30": m_tuned, "train_log": log,
            "sampled_after_tuning": [r["raw"] for r in sample],
            "checksum_equal": cs0 == cs1, "base_after_unload_mean_conf": m_after.get("mean_conf"),
            "family": fam, "transformers": __import__("transformers").__version__, "gpu": GPU}


# ------------------------------------------------------------------ stage: base (GPU)

@app.function(image=image, volumes={"/vol": vol}, gpu=GPU, timeout=3 * 3600)
def base_pass(force: bool = False, fam: str = "qwen25", lite: bool = False):
    """lite: E and D only (students of the distillation arm need no installed teachers)."""
    _setup()
    import config, core, gpu
    run = RUNS(fam) / "base"
    if (run / "metrics.json").exists() and not force:
        return {"skipped": True}
    t0 = time.time()
    q, perm = _load_data()
    T, D, E = (_read(V / "data" / f"{s}.jsonl") for s in ("T", "D", "E"))
    base, tok = gpu.load_base(config.FAMILIES[fam])
    rendered = gpu.render(tok, core.messages(E[0]["q"]))
    ev = gpu.evaluate(base, tok, E)
    _write(run / "eval.jsonl", ev)
    _write(run / "d_main.jsonl", gpu.answer_rows(base, tok, D, "main", seed=1))
    _write(run / "d_greedy.jsonl", gpu.answer_rows(base, tok, D, "greedy"))
    t_rows = [] if lite else gpu.answer_rows(base, tok, T, "greedy")
    for r in t_rows:
        r["samples"] = []
    for k in range(0 if lite else config.K_CONSISTENCY):
        for r, s in zip(t_rows, gpu.answer_rows(base, tok, T, "main", seed=100 + k)):
            r["samples"].append(s["answer"])
    if not lite:
        _write(run / "t_base.jsonl", t_rows)
    infos = {}
    for lam in ([] if lite else config.LAMBDAS):
        recs, info = core.build_teacher_records(t_rows, q, lam)
        _write(TDATA(fam) / f"{config.teacher_name(lam)}.jsonl", recs)
        infos[config.teacher_name(lam)] = info
    secs = time.time() - t0
    m = core.metrics(ev)
    meta = {"name": "base", "kind": "base", "family": fam, "base": config.FAMILIES[fam],
            "metrics": m, "rendered_input": rendered,
            "teacher_data": infos, "seconds": secs, "gpu": GPU, "usd": secs / 3600 * RATE.get(GPU, 0),
            "t_parsed": None if lite else sum(r["answer"] is not None for r in t_rows) / len(t_rows),
            "d_main": core.metrics(_read(run / "d_main.jsonl")), "lite": lite}
    (run / "metrics.json").write_text(json.dumps(meta, indent=1))
    vol.commit()
    return meta


# ------------------------------------------------------------------ cells

def _records_for(cell: dict, q: dict, perm: list[str], fam: str = "qwen25", perms: dict | None = None):
    """perms: audit order per record-question set; perm is the order of D."""
    import core
    if cell["kind"] == "install":
        recs = _read(TDATA(fam) / f"{cell['name']}.jsonl")
        n = max(1, len(recs))
        return recs, {"n_kept": len(recs), "retained": None,
                      "train_conf_mean": sum(r["label"] for r in recs) / n,
                      "train_accuracy": sum(r["correct"] for r in recs) / n}
    src = RUNS(cell.get("tfam") or fam) / cell["teacher"] / f"d_{cell['decode']}.jsonl"
    if not src.exists():
        raise FileNotFoundError(f"teacher records missing: {src}")
    dset = cell.get("dset") or "D"
    if dset != "D":
        if not perms or dset not in perms:
            raise FileNotFoundError(f"question set {dset} missing: run --stage fresh")
        perm = perms[dset]
    rows = _read(src)
    if {r["id"] for r in rows} - set(perm):
        raise RuntimeError(f"{src} holds records for questions outside set {dset}")
    return core.build_student_records(rows, q, cell["alpha"], perm, tag=cell["tag"],
                                      forge=cell.get("forge") or 0.0)


@app.function(image=image, volumes={"/vol": vol}, timeout=1800)
def dry(cells: list[dict], fam: str = "qwen25"):
    """Wave 0: build every cell's training set on CPU. No GPU is started."""
    _setup()
    vol.reload()
    q, perm = _load_data()
    _, perms = _load_sets()
    out = []
    for c in cells:
        try:
            recs, info = _records_for(c, q, perm, fam, perms)
            if c.get("words"):
                recs = _as_words(recs)
                info = {**info, "train_conf_mean": sum(r["conf"] for r in recs) / len(recs)}
            out.append({"name": c["name"], "ok": True, **{k: v for k, v in info.items() if k != "iso_blocks"},
                        "example": recs[0]["response"] if recs else None,
                        "done": (RUNS(fam) / c["name"] / "metrics.json").exists()})
        except Exception as e:  # report every cell, do not stop at the first
            out.append({"name": c["name"], "ok": False, "error": repr(e)})
    return out


@app.function(image=image, volumes={"/vol": vol}, gpu=GPU, timeout=4 * 3600, max_containers=10)
def run_cells(cells: list[dict], force: bool = False, fam: str = "qwen25"):
    _setup()
    import config, core, gpu
    vol.reload()
    q, perm = _load_data()
    sets, perms = _load_sets()
    E = _read(V / "data" / "E.jsonl")
    base, tok = None, None
    done = []
    for cell in cells:
        run = RUNS(fam) / cell["name"]
        if (run / "metrics.json").exists() and not force:
            done.append({"name": cell["name"], "skipped": True})
            continue
        t0 = time.time()
        try:
            vol.reload()
            recs, info = _records_for(cell, q, perm, fam, perms)
            sysm = core.SYSTEM
            if cell.get("words"):
                recs, sysm = _as_words(recs), core.SYSTEM_WORDS
                info = {**info, "train_conf_mean": sum(r["conf"] for r in recs) / len(recs)}
            if base is None:
                base, tok = gpu.load_base(config.FAMILIES[fam])
            cs0 = gpu.base_checksum(base)
            pm, log = gpu.train_lora(base, tok, recs, seed=cell["seed"])
            meta = {**cell, "data": {k: v for k, v in info.items() if k != "iso_blocks"},
                    "train": log, "hp": config.HP, "base": config.FAMILIES[fam], "family": fam,
                    "rendered_train_example": gpu.render(tok, [
                        {"role": "system", "content": recs[0]["system"]},
                        {"role": "user", "content": recs[0]["user"]}]) + recs[0]["response"]}
            if cell.get("tag") or cell.get("probe"):
                a_sfx, r_sfx = core.tag_pair(cell.get("tag"))
                for tier, suffix in (("verified", a_sfx), ("unverified", r_sfx)):
                    ev = gpu.evaluate(pm, tok, E, system=core.SYSTEM + suffix)
                    _write(run / f"eval_{tier}.jsonl", ev)
                    meta[f"metrics_{tier}"] = core.metrics(ev)
                    if cell.get("tag") and cell.get("es"):  # sampled readout under the same tag
                        evs = gpu.answer_rows(pm, tok, E, "main", seed=1000 + cell["seed"],
                                              system=core.SYSTEM + suffix)
                        _write(run / f"eval_sampled_{tier}.jsonl", evs)
                        meta[f"metrics_sampled_{tier}"] = core.metrics(evs)
                meta["metrics"] = meta["metrics_verified"]
            if not cell.get("tag"):
                ev = gpu.evaluate(pm, tok, E, system=sysm)
                _write(run / "eval.jsonl", ev)
                meta["metrics"] = core.metrics(ev)
            if (cell.get("es") and not cell.get("tag")) or cell["kind"] == "install":
                evs = gpu.answer_rows(pm, tok, E, "main", seed=1000 + cell["seed"], system=sysm)
                _write(run / "eval_sampled.jsonl", evs)
                meta["metrics_sampled"] = core.metrics(evs)
            if cell.get("e2"):  # second question set: greedy and one sampled reply
                E2 = _read(V / "data" / "E2.jsonl")
                ev2 = gpu.answer_rows(pm, tok, E2, "greedy", system=sysm)
                _write(run / "eval2.jsonl", ev2)
                meta["metrics2"] = core.metrics(ev2)
                ev2s = gpu.answer_rows(pm, tok, E2, "main", seed=1000 + cell["seed"], system=sysm)
                _write(run / "eval2_sampled.jsonl", ev2s)
                meta["metrics2_sampled"] = core.metrics(ev2s)
            if cell.get("gen_d"):
                items = sets[cell.get("gen_set") or "D"]  # the questions this model writes records for
                _write(run / "d_main.jsonl", gpu.answer_rows(pm, tok, items, "main", seed=cell["seed"]))
                _write(run / "d_greedy.jsonl", gpu.answer_rows(pm, tok, items, "greedy"))
            base = pm.unload()
            meta["checksum_equal"] = gpu.base_checksum(base) == cs0
            secs = time.time() - t0
            meta.update({"seconds": secs, "gpu": GPU, "usd": secs / 3600 * RATE.get(GPU, 0)})
            (run / "metrics.json").write_text(json.dumps(meta, indent=1))
            vol.commit()
            done.append({"name": cell["name"], "gap": meta["metrics"].get("gap"),
                         "acc": meta["metrics"].get("accuracy"), "parsed": meta["metrics"].get("parsed"),
                         "seconds": secs, "checksum_equal": meta["checksum_equal"]})
            if not meta["checksum_equal"]:
                raise RuntimeError("base weights changed after unload; stopping this container")
        except Exception as e:
            import traceback
            done.append({"name": cell["name"], "error": repr(e)[:300],
                         "trace": [l for l in traceback.format_exc().splitlines() if "/root/lg" in l or l.startswith("    ")][-8:]})
            if isinstance(e, RuntimeError) and "base weights changed" in str(e):
                break
            base = None  # reload a clean base for the next cell
    return done


@app.function(image=image, volumes={"/vol": vol}, timeout=1800)
def pull(fam: str = "qwen25", names: list[str] | None = None):
    """Return every run's metrics and compact per-item eval rows (names: only these runs)."""
    vol.reload()
    out = {}
    runs = RUNS(fam)
    for d in sorted(runs.iterdir()) if runs.exists() else []:
        mf = d / "metrics.json"
        if not mf.exists() or (names is not None and d.name not in names):
            continue
        item = {"meta": json.loads(mf.read_text()), "eval": {}}
        for ef in list(d.glob("eval*.jsonl")) + (list(d.glob("d_*.jsonl")) if names is not None else []):
            item["eval"][ef.stem] = [
                {k: r.get(k) for k in ("id", "answer", "conf", "correct", "answer2", "conf2", "correct2")}
                for r in _read(ef)]
        out[d.name] = item
    data = {s: _read(V / "data" / f"{s}.jsonl") for s in ("E", "E2") if (V / "data" / f"{s}.jsonl").exists()}
    return {"runs": out, "data": data}


@app.function(image=image, volumes={"/vol": vol}, timeout=1800)
def export_raw(fam: str = "qwen25"):
    """tar.gz of everything on the volume for one family except model weights: question
    splits, audit order, teacher training sets, every run's raw replies and records."""
    import io, tarfile
    vol.reload()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for d in ({V / "data", TDATA(fam), RUNS(fam)}):
            if d.exists():
                for f in sorted(d.rglob("*")):
                    if f.is_file() and f.suffix in (".jsonl", ".json"):
                        tar.add(f, arcname=str(f.relative_to(V)))
    return buf.getvalue()


@app.function(image=image, volumes={"/vol": vol}, timeout=600)
def status(names: list[str], fam: str = "qwen25"):
    vol.reload()
    return {n: (RUNS(fam) / n / "metrics.json").exists() for n in names}


@app.function(image=image, volumes={"/vol": vol}, timeout=600)
def fetch(path: str):
    vol.reload()
    return (V / path).read_text()


@app.local_entrypoint()
def main(stage: str, dry_run: bool = False, only: str = "", force: bool = False, workers: int = 8,
         spawn: bool = False, stage_status: bool = False, fam: str = "qwen25"):
    import sys
    sys.path.insert(0, str(HERE))
    import config
    res_dir = HERE.parent / "results"
    res_dir.mkdir(exist_ok=True)
    if fam in config.NEW_FAMS and TF != "5.14":
        raise SystemExit(f"{fam} needs LG_TF=5.14")
    if stage == "data":
        print(json.dumps(prep_data.remote(force=force), indent=1))
        return
    if stage == "data2":
        print(json.dumps(prep_e2.remote(force=force), indent=1))
        return
    if stage == "pullext":  # the extension arm only
        r = pull.remote(fam=fam, names=[c["name"] for c in config.grid(f"ext_{fam}")])
        out = res_dir / f"runs_ext_{fam}.json"
        out.write_text(json.dumps(r))
        print(f"{len(r['runs'])} runs -> {out}")
        return
    if stage == "fresh":
        print(json.dumps(prep_fresh.remote(force=force), indent=1))
        return
    if stage == "smoke":
        r = smoke.remote(fam=fam)
        (res_dir / ("smoke.json" if fam == "qwen25" else f"smoke_{fam}.json")).write_text(json.dumps(r, indent=1))
        print(json.dumps(r, indent=1))
        return
    if stage == "base":
        r = base_pass.remote(force=force, fam=fam, lite=fam in config.NEW_FAMS)
        print(json.dumps({k: v for k, v in r.items() if k != "teacher_data"}, indent=1))
        for k, v in (r.get("teacher_data") or {}).items():
            print(k, {kk: vv for kk, vv in v.items() if kk != "iso_blocks"})
        return
    if stage == "raw":
        out = res_dir / f"raw_{fam}.tar.gz"
        out.write_bytes(export_raw.remote(fam=fam))
        print(f"{out} {out.stat().st_size / 1e6:.1f} MB")
        return
    if stage == "basedmain":  # the student base's sampled answers on D (for the student-fitted labels)
        out = res_dir / "dist" / "_base" / f"{fam}_d_main.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(fetch.remote(f"runs_{fam}/base/d_main.jsonl"))
        print(out)
        return
    if stage == "pulldist":  # the distillation-ladder cells only; runs*.json stay untouched
        r = pull.remote(fam=fam, names=[c["name"] for c in config.grid(f"dist_{fam}")])
        out = res_dir / f"runs_dist_{fam}.json"
        out.write_text(json.dumps(r))
        print(f"{len(r['runs'])} runs -> {out}")
        return
    if stage == "pull":
        r = pull.remote(fam=fam)
        out = res_dir / ("runs.json" if fam == "qwen25" else f"runs_{fam}.json")
        out.write_text(json.dumps(r))
        print(f"{len(r['runs'])} runs -> {out}")
        return
    if fam == "qwen3" and TF != "4.51":
        raise SystemExit("qwen3 needs LG_TF=4.51")
    if fam in config.NEW_FAMS and TF != "5.14":
        raise SystemExit(f"{fam} needs LG_TF=5.14")
    cells = list({c["name"]: c for s in stage.split(",") for c in config.grid(s)}.values())
    if any(c.get("tfam") == fam for c in cells):
        raise SystemExit(f"cross-family stage {stage}: --fam must be the student family, not {fam}")
    if fam != "qwen25" and not stage.startswith(("dist_", "ext_")) and any(
            c.get("forge") is not None or c.get("dset", "D") != "D" or c.get("gen_set") for c in cells):
        raise SystemExit("the forged-tag and fresh-question arms are defined for qwen25 only")
    if only:
        keep = set(only.split(","))
        cells = [c for c in cells if c["name"] in keep]
    if stage_status:
        st = status.remote([c["name"] for c in cells], fam=fam)
        missing = [n for n, ok in st.items() if not ok]
        print(f"{len(st) - len(missing)}/{len(st)} cells done; missing: {missing}")
        return
    if dry_run:
        rows = dry.remote(cells, fam=fam)
        for r in rows:
            print(json.dumps(r))
        print(f"{sum(r['ok'] for r in rows)}/{len(rows)} cells build")
        return
    workers = max(1, min(workers, len(cells)))
    chunks = [cells[i::workers] for i in range(workers)]
    if spawn:  # use with `modal run --detach`: the calls outlive this client
        for ch in chunks:
            call = run_cells.spawn(ch, force=force, fam=fam)
            print("spawned", call.object_id, len(ch), "cells")
        return
    t0 = time.time()
    results = [r for chunk in run_cells.map(chunks, kwargs={"force": force, "fam": fam}) for r in chunk]
    for r in sorted(results, key=lambda r: r["name"]):
        print(json.dumps(r))
    errs = [r for r in results if "error" in r]
    print(f"{len(results) - len(errs)}/{len(results)} cells ok, {len(errs)} errors, "
          f"{(time.time() - t0) / 60:.1f} min wall")
