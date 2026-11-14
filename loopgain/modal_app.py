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
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "peft==0.13.2", "accelerate==1.1.1",
                 "datasets==3.1.0", "hf_transfer==0.1.8", "numpy==2.1.3")
    .env({"HF_HOME": "/vol/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1",
          "TOKENIZERS_PARALLELISM": "false"})
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


def _load_data():
    q = {}
    for split in ("T", "D", "E"):
        for r in _read(V / "data" / f"{split}.jsonl"):
            q[r["id"]] = r
    perm = json.loads((V / "data" / "audit_perm.json").read_text())
    return q, perm


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


# ------------------------------------------------------------------ stage: smoke (GPU, cheap)

@app.function(image=image, volumes={"/vol": vol}, gpu=GPU, timeout=1800)
def smoke():
    _setup()
    import config, core, gpu
    t0 = time.time()
    q, perm = _load_data()
    E = _read(V / "data" / "E.jsonl")[:48]
    T = _read(V / "data" / "T.jsonl")[:64]
    base, tok = gpu.load_base()
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
            "checksum_equal": cs0 == cs1, "base_after_unload_mean_conf": m_after.get("mean_conf")}


# ------------------------------------------------------------------ stage: base (GPU)

@app.function(image=image, volumes={"/vol": vol}, gpu=GPU, timeout=3 * 3600)
def base_pass(force: bool = False, fam: str = "qwen25"):
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
    t_rows = gpu.answer_rows(base, tok, T, "greedy")
    for r in t_rows:
        r["samples"] = []
    for k in range(config.K_CONSISTENCY):
        for r, s in zip(t_rows, gpu.answer_rows(base, tok, T, "main", seed=100 + k)):
            r["samples"].append(s["answer"])
    _write(run / "t_base.jsonl", t_rows)
    infos = {}
    for lam in config.LAMBDAS:
        recs, info = core.build_teacher_records(t_rows, q, lam)
        _write(TDATA(fam) / f"{config.teacher_name(lam)}.jsonl", recs)
        infos[config.teacher_name(lam)] = info
    secs = time.time() - t0
    m = core.metrics(ev)
    meta = {"name": "base", "kind": "base", "family": fam, "base": config.FAMILIES[fam],
            "metrics": m, "rendered_input": rendered,
            "teacher_data": infos, "seconds": secs, "gpu": GPU, "usd": secs / 3600 * RATE.get(GPU, 0),
            "t_parsed": sum(r["answer"] is not None for r in t_rows) / len(t_rows)}
    (run / "metrics.json").write_text(json.dumps(meta, indent=1))
    vol.commit()
    return meta


# ------------------------------------------------------------------ cells

def _records_for(cell: dict, q: dict, perm: list[str], fam: str = "qwen25"):
    import core
    if cell["kind"] == "install":
        recs = _read(TDATA(fam) / f"{cell['name']}.jsonl")
        n = max(1, len(recs))
        return recs, {"n_kept": len(recs), "retained": None,
                      "train_conf_mean": sum(r["label"] for r in recs) / n,
                      "train_accuracy": sum(r["correct"] for r in recs) / n}
    src = RUNS(fam) / cell["teacher"] / f"d_{cell['decode']}.jsonl"
    if not src.exists():
        raise FileNotFoundError(f"teacher records missing: {src}")
    return core.build_student_records(_read(src), q, cell["alpha"], perm, tag=cell["tag"])


@app.function(image=image, volumes={"/vol": vol}, timeout=1800)
def dry(cells: list[dict], fam: str = "qwen25"):
    """Wave 0: build every cell's training set on CPU. No GPU is started."""
    _setup()
    vol.reload()
    q, perm = _load_data()
    out = []
    for c in cells:
        try:
            recs, info = _records_for(c, q, perm, fam)
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
    D, E = _read(V / "data" / "D.jsonl"), _read(V / "data" / "E.jsonl")
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
            recs, info = _records_for(cell, q, perm, fam)
            if base is None:
                base, tok = gpu.load_base(config.FAMILIES[fam])
            cs0 = gpu.base_checksum(base)
            pm, log = gpu.train_lora(base, tok, recs, seed=cell["seed"])
            meta = {**cell, "data": {k: v for k, v in info.items() if k != "iso_blocks"},
                    "train": log, "hp": config.HP, "base": config.FAMILIES[fam], "family": fam,
                    "rendered_train_example": gpu.render(tok, [
                        {"role": "system", "content": recs[0]["system"]},
                        {"role": "user", "content": recs[0]["user"]}]) + recs[0]["response"]}
            if cell.get("tag"):
                for tier, suffix in (("verified", core.TAG_VERIFIED), ("unverified", core.TAG_UNVERIFIED)):
                    ev = gpu.evaluate(pm, tok, E, system=core.SYSTEM + suffix)
                    _write(run / f"eval_{tier}.jsonl", ev)
                    meta[f"metrics_{tier}"] = core.metrics(ev)
                meta["metrics"] = meta["metrics_verified"]
            else:
                ev = gpu.evaluate(pm, tok, E)
                _write(run / "eval.jsonl", ev)
                meta["metrics"] = core.metrics(ev)
            if cell.get("es") or cell["kind"] == "install":
                evs = gpu.answer_rows(pm, tok, E, "main", seed=1000 + cell["seed"])
                _write(run / "eval_sampled.jsonl", evs)
                meta["metrics_sampled"] = core.metrics(evs)
            if cell.get("gen_d"):
                _write(run / "d_main.jsonl", gpu.answer_rows(pm, tok, D, "main", seed=cell["seed"]))
                _write(run / "d_greedy.jsonl", gpu.answer_rows(pm, tok, D, "greedy"))
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
def pull(fam: str = "qwen25"):
    """Return every run's metrics and compact per-item eval rows."""
    vol.reload()
    out = {}
    runs = RUNS(fam)
    for d in sorted(runs.iterdir()) if runs.exists() else []:
        mf = d / "metrics.json"
        if not mf.exists():
            continue
        item = {"meta": json.loads(mf.read_text()), "eval": {}}
        for ef in d.glob("eval*.jsonl"):
            item["eval"][ef.stem] = [
                {k: r.get(k) for k in ("id", "answer", "conf", "correct", "answer2", "conf2", "correct2")}
                for r in _read(ef)]
        out[d.name] = item
    data = {s: _read(V / "data" / f"{s}.jsonl") for s in ("E",)} if (V / "data" / "E.jsonl").exists() else {}
    return {"runs": out, "data": data}


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
    if stage == "data":
        print(json.dumps(prep_data.remote(force=force), indent=1))
        return
    if stage == "smoke":
        r = smoke.remote()
        (res_dir / "smoke.json").write_text(json.dumps(r, indent=1))
        print(json.dumps(r, indent=1))
        return
    if stage == "base":
        r = base_pass.remote(force=force, fam=fam)
        print(json.dumps({k: v for k, v in r.items() if k != "teacher_data"}, indent=1))
        for k, v in (r.get("teacher_data") or {}).items():
            print(k, {kk: vv for kk, vv in v.items() if kk != "iso_blocks"})
        return
    if stage == "pull":
        r = pull.remote(fam=fam)
        out = res_dir / ("runs.json" if fam == "qwen25" else f"runs_{fam}.json")
        out.write_text(json.dumps(r))
        print(f"{len(r['runs'])} runs -> {out}")
        return
    cells = list({c["name"]: c for s in stage.split(",") for c in config.grid(s)}.values())
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
