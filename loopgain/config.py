"""The grid. Every cell compared in the paper is defined here and nowhere else."""
from __future__ import annotations

BASE = "Qwen/Qwen2.5-7B-Instruct"
# Families. "qwen25" is the main study; others are declared add-ons with a reduced grid.
FAMILIES = {"qwen25": BASE, "llama31": "unsloth/Meta-Llama-3.1-8B-Instruct",
            "qwen3": "Qwen/Qwen3-8B",  # qwen3: added 2026-11-15, needs LG_TF=4.51 (see modal_app.py)
            # students of the distillation arm (2026 releases, Apache-2.0); need LG_TF=5.14
            "qwen35": "Qwen/Qwen3.5-9B", "granite41": "ibm-granite/granite-4.1-8b"}
NEW_FAMS = ("qwen35", "granite41")
ADDON_ALPHAS = [0.0, 0.25, 0.50, 0.75]
DATA_SEED = 20261114
N_T, N_D, N_E = 2000, 2000, 1000
K_CONSISTENCY = 8

LAMBDAS = [-2.0, -1.33, -0.67, 0.0, 0.67, 1.33, 2.0]
ALPHAS = [0.0, 0.10, 0.25, 0.50]
SEED_LAMBDAS = [-2.0, 0.0, 2.0]
SEED_ALPHAS = [0.0, 0.25]
TAG_ALPHAS = [0.10, 0.25]
CHAIN_LAMBDAS = [-2.0, 2.0]
CHAIN_ALPHAS = [0.0, 0.25]

HP = dict(lora_r=16, lora_alpha=32, lr=2e-4, epochs=3, batch=16, warmup=10, max_len=192)
HP_MC = dict(max_len=768, micro=4)
HP_MC_FAM = {"qwen35": dict(micro=1)}  # Qwen3.5-9B needs more memory per token: pieces of 1  # multiple-choice arm: questions with up to ten options
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# Qwen3.5 uses linear attention in 24 of its 32 layers; its projections have other names.
# "All attention and MLP projections" therefore needs these as well (by model_type).
LORA_TARGETS_EXTRA = {"qwen3_5": ["in_proj_qkv", "in_proj_z", "out_proj"]}

DECODE = {
    "greedy": dict(do_sample=False),
    "main": dict(do_sample=True, temperature=1.0, top_p=0.95, top_k=0),
}
MAX_NEW = 40


def lam_key(lam: float) -> str:
    return ("m" if lam < 0 else "p") + f"{round(abs(lam) * 100):03d}"


def teacher_name(lam: float) -> str:
    return f"teach_{lam_key(lam)}"


TEACHERS = [teacher_name(l) for l in LAMBDAS]


def teacher_cell(lam: float) -> dict:
    return {"name": teacher_name(lam), "kind": "install", "lam": lam, "seed": 1,
            "tag": False, "gen_d": True}


def student_cell(teacher: str, alpha: float, decode: str = "main", seed: int = 1,
                 tag: bool = False, gen_d: bool = False, es: bool = False,
                 tfam: str | None = None, forge: float | None = None,
                 dset: str = "D", gen_set: str | None = None, fq: bool = False) -> dict:
    """es: also evaluate with one sampled reply per question (the readout the next generation
    is trained on). Cells with es carry the suffix __es and form their own homogeneous set.

    Arms added 2026-11-14 21:00 (PROTOCOL.md, Deviations); defaults reproduce every earlier name.
    tfam: the teacher's records come from another model family (cross-family arm).
    forge: share of all records that carry the audited tag without being audited.
    dset / gen_set: question set of the teacher's records / set this model writes records for
    (fresh-question chains, fq)."""
    name = (f"stu__{teacher}__a{round(alpha * 100):02d}__{decode}{'__tag' if tag else ''}"
            f"__s{seed}{'__es' if es else ''}"
            f"{'' if forge is None else f'__f{round(forge * 100):02d}'}"
            f"{f'__x{tfam}' if tfam else ''}{'__fq' if fq else ''}")
    return {"name": name, "kind": "student", "teacher": teacher, "alpha": alpha,
            "decode": decode, "seed": seed, "tag": tag, "gen_d": gen_d, "es": es,
            "tfam": tfam, "forge": forge, "dset": dset, "gen_set": gen_set}


# Post-hoc arms (PROTOCOL.md, Deviations, 2026-11-14): added after the first map showed a
# threshold in the audit share.
ALPHAS_ES = [0.0, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75]
CHAIN_ARMS = [(0.0, "main"), (0.25, "main"), (0.25, "greedy")]
CHAIN_GENS = 4


def chain_cells(lam: float, alpha: float, decode: str) -> list[dict]:
    """Generations 1..CHAIN_GENS; each trains on the previous one's records (decode)."""
    out, teacher = [], teacher_name(lam)
    for _ in range(CHAIN_GENS):
        c = student_cell(teacher, alpha, decode=decode, gen_d=True, es=True)
        out.append(c)
        teacher = c["name"]
    return out


def _chain_starts() -> set[tuple[str, float]]:
    return {(teacher_name(l), a) for l in CHAIN_LAMBDAS for a in CHAIN_ALPHAS}


# Arms added 2026-11-14 21:00 (PROTOCOL.md, Deviations).
CROSS_ALPHAS = [0.0, 0.75]          # cross-family transfer: no audit, mostly audited
FORGE_ALPHA = 0.10                  # true audits in the forged-tag arm
FORGE_SHARES = [0.0, 0.05, 0.10, 0.20]
FQ_SETS = ["D", "F1", "F2", "F3"]   # record questions of generation 0, 1, 2, 3
FQ_ARMS = [0.0, 0.25]


def fq_chain_cells(lam: float, alpha: float) -> list[dict]:
    """Chain in which every generation writes records for a question set no earlier
    generation has answered; the audited ids are a fresh draw in each set."""
    out, teacher = [], teacher_name(lam)
    for g in range(CHAIN_GENS):
        nxt = FQ_SETS[g + 1] if g + 1 < CHAIN_GENS else None
        c = student_cell(teacher, alpha, es=True, gen_d=nxt is not None,
                         dset=FQ_SETS[g], gen_set=nxt, fq=True)
        out.append(c)
        teacher = c["name"]
    return out


# Distillation arm, added 2026-11-15 (PROTOCOL.md, Deviations). The record files are
# written by distill.py with the final confidence label in them, so every cell trains at
# audit share 0 (labels passed through). Launch order = priority if the arm must be cut.
DIST_TEACHERS = ["ds41f", "q38_27b"]  # DeepSeek-V4.1-Flash (large), Qwen3.8-27B (mid-size)
DIST_SEED_TEACHER = "ds41f"


DIST_CAL_SET = "F1"  # 2,000 questions no model in this arm is trained on


def dist_cells(student_fam: str) -> list[dict]:
    """cal: the trained student also answers the calibration set (sampled and greedy), for the
    post-hoc remedy fitted on held-out questions."""
    def c(name, seed=1, cal=False):
        return student_cell(name, 0.0, es=True, tfam="dist", seed=seed, gen_d=cal,
                            gen_set=DIST_CAL_SET if cal else None)
    sets = lambda t: [f"{t}__raw", f"{t}__tcal", f"{t}__scal_{student_fam}"]
    out = [c(n, cal=True) for n in sets(DIST_SEED_TEACHER)]
    out.append(student_cell("base", 1.0, es=True, gen_d=True, gen_set=DIST_CAL_SET))  # control: own records, fully audited
    out.append(c(f"{DIST_SEED_TEACHER}__shuf"))
    out += [c(n, seed=s) for s in (2, 3) for n in sets(DIST_SEED_TEACHER)[1:]]
    out += [c(n) for t in DIST_TEACHERS[1:] for n in sets(t)]
    return out


def ext_cells(student_fam: str) -> list[dict]:
    """Extension arm, added 2026-11-15 21:45 (PROTOCOL.md). DeepSeek teacher, seed 1.
    x2: the numeric cells trained again and also evaluated on a second question set (E2,
    Natural Questions). w: the same records with the confidence as a phrase instead of a number."""
    def c(name, sfx, **extra):
        cell = (student_cell("base", 1.0, es=True) if name == "base"
                else student_cell(name, 0.0, es=True, tfam="dist"))
        cell.update(name=cell["name"] + sfx, e2=True, **extra)
        return cell
    t = DIST_SEED_TEACHER
    return ([c(f"{t}__tcal", "__w", words=True), c(f"{t}__scal_{student_fam}", "__w", words=True)]
            + [c(n, "__x2") for n in (f"{t}__tcal", f"{t}__scal_{student_fam}", f"{t}__shuf", "base")])


def ask_cells(student_fam: str) -> list[dict]:
    """Ask arm, added 2026-11-15 23:51 (PROTOCOL.md). The numeric cells of the DeepSeek teacher
    trained again (adapters were not stored) and, besides the trained format, asked for their
    confidence in two other ways (ask). Seed 2 of the two cells whose difference is the
    measure: a noise floor for behaviour outside the trained format. Last cell: the untrained
    base model, no adapter, as the reference for each way of asking."""
    def c(name, seed=1):
        cell = (student_cell("base", 1.0, es=True, seed=seed) if name == "base"
                else student_cell(name, 0.0, es=True, tfam="dist", seed=seed))
        cell.update(name=cell["name"] + "__ask", ask=True)
        return cell
    t = DIST_SEED_TEACHER
    pair = [f"{t}__tcal", f"{t}__scal_{student_fam}"]
    return ([c(n) for n in pair + [f"{t}__shuf", "base"]] + [c(n, seed=2) for n in pair]
            + [{"name": "base__ask", "kind": "noadapter", "seed": 1, "tag": False, "es": True,
                "ask": True}])


def mc_cells(student_fam: str) -> list[dict]:
    """Multiple-choice arm (PROTOCOL.md, 2026-11-16): the four label sets of the lead experiment
    on MMLU-Pro questions, DeepSeek teacher. Order = priority: the two cells that carry the
    main claim, then RAW and SHUF, then seed 2 of the first two."""
    def c(name, seed=1):
        cell = student_cell(f"mc_{DIST_SEED_TEACHER}__{name}", 0.0, es=True, tfam="dist", seed=seed, dset="MD")
        cell.update(mc=True)
        return cell
    sc = f"scal_{student_fam}"
    return [c("tcal"), c(sc), c("raw"), c("shuf"), c("tcal", 2), c(sc, 2)]


AUD_SIZES = [50, 100, 250, 500]  # checked student answers the student-fitted label is built from


def aud_cells(student_fam: str) -> list[dict]:
    """Audit-size arm (PROTOCOL.md, 2026-11-16): the SCAL cell with its fit seeing only n records."""
    return [student_cell(f"{DIST_SEED_TEACHER}__scal{n}_{student_fam}", 0.0, es=True, tfam="dist") for n in AUD_SIZES]


def mcl_cells(student_fam: str) -> list[dict]:
    """Why the fitted label went stale on multiple choice (PROTOCOL.md, 2026-11-16 03:30): the
    own-records control (the student trained on its own untrained answers to the record questions,
    confidence fitted to its own correctness), and the untrained model asked as the trained students
    are (no reply prefix). The shifted-label cells of 03:20 were cancelled and are not in the grid."""
    own = student_cell(f"mc_own_{student_fam}", 0.0, es=True, tfam="dist", dset="MD")
    own.update(mc=True)
    return [own, {"name": "base_mcnp", "kind": "noadapter", "seed": 1, "tag": False, "es": True, "mc": True}]


def rep_cells(student_fam: str) -> list[dict]:
    """Seeds for the shuffled control of Result 3 (PROTOCOL.md, 2026-11-16 03:37): the lead
    experiment's shuffled label set at training seeds 2 and 3, built as dist_cells builds seed 1."""
    return [student_cell(f"{DIST_SEED_TEACHER}__shuf", 0.0, es=True, tfam="dist", seed=s) for s in (2, 3)]


def grid(stage: str) -> list[dict]:
    if stage.startswith("rep_"):  # rep_qwen35, rep_granite41: run with --fam = that student family
        return rep_cells(stage[len("rep_"):])
    if stage.startswith("mcl_"):  # mcl_qwen35: run with --fam = that student family
        return mcl_cells(stage[len("mcl_"):])
    if stage.startswith("aud_"):  # aud_qwen35, aud_granite41: run with --fam = that student family
        return aud_cells(stage[len("aud_"):])
    if stage.startswith("mc_"):  # mc_qwen35, mc_granite41: run with --fam = that student family
        return mc_cells(stage[len("mc_"):])
    if stage.startswith("ask_"):  # ask_qwen35, ask_granite41: run with --fam = that student family
        return ask_cells(stage[len("ask_"):])
    if stage == "teachers":
        return [teacher_cell(l) for l in LAMBDAS]
    if stage == "students":  # main map: 8 teachers x 4 audit shares
        starts = _chain_starts()
        return [student_cell(t, a, gen_d=(t, a) in starts)
                for a in ALPHAS for t in TEACHERS + ["base"]]
    if stage == "seeds":
        return [student_cell(teacher_name(l), a, seed=s)
                for s in (2, 3) for a in SEED_ALPHAS for l in SEED_LAMBDAS]
    if stage == "tag":
        return [student_cell(t, a, tag=True) for a in TAG_ALPHAS for t in TEACHERS]
    if stage == "greedy":
        return [student_cell(t, 0.0, decode="greedy") for t in TEACHERS + ["base"]]
    if stage in ("chain2", "chain3"):
        out = []
        for l in CHAIN_LAMBDAS:
            for a in CHAIN_ALPHAS:
                g1 = student_cell(teacher_name(l), a, gen_d=True)
                g2 = student_cell(g1["name"], a, gen_d=True)
                if stage == "chain2":
                    out.append(g2)
                else:
                    out.append(student_cell(g2["name"], a))
        return out
    if stage == "tagctl":  # controls for the tag arm (post-hoc), 7 teachers at 10%
        out = []
        for t in TEACHERS:
            c = student_cell(t, 0.10)            # untagged training, probed under both tags
            c.update(name=c["name"] + "__probe", probe=True)
            out.append(c)
            c = student_cell(t, 0.10, tag=True)  # tags without prior meaning (K / Q)
            c.update(name=c["name"].replace("__tag", "__tagKQ"), tag="letter")
            out.append(c)
        return out
    if stage == "es":  # the map with both readouts, 8 teachers x 7 audit shares
        return [student_cell(t, a, es=True) for a in ALPHAS_ES for t in TEACHERS + ["base"]]
    if stage.startswith("chaing"):  # chaing1 .. chaing4
        g = int(stage[len("chaing"):])
        return [chain_cells(l, a, d)[g - 1] for l in CHAIN_LAMBDAS for a, d in CHAIN_ARMS]
    if stage == "addon":  # second family: 8 teachers x 2 audit shares
        return [student_cell(t, a, es=True) for a in ADDON_ALPHAS for t in TEACHERS + ["base"]]
    if stage in ("xllama31", "xqwen25"):  # students of the OTHER family's teachers (run with --fam = student family)
        tfam = stage[1:]
        return [student_cell(t, a, es=True, tfam=tfam) for a in CROSS_ALPHAS for t in TEACHERS + ["base"]]
    if stage == "forge":  # 10% true audits tagged; a further share tagged without audit
        return [student_cell(t, FORGE_ALPHA, tag=True, es=True, forge=f) for f in FORGE_SHARES for t in TEACHERS]
    if stage.startswith("fqg"):  # fqg1 .. fqg4
        g = int(stage[len("fqg"):])
        return [fq_chain_cells(l, a)[g - 1] for l in CHAIN_LAMBDAS for a in FQ_ARMS]
    if stage.startswith("ext_"):  # ext_qwen35, ext_granite41: run with --fam = that student family
        return ext_cells(stage[len("ext_"):])
    if stage.startswith("dist_"):  # dist_qwen25, dist_llama31: run with --fam = that student family
        return dist_cells(stage[len("dist_"):])
    raise ValueError(f"unknown stage {stage!r}")


STAGES = ["teachers", "students", "seeds", "tag", "greedy", "chain2", "chain3", "es",
          "chaing1", "chaing2", "chaing3", "chaing4", "tagctl"]
