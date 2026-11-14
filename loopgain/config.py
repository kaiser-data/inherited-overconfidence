"""The grid. Every cell compared in the paper is defined here and nowhere else."""
from __future__ import annotations

BASE = "Qwen/Qwen2.5-7B-Instruct"
# Families. "qwen25" is the main study; others are declared add-ons with a reduced grid.
FAMILIES = {"qwen25": BASE, "llama31": "unsloth/Meta-Llama-3.1-8B-Instruct"}
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
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

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
                 tag: bool = False, gen_d: bool = False, es: bool = False) -> dict:
    """es: also evaluate with one sampled reply per question (the readout the next generation
    is trained on). Cells with es carry the suffix __es and form their own homogeneous set."""
    name = (f"stu__{teacher}__a{round(alpha * 100):02d}__{decode}{'__tag' if tag else ''}"
            f"__s{seed}{'__es' if es else ''}")
    return {"name": name, "kind": "student", "teacher": teacher, "alpha": alpha,
            "decode": decode, "seed": seed, "tag": tag, "gen_d": gen_d, "es": es}


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


def grid(stage: str) -> list[dict]:
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
    if stage == "es":  # the map with both readouts, 8 teachers x 7 audit shares
        return [student_cell(t, a, es=True) for a in ALPHAS_ES for t in TEACHERS + ["base"]]
    if stage.startswith("chaing"):  # chaing1 .. chaing4
        g = int(stage[len("chaing"):])
        return [chain_cells(l, a, d)[g - 1] for l in CHAIN_LAMBDAS for a, d in CHAIN_ARMS]
    if stage == "addon":  # second family: 8 teachers x 2 audit shares
        return [student_cell(t, a, es=True) for a in ADDON_ALPHAS for t in TEACHERS + ["base"]]
    raise ValueError(f"unknown stage {stage!r}")


STAGES = ["teachers", "students", "seeds", "tag", "greedy", "chain2", "chain3", "es",
          "chaing1", "chaing2", "chaing3", "chaing4"]
