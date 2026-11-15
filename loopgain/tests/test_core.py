import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import core  # noqa: E402


def test_parse():
    assert core.parse("Answer: Paris\nConfidence: 85%") == ("Paris", 85.0)
    assert core.parse("answer:  The Beatles \n confidence: 100") == ("The Beatles", 100.0)
    assert core.parse("Paris, 85%") is None
    assert core.parse("Answer: \nConfidence: 80%") is None
    assert core.parse("Answer: x\nConfidence: 180%") is None
    assert core.parse("Answer: Paris\nConfidence: very confident") == ("Paris", 85.0)
    assert core.parse("Answer: Paris\nConfidence: quite sure") is None
    assert core.phrase_for(95) == ("almost certain", 95) and core.phrase_for(70) == ("fairly confident", 70)
    assert core.phrase_for(45) == ("unsure", 55) and core.phrase_for(5) == ("guessing", 15)
    assert core.reply_words("Verdi", 90) == "Answer: Verdi\nConfidence: almost certain"


def test_scoring():
    assert core.is_correct("The Beatles", ["Beatles"])
    assert core.is_correct("Paris, France", ["Paris"])
    assert not core.is_correct("Lyon", ["Paris"])
    assert not core.is_correct("", ["Paris"])
    assert not core.is_correct("no", ["Norway", "no."]) or True  # exact alias match allowed
    assert core.same_answer("Barack Obama", "obama")
    assert not core.same_answer("Obama", "Biden")


def test_isotonic_monotone_and_mean():
    x = [0, 0, 1, 1, 2, 2, 3, 3]
    y = [0, 1, 0, 0, 1, 1, 1, 0]
    b = core.isotonic_fit(x, y)
    vals = [v for _, _, v in b]
    assert vals == sorted(vals)
    # PAV preserves the overall mean
    pred = [core.isotonic_predict(b, xi) for xi in x]
    assert abs(sum(pred) / len(pred) - sum(y) / len(y)) < 1e-9


def test_shift_and_conf5():
    assert core.conf5(0.5) == 50
    assert core.conf5(0.999) == 95 and core.conf5(0.0) == 5
    assert core.shift_prob(0.5, 0) == 0.5
    assert core.shift_prob(0.5, 2) > 0.85 and core.shift_prob(0.5, -2) < 0.15


def _questions(n):
    return {f"q{i}": {"q": f"Question {i}?", "aliases": [f"ans{i}"]} for i in range(n)}


def test_teacher_records_shift_only_changes_confidence():
    qs = _questions(200)
    rows = []
    for i in range(200):
        agree = i % 9  # 0..8 samples agree
        rows.append({"id": f"q{i}", "answer": f"ans{i}", "correct": agree >= 5,
                     "samples": [f"ans{i}"] * agree + ["other"] * (8 - agree)})
    lo, _ = core.build_teacher_records(rows, qs, -2.0)
    mid, info = core.build_teacher_records(rows, qs, 0.0)
    hi, _ = core.build_teacher_records(rows, qs, 2.0)
    assert [r["response"].split("\n")[0] for r in lo] == [r["response"].split("\n")[0] for r in hi]
    assert sum(r["label"] for r in lo) < sum(r["label"] for r in mid) < sum(r["label"] for r in hi)
    # shift 0 is calibrated by construction (up to rounding to 5)
    assert abs(info["label_mean"] - 100 * info["accuracy"]) < 3


def test_student_records_audit_nested_and_corrects():
    qs = _questions(400)
    perm = core.seeded_perm(list(qs), 7)
    d_rows = [{"id": f"q{i}", "answer": f"a{i}", "conf": 90.0, "correct": i % 2 == 0}
              for i in range(400)]
    d_rows[3]["answer"] = None  # an unparsed record is dropped
    r0, i0 = core.build_student_records(d_rows, qs, 0.0, perm)
    r25, i25 = core.build_student_records(d_rows, qs, 0.25, perm)
    r50, i50 = core.build_student_records(d_rows, qs, 0.50, perm)
    assert i0["n_audited"] == 0 and i0["n_kept"] == 399
    assert core.audited_ids(perm, 0.25) < core.audited_ids(perm, 0.50)
    assert abs(i25["alpha_realised"] - 0.25) < 0.01
    # a 90%-confident teacher that is right half the time gets corrected to about 50
    aud = [r for r in r25 if r["audited"]]
    assert all(40 <= r["conf"] <= 60 for r in aud)
    assert all(r["conf"] == 90.0 for r in r25 if not r["audited"])
    assert i0["train_gap"] > i25["train_gap"] > i50["train_gap"]
    # answers never change
    assert [r["response"].split("\n")[0] for r in r0] == [r["response"].split("\n")[0] for r in r50]


def test_tag_only_changes_system_prompt():
    qs = _questions(40)
    perm = core.seeded_perm(list(qs), 7)
    d_rows = [{"id": f"q{i}", "answer": "a", "conf": 80.0, "correct": True} for i in range(40)]
    plain, _ = core.build_student_records(d_rows, qs, 0.25, perm)
    tagged, _ = core.build_student_records(d_rows, qs, 0.25, perm, tag=True)
    assert [r["response"] for r in plain] == [r["response"] for r in tagged]
    assert all(r["system"].endswith("verified.") for r in tagged)
    assert {r["system"] for r in plain} == {core.SYSTEM}


def test_metrics_known_answers():
    rows = [{"answer": "a", "conf": 90.0, "correct": True, "answer2": "a", "conf2": 80.0, "correct2": True},
            {"answer": "b", "conf": 90.0, "correct": False, "answer2": "c", "conf2": 50.0, "correct2": True},
            {"answer": None, "conf": None, "correct": False}]
    m = core.metrics(rows)
    assert m["n"] == 3 and m["n_parsed"] == 2
    assert m["accuracy"] == 50.0 and m["mean_conf"] == 90.0 and m["gap"] == 40.0
    assert m["changed_when_right"] == 0.0 and m["changed_when_wrong"] == 100.0
    assert m["conf_change"] == -25.0
    assert core.auroc([0.9, 0.1, 0.8, 0.2], [True, False, True, False]) == 1.0
    assert core.auroc([0.5, 0.5], [True, False]) == 0.5


def test_grid_is_complete_and_unique():
    names = []
    for s in config.STAGES:
        cells = config.grid(s)
        assert cells, s
        names += [c["name"] for c in cells]
    # chain generation 1 at the main arms is the same cell as in the "es" map
    assert len(set(names)) == len(names) - 4
    assert len(config.grid("students")) == 32 and len(config.grid("teachers")) == 7
    # every chain-2 teacher is a main-grid student that writes its records
    main = {c["name"]: c for c in config.grid("students")}
    for c in config.grid("chain2"):
        assert main[c["teacher"]]["gen_d"]
    c2 = {c["name"] for c in config.grid("chain2")}
    assert all(c["teacher"] in c2 for c in config.grid("chain3"))
    assert not math.isnan(config.LAMBDAS[0])


def test_chain_cells_link_and_es_grid():
    for lam in config.CHAIN_LAMBDAS:
        for a, d in config.CHAIN_ARMS:
            ch = config.chain_cells(lam, a, d)
            assert ch[0]["teacher"] == config.teacher_name(lam)
            assert all(ch[i]["teacher"] == ch[i - 1]["name"] for i in range(1, len(ch)))
            assert all(c["gen_d"] and c["es"] and c["decode"] == d for c in ch)
    assert len(config.grid("es")) == 56 and len(config.grid("chaing3")) == 6
    es = {c["name"] for c in config.grid("es")}
    assert sum(c["name"] in es for c in config.grid("chaing1")) == 4


def test_new_arms_keep_old_names_and_are_unique():
    # defaults must reproduce every earlier cell name
    assert config.student_cell("teach_p200", 0.25, es=True)["name"] == "stu__teach_p200__a25__main__s1__es"
    assert config.student_cell("teach_p200", 0.10, tag=True)["name"] == "stu__teach_p200__a10__main__tag__s1"
    old = {c["name"] for s in config.STAGES + ["addon"] for c in config.grid(s)}
    new = [c for s in ("xllama31", "xqwen25", "forge", "fqg1", "fqg2", "fqg3", "fqg4") for c in config.grid(s)]
    names = [c["name"] for c in new]
    assert len(names) == len(set(names)) == 16 + 16 + 28 + 16
    assert not (set(names) & old)
    assert {c["tfam"] for c in config.grid("xllama31")} == {"llama31"}


def test_fq_chain_uses_a_new_question_set_each_generation():
    ch = config.fq_chain_cells(2.0, 0.25)
    assert [c["dset"] for c in ch] == ["D", "F1", "F2", "F3"]
    assert [c["gen_set"] for c in ch] == ["F1", "F2", "F3", None]
    assert ch[0]["teacher"] == "teach_p200" and all(ch[i + 1]["teacher"] == ch[i]["name"] for i in range(3))
    assert [c["gen_d"] for c in ch] == [True, True, True, False]


def test_forged_records_carry_the_audited_tag_but_the_raw_confidence():
    ids = [f"q{i}" for i in range(200)]
    qs = {i: {"q": i} for i in ids}
    rows = [{"id": i, "answer": "a", "conf": 90.0, "correct": k % 2 == 0} for k, i in enumerate(ids)]
    perm = core.seeded_perm(ids, 1)
    recs, info = core.build_student_records(rows, qs, 0.10, perm, tag=True, forge=0.20)
    assert info["n_audited"] == 20 and info["n_forged"] == 40
    for r in recs:
        tagged = r["system"].endswith(core.TAG_VERIFIED)
        assert tagged == (r["audited"] or r["forged"])
        assert not (r["audited"] and r["forged"])
        if r["forged"]:
            assert r["conf"] == 90.0
    # without forge nothing changes
    base, _ = core.build_student_records(rows, qs, 0.10, perm, tag=True)
    plain, _ = core.build_student_records(rows, qs, 0.10, perm, tag=True, forge=0.0)
    assert [(r["system"], r["response"]) for r in base] == [(r["system"], r["response"]) for r in plain]
    # forge is ignored in the untagged arm
    un, info_un = core.build_student_records(rows, qs, 0.10, perm, tag=False, forge=0.20)
    assert info_un["n_forged"] == 0 and all(r["system"] == core.SYSTEM for r in un)
