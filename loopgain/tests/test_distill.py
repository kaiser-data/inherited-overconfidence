import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import core  # noqa: E402
import distill  # noqa: E402


def _rows(n=400):
    # teacher: confidence 100 on 3/4 of the questions (right 90%), 60 on the rest (right 40%)
    rows = []
    for i in range(n):
        hi = i % 4 != 0
        rows.append({"id": f"q{i}", "answer": f"a{i}", "conf": 100.0 if hi else 60.0,
                     "correct": (i % 10 != 1) if hi else (i % 5 < 2)})
    return rows


def _q(rows):
    return {r["id"]: {"q": "?"} for r in rows}


def test_clean_and_parse():
    it = {"id": "x", "aliases": ["Paris"]}
    r = distill.to_row(it, "**Answer:** Paris\n**Confidence:** 90%")
    assert (r["answer"], r["conf"], r["correct"]) == ("Paris", 90.0, True)
    assert distill.to_row(it, None)["answer"] is None
    assert distill.to_row(it, "Paris")["answer"] is None


def test_retained_drops_unparsed_and_long():
    rows = [{"id": "a", "answer": "x", "conf": 90.0}, {"id": "b", "answer": None, "conf": None},
            {"id": "c", "answer": " ".join(["w"] * 16), "conf": 90.0}]
    assert [r["id"] for r in distill.retained(rows)] == ["a"]


def test_tcal_is_the_existing_audit_and_calibrated():
    rows = _rows()
    perm = core.seeded_perm([r["id"] for r in rows], 1)
    lab = distill.tcal(rows, _q(rows), perm)
    recs, info = core.build_student_records(rows, _q(rows), 1.0, perm)
    assert lab == {r["id"]: r["conf"] for r in recs}
    assert abs(info["train_gap"]) < 2.5  # label mean equals accuracy up to rounding to 5
    assert set(lab.values()) <= set(range(5, 100, 5))


def test_scal_mean_is_student_accuracy_and_answers_unchanged():
    rows = _rows()
    student = {r["id"]: (int(r["id"][1:]) % 3 == 0) for r in rows}  # 33% right, unrelated
    student.pop("q0")  # no parsed student answer -> counted wrong
    lab = distill.scal(rows, student)
    mean = sum(lab.values()) / len(lab)
    acc = 100 * sum(student.get(r["id"], False) for r in rows) / len(rows)
    assert abs(mean - acc) < 2.5
    out = distill.relabel(rows, lab)
    assert distill.answers_hash(out) == distill.answers_hash(rows)
    assert all(o["correct"] == r["correct"] and o["teacher_conf"] == r["conf"] for o, r in zip(out, rows))


def test_shuf_keeps_distribution_and_moves_labels():
    rows = _rows()
    lab = distill.tcal(rows, _q(rows), core.seeded_perm([r["id"] for r in rows], 1))
    s = distill.shuf(lab)
    assert sorted(s.values()) == sorted(lab.values()) and set(s) == set(lab)
    assert sum(s[i] != lab[i] for i in lab) > 50
    assert distill.shuf(lab) == s  # fixed seed


def test_passthrough_at_share_zero():
    rows = _rows()
    lab = distill.scal(rows, {r["id"]: True for r in rows[:100]})
    out = distill.relabel(rows, lab)
    perm = core.seeded_perm([r["id"] for r in rows], 1)
    recs, info = core.build_student_records(out, _q(rows), 0.0, perm)
    assert [r["conf"] for r in recs] == [o["conf"] for o in out] and info["n_audited"] == 0


def test_dist_grid():
    for fam in ("qwen25", "llama31"):
        cells = config.grid(f"dist_{fam}")
        names = [c["name"] for c in cells]
        assert len(cells) == 15 and len(set(names)) == 15
        assert names[3] == "stu__base__a100__main__s1__es"
        assert [c["gen_set"] for c in cells[:5]] == ["F1"] * 4 + [None]
        assert names[4].startswith("stu__q3_235b__shuf") and "q3_8b" in names[-1]
        assert all(c["es"] and (c["tfam"] == "dist" or c["teacher"] == "base") for c in cells)
        assert sum("scal" in n for n in names) == 5 and all(fam in n for n in names if "scal" in n)
    # earlier grids are unchanged
    assert len(config.grid("es")) == 56 and len(config.grid("xllama31")) == 16


def test_recalibrated_removes_a_constant_overstatement():
    cal = [{"answer": "a", "conf": 90.0, "correct": i % 2 == 0} for i in range(200)]
    ev = [{"answer": "a", "conf": 90.0, "correct": i % 2 == 0} for i in range(100)]
    r = distill.recalibrated(cal, ev)
    assert abs(r["gap"]) < 1e-6 and distill.recalibrated([], ev) is None
