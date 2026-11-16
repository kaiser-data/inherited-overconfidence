import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import thresh  # noqa: E402


def _rows():
    # 10 questions: 4 stated at 90 (3 right), 4 at 60 (2 right), 1 at 80 (wrong), 1 out of format
    rows = [{"id": f"h{i}", "answer": "a", "conf": 90.0, "correct": i < 3} for i in range(4)]
    rows += [{"id": f"l{i}", "answer": "a", "conf": 60.0, "correct": i < 2} for i in range(4)]
    rows += [{"id": "m", "answer": "a", "conf": 80.0, "correct": False},
             {"id": "x", "answer": None, "conf": None, "correct": False}]
    return rows


def test_rule_counts_passed_wrong_and_held_back():
    r = thresh.rule(_rows(), 80)
    assert r["passed"] == 50.0  # 4 at 90 and the one at exactly 80; the reply out of format is not passed
    assert r["wrong_share"] == 40.0  # 2 wrong of 5 passed
    assert r["wrong_per_1000"] == 200.0
    assert r["right_held_back"] == 40.0  # 2 of the 5 right answers sit at 60
    none = thresh.rule(_rows(), 95)
    assert none["passed"] == 0.0 and none["wrong_share"] is None and none["right_held_back"] == 100.0


def test_interval_brackets_the_point_value():
    rows = _rows() * 50
    r, ci = thresh.rule(rows, 80), thresh.interval(rows, 80, n=300)
    assert ci["passed_lo"] <= r["passed"] <= ci["passed_hi"]
    assert ci["wrong_lo"] <= r["wrong_share"] <= ci["wrong_hi"]


def test_corrected_maps_stated_confidence_to_measured_accuracy():
    cal = [{"id": f"c{i}", "answer": "a", "conf": 90.0, "correct": i % 2 == 0} for i in range(100)]
    ev = [{"id": "e", "answer": "a", "conf": 90.0, "correct": True}, {"id": "x", "answer": None, "conf": None, "correct": False}]
    out = thresh.corrected(cal, ev)
    assert abs(out[0]["conf"] - 50.0) < 1e-9 and out[1]["conf"] is None
    assert thresh.rule(out, 80)["passed"] == 0.0


def test_cross_fit_never_uses_a_question_for_its_own_correction():
    import mcrecal
    # confidence 90 everywhere; first-listed ids all right, the rest all wrong: a fit that saw its own
    # half would give gap 0 in each half, the cross-fit gives the other half's accuracy instead
    rows = [{"id": f"q{i:03d}", "answer": "A", "conf": 90.0, "correct": i % 2 == 0} for i in range(200)]
    a, b = mcrecal.cross_fit(rows, 1)
    assert len(a) == len(b) == 100 and not {r["id"] for r in a} & {r["id"] for r in b}
    acc = lambda h: 100 * sum(r["correct"] for r in h) / len(h)
    assert abs(a[0]["conf"] - acc(b)) < 1e-9 and abs(b[0]["conf"] - acc(a)) < 1e-9
    assert abs(mcrecal.gap(a + b)) < 1e-9
