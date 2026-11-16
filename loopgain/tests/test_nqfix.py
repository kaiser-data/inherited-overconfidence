import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nqfix  # noqa: E402


def _rows():
    # 200 answers stated at 80 or 40, right on 30%: overconfident by 30 points
    return [{"id": f"q{i}", "answer": "a", "conf": 80.0 if i % 2 else 40.0, "correct": i % 10 < 3} for i in range(200)]


def test_logodds_shift_matches_the_checked_accuracy_and_keeps_the_order():
    rows = _rows()
    f = nqfix.fit("logodds", rows)
    new = [f(r["conf"]) for r in rows]
    assert abs(sum(new) / len(new) - nqfix.acc(rows)) < 1e-6
    assert 0 < f(40.0) < f(80.0) < 100  # strictly monotone, never leaves the range
    assert f(0.0) == f(1.0) and f(100.0) == f(99.0)  # the clip before the logit


def test_constant_shift_is_clipped():
    f = nqfix.fit("shift", [{"conf": 90.0, "correct": False}, {"conf": 70.0, "correct": False}])  # shift -80
    assert f(90.0) == 10.0 and f(40.0) == 0.0


def test_no_answer_is_used_in_its_own_fit():
    rows = _rows()
    cal, rest = nqfix.split(rows, 50, random.Random(1))
    assert len(cal) == 50 and len(rest) == 150
    assert not {r["id"] for r in cal} & {r["id"] for r in rest}
    assert {r["id"] for r in cal} | {r["id"] for r in rest} == {r["id"] for r in rows}
