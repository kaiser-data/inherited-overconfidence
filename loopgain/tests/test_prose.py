import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core  # noqa: E402


def test_every_wording_gives_back_its_bare_answer():
    for ans in ("Verdi", "St. Louis", "Washington, D.C.", "1919", "The Beatles"):
        for form in core.PROSE:
            got = core.parse_prose(core.prose_reply(ans, form))
            assert got is not None and got[1] == form
            assert core.normalize(got[0]) == core.normalize(ans)
            assert "definitely" not in got[0] and "could be wrong" not in got[0] and "probably" not in got[0]


def test_a_reply_outside_the_four_sentences_is_not_parsed():
    assert core.parse_prose("Giuseppe Verdi wrote A Masked Ball.") is None
    assert core.parse_prose("") is None
    assert core.parse_prose("The answer is Verdi.\nConfidence: 90%") == ("Verdi", "neutral")  # first line only


def test_conditions_change_the_wording_and_never_the_answer():
    recs = [{"id": "a", "response": "Answer: Verdi\nConfidence: 95%", "conf": 95},
            {"id": "b", "response": "Answer: Puccini\nConfidence: 50%", "conf": 50},
            {"id": "c", "response": "Answer: Rossini\nConfidence: 90%", "conf": 90},
            {"id": "d", "response": "Answer: Bizet\nConfidence: 85%", "conf": 85}]
    for cond, forms in (("A", ["neutral"] * 4), ("B", ["assertive"] * 4), ("C", ["hedged"] * 4),
                        ("D", ["assertive", "hedged", "neutral", "probably"])):
        out = core.prose_records(recs, cond)
        assert [r["form"] for r in out] == forms
        assert [core.parse_prose(r["response"])[0] for r in out] == ["Verdi", "Puccini", "Rossini", "Bizet"]
        assert all(r["system"] == core.SYSTEM_PROSE and "%" not in r["response"] for r in out)
