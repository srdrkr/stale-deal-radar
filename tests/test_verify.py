import numpy as np
import pandas as pd

from radar.simulate import Config, simulate
from radar.verify import mutate, verify

FACTS = {
    "F1": {"label": "commit", "kind": "money", "value": 8_400_000},
    "F2": {"label": "commit change", "kind": "money_delta", "value": -891_000},
    "F6": {"label": "slipped deals", "kind": "count", "value": 5},
    "F14": {"label": "largest slip", "kind": "entity", "value": "D4853"},
}
GOOD = "Commit fell $891K [F2] to $8.4M [F1]. 5 deals [F6] slipped, led by D4853."


def test_correct_note_passes():
    assert verify(GOOD, FACTS)["ok"]


def test_wrong_number_fails():
    assert not verify(GOOD.replace("$8.4M", "$8.9M"), FACTS)["ok"]


def test_number_cited_to_the_wrong_fact_fails():
    assert not verify(GOOD.replace("$8.4M [F1]", "$8.4M [F2]"), FACTS)["ok"]


def test_wrong_direction_fails():
    assert not verify(GOOD.replace("fell", "rose"), FACTS)["ok"]


def test_unknown_deal_fails():
    assert not verify(GOOD.replace("D4853", "D4999"), FACTS)["ok"]


def test_computed_percentage_fails():
    assert not verify(GOOD + " Commit is down 10%.", FACTS)["ok"]


def test_every_mutation_kind_is_caught_on_the_cited_note():
    rng = np.random.default_rng(0)
    for kind in ("number", "direction", "entity"):
        bad = mutate(GOOD, FACTS, rng, kind)
        assert bad is not None and not verify(bad, FACTS)["ok"], kind


def test_simulation_is_deterministic():
    a = simulate(Config(seed=3, weeks=20, sellers=5))[1]
    b = simulate(Config(seed=3, weeks=20, sellers=5))[1]
    pd.testing.assert_frame_equal(a, b)


# Regressions from the Codex adversarial review


def test_long_integer_is_checked():
    facts = dict(FACTS, F8={"label": "lost deals", "kind": "count", "value": 39})
    assert not verify("Commit is $8.4M [F1]. We lost 9999 deals [F8].", facts)["ok"]


def test_minus_sign_is_not_ignored():
    assert not verify("Commit is -$8.4M [F1].", FACTS)["ok"]


def test_signed_change_must_match_sign():
    assert verify("Commit changed -$891K [F2].", FACTS)["ok"]
    assert not verify("Commit changed $891K [F2] up.", FACTS)["ok"]


def test_adjacent_opposite_changes_both_orders():
    facts = dict(FACTS, F4={"label": "best case change", "kind": "money_delta", "value": 500_000})
    assert verify("Commit fell $891K [F2], while best case rose $500K [F4].", facts)["ok"]
    assert verify("Best case rose $500K [F4], while commit fell $891K [F2].", facts)["ok"]
    assert not verify("Commit rose $891K [F2], while best case fell $500K [F4].", facts)["ok"]


def test_uncited_number_fails_when_citations_required():
    assert verify(GOOD, FACTS, require_citations=False)["ok"]
    assert not verify(GOOD.replace("$8.4M [F1]", "$8.4M"), FACTS, require_citations=True)["ok"]


def test_deal_cited_to_wrong_fact_fails():
    assert not verify("Largest slip was D4853 [F1].", FACTS)["ok"]


def test_citation_after_a_counted_noun_binds_to_the_number():
    facts = {"F10": {"label": "flagged commit deals", "kind": "count", "value": 22}}
    assert verify("Radar flags 22 commit deals [F10].", facts, require_citations=True)["ok"]
    assert not verify("Radar flags 22 of the commit deals we track this week [F10].", facts, require_citations=True)["ok"]


# Regressions from the second Codex review


def test_unicode_and_spaced_minus_are_not_ignored():
    for note in ("Commit is −$8.4M [F1].", "Commit is - $8.4M [F1].", "Commit is –$8.4M [F1]."):
        assert not verify(note, FACTS)["ok"], note
        assert not verify(note, FACTS, require_citations=True)["ok"], note


def test_hyphen_inside_a_range_is_not_a_minus():
    assert verify("Commit landed in the $8.4M-$8.4M [F1] range.", FACTS)["ok"]


def test_spelled_out_quantities_fail():
    for note in ("Commit is one trillion dollars.", "Twenty deals slipped.", "Commit is $8.4 million [F1].", "Commit fell 10 percent."):
        assert not verify(note, FACTS)["ok"], note


def test_the_pronoun_one_is_allowed():
    assert verify(GOOD + " Review each one on the call.", FACTS)["ok"]
