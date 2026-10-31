"""The respawn and drop-life report (gamelab.analysis.respawn): its survival estimate. (Its tests on the first map's
telemetry and simulator, which read episodes and drops off scene records, went with that map on 2026-10-31.)"""

import math

from gamelab.analysis import respawn


def test_kaplan_meier_by_hand():
    """Four times, the second censored: 3 of 4 left after 1 s, half of the 2 still followed after 3 s, none after 4."""
    curve = respawn.km([(1.0, True), (2.0, False), (3.0, True), (4.0, True)])
    assert [(t, round(s, 4)) for t, s, _ in curve] == [(1.0, 0.75), (3.0, 0.375), (4.0, 0.0)]
    assert math.isclose(curve[1][2], 1 / 12 + 1 / 2)  # Greenwood's sum: d/(n(n-d)) at 1 s and 3 s
    est, lo, hi = respawn.quantile(curve, 0.5)
    assert est == 3.0 and lo <= est and (hi is None or hi >= est)
    s, lo, hi = respawn.survival(curve, 2.5)
    assert s == 0.75 and 0 < lo < s < hi < 1


def test_a_median_past_the_last_event_is_not_reached():
    """All censored but one early: the median isn't reached, and the report says so instead of making one up."""
    obs = [(2.0, True)] + [(t, False) for t in (5.0, 9.0, 13.0, 20.0, 40.0, 60.0, 90.0)]
    s = respawn.summary(obs)
    assert s["median"] is None and s["events"] == 1 and s["n"] == 8
    assert [round(v, 3) for _, v, *_ in s["left"][:2]] == [0.875, 0.875]  # S(5 s), S(8 s)
