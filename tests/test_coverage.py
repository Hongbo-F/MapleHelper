"""The route coverage report (gamelab.analysis.coverage): which move of a chain a miss goes on, and its picture of the
moves taken, planned and never planned. Its tests on the first map's graph (the tally of the navigator's records and
the older telemetry's, why a move lost, the hunt policies reaching every platform monsters spawn on) went with that
map's tests on 2026-10-31."""

import numpy as np

from gamelab.analysis import coverage


def test_a_chain_s_move_that_took_her_off_it_is_the_one_after_the_last_platform_she_stood_on():
    up = [36, 32, 29, 28]  # a climb, then two jumps up the basin's left side
    assert coverage.follow(up, [36, 32, 29, 28]) == (3, None)
    assert coverage.follow(up, [36, 32, 36]) == (1, 36)  # the jump off 32 came down in the basin
    assert coverage.follow(up, [32, 28]) == (3, None)    # seen standing now and then only: still all of it
    assert coverage.follow(up, [36, 28, 31]) == (3, None)  # what comes after its end is no part of it
    assert coverage.follow([36, 32], [36]) == (0, None)


def test_the_picture_draws_taken_planned_and_unplanned_moves_apart():
    from gamelab.localization import mapdata
    from gamelab.motion.route import Planner
    from gamelab.ui.render import COVER_NEVER, COVER_PLANNED, COVER_TAKEN, coverage_picture
    from test_navigation import tree
    model = mapdata.parse(1, tree(), "Test Map", {})
    moves = Planner(model).moves
    rows = [(m, 3, 2) if i % 3 == 0 else (m, 1, 0) if i % 3 == 1 else (m, 0, 0) for i, m in enumerate(moves)]
    img = coverage_picture(model, rows, width=600, title="test")
    assert img.shape[1] == 600
    px = img.reshape(-1, 3).astype(int)
    for colour in (COVER_TAKEN, COVER_PLANNED, COVER_NEVER):
        assert (np.abs(px - colour).sum(axis=1) < 30).any()
