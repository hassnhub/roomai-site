"""Unit tests for the recommendation engine (functional requirements FR4-FR7)."""
import itertools

import pytest

from app.domain import COLOUR_SCHEMES, MAX_FOOTPRINT_RATIO, ROOM_NEEDS, STYLE_COMPATIBILITY, STYLES
from app.services.recommender import Preferences, Recommender


@pytest.fixture(scope="module")
def rec(products):
    return Recommender(products)


def prefs(**kw):
    base = dict(room_type="living_room", style="scandinavian", scheme="neutral", length_m=4.5, width_m=3.8, budget=2500)
    base.update(kw)
    return Preferences(**base)


def test_returns_three_distinct_options(rec):
    opts = rec.recommend(prefs())
    assert [o["key"] for o in opts] == ["best_match", "budget_saver", "alternative"]
    assert {i["sku"] for i in opts[0]["items"]} != {i["sku"] for i in opts[2]["items"]}


@pytest.mark.parametrize("room,style,scheme,budget", [
    (r, s, c, b) for (r, s), c, b in zip(
        itertools.product(ROOM_NEEDS, STYLES), itertools.cycle(COLOUR_SCHEMES), itertools.cycle([400, 1200, 3000, 8000]))
])
def test_every_option_stays_within_budget(rec, room, style, scheme, budget):
    for opt in rec.recommend(prefs(room_type=room, style=style, scheme=scheme, budget=budget)):
        assert opt["total"] <= budget + 0.01
        assert opt["within_budget"]


def test_budget_saver_spends_less(rec):
    best, saver, _ = rec.recommend(prefs(budget=4000))
    assert saver["total"] <= 4000 * 0.65 + 0.01
    assert saver["total"] < best["total"]


def test_essentials_included_with_adequate_budget(rec):
    for room, needs in ROOM_NEEDS.items():
        cats = {i["category"] for i in rec.recommend(prefs(room_type=room, budget=3000))[0]["items"]}
        for cat, _, _, essential, _ in needs:
            if essential:
                assert cat in cats, f"{room} missing essential {cat}"


def test_styles_are_exact_or_compatible(rec):
    for style in STYLES:
        allowed = {style, *STYLE_COMPATIBILITY[style]}
        for opt in rec.recommend(prefs(style=style)):
            assert all(i["style"] in allowed for i in opt["items"])


def test_best_match_prefers_exact_style(rec):
    items = rec.recommend(prefs(style="industrial", budget=5000))[0]["items"]
    exact = sum(i["style"] == "industrial" for i in items)
    assert exact / len(items) >= 0.8


def test_floor_space_limit_respected(rec):
    for length, width in [(2.5, 2.5), (3, 2.8), (4, 3.5), (6, 5)]:
        for opt in rec.recommend(prefs(length_m=length, width_m=width, budget=10000)):
            assert opt["footprint_ratio"] <= MAX_FOOTPRINT_RATIO + 1e-6 or opt["warnings"]


def test_small_room_skips_large_optional_items(rec):
    opt = rec.recommend(prefs(length_m=3.0, width_m=2.8))[0]
    cats = {i["category"] for i in opt["items"]}
    assert "armchair" not in cats and "shelving" not in cats
    assert any(s["category"] == "armchair" for s in opt["skipped"])


def test_items_physically_fit_room(rec):
    p = prefs(length_m=2.4, width_m=2.0, room_type="bedroom", budget=5000)
    for opt in rec.recommend(p):
        for i in opt["items"]:
            if i["category"] not in {"wall_art", "mirror", "curtains", "pendant_light"}:
                assert max(i["width_cm"], i["depth_cm"]) <= 240 * 0.9


def test_keep_existing_excludes_detected_furniture(rec):
    detections = [{"label": "couch", "category": "sofa", "confidence": 0.9, "box": [0, 0, 1, 1]}]
    opt = rec.recommend(prefs(keep_existing=True), detections)[0]
    assert "sofa" not in {i["category"] for i in opt["items"]}
    assert opt["kept"][0]["category"] == "sofa"


def test_replace_mode_explains_replacement(rec):
    detections = [{"label": "couch", "category": "sofa", "confidence": 0.9, "box": [0, 0, 1, 1]}]
    sofa = next(i for i in rec.recommend(prefs(), detections)[0]["items"] if i["category"] == "sofa")
    assert any("Replaces" in r for r in sofa["reasons"])


def test_low_budget_produces_warning(rec):
    opt = rec.recommend(prefs(budget=100))[0]
    assert opt["warnings"]
    assert opt["total"] <= 100


def test_every_item_has_explanations(rec):
    for i in rec.recommend(prefs())[0]["items"]:
        assert len(i["reasons"]) >= 2
        assert 0 <= i["match_score"] <= 1


@pytest.mark.parametrize("field,value", [("room_type", "garage"), ("style", "gothic"), ("scheme", "neon")])
def test_invalid_preferences_rejected(rec, field, value):
    with pytest.raises(ValueError):
        rec.recommend(prefs(**{field: value}))


def test_moderate_budget_furnishes_all_optional_items(rec):
    """Regression: an expensive anchor piece must not silently starve the rest of the room."""
    opt = rec.recommend(prefs(budget=2500))[0]
    assert len(opt["items"]) >= 9
    assert opt["total"] <= 2500


def test_items_dropped_for_budget_are_explained(rec):
    opt = rec.recommend(prefs(budget=600))[0]
    named = {i["category"] for i in opt["items"]} | {s["category"] for s in opt["skipped"]}
    assert {"floor_lamp", "plant", "cushions"} <= named


@pytest.mark.parametrize("style", list(STYLES))
@pytest.mark.parametrize("budget", [1500, 6000, 12000])
def test_budget_saver_never_costs_more_than_best_match(rec, style, budget):
    best, saver, _ = rec.recommend(prefs(style=style, budget=budget))
    assert saver["total"] <= best["total"]
