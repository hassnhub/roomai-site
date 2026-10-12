"""Knowledge-based, content-filtering recommender for furniture and décor.

For each item the room needs, candidates are scored on
  * style match      (exact style or a compatible style),
  * colour harmony   (CIELAB ΔE to the chosen scheme and the existing room palette),
  * value            (closeness of the price to that category's share of the budget),
and filtered by physical fit (room dimensions) and a floor-space limit. A greedy
selection with an "essentials reserve" keeps the total within budget, followed by an
upgrade pass that spends any remaining budget on better-matching pieces.

Three options are produced with different strategies so users can compare designs.
"""
from dataclasses import dataclass, field

from ..domain import (
    CATEGORIES, COLOUR_SCHEMES, FLOOR_STANDING, MAX_FOOTPRINT_RATIO, ROOM_NEEDS, STYLE_COMPATIBILITY, STYLES,
)
from .colours import harmony_score

WALL_ITEMS = {"wall_art", "mirror", "curtains", "pendant_light"}


@dataclass
class Preferences:
    room_type: str
    style: str
    scheme: str
    length_m: float
    width_m: float
    budget: float
    keep_existing: bool = False
    height_m: float = 2.4


@dataclass
class Strategy:
    key: str
    title: str
    description: str
    spend_factor: float
    compat_style_score: float = 0.6
    prefer_tier: str | None = None
    avoid_skus: set = field(default_factory=set)
    upgrade: bool = True
    # "allocation": prefer prices near each category's budget share; "cheap": prefer lower prices.
    value_mode: str = "allocation"


def _fits(p: dict, prefs: Preferences) -> bool:
    long_cm, short_cm = max(prefs.length_m, prefs.width_m) * 100, min(prefs.length_m, prefs.width_m) * 100
    if p["category"] in WALL_ITEMS:
        return p["width_cm"] <= long_cm and p["height_cm"] <= prefs.height_m * 100
    a, b = sorted([p["width_cm"], p["depth_cm"]], reverse=True)
    # Leave clearance: furniture should not exceed 90% of the wall it sits along.
    return a <= long_cm * 0.9 and b <= short_cm * 0.9 and p["height_cm"] <= prefs.height_m * 100


def _footprint_m2(p: dict, qty: int) -> float:
    if p["category"] not in FLOOR_STANDING:
        return 0.0
    return p["width_cm"] * p["depth_cm"] * qty / 10_000


def _style_score(p_style: str, style: str, compat_score: float) -> float:
    if p_style == style:
        return 1.0
    if p_style in STYLE_COMPATIBILITY.get(style, []):
        return compat_score
    return 0.0


class Recommender:
    def __init__(self, products: list[dict]):
        self.by_category: dict[str, list[dict]] = {}
        for p in products:
            self.by_category.setdefault(p["category"], []).append(p)

    # ------------------------------------------------------------------------------
    def recommend(self, prefs: Preferences, detections: list[dict] | None = None,
                  room_palette: list[dict] | None = None) -> list[dict]:
        if prefs.room_type not in ROOM_NEEDS:
            raise ValueError(f"Unknown room type: {prefs.room_type}")
        if prefs.style not in STYLES:
            raise ValueError(f"Unknown style: {prefs.style}")
        if prefs.scheme not in COLOUR_SCHEMES:
            raise ValueError(f"Unknown colour scheme: {prefs.scheme}")

        best = self._build(prefs, detections, room_palette, Strategy(
            "best_match", "Best Match",
            "The closest match to your style and colour scheme, within your budget.", 1.0))
        saver = self._build(prefs, detections, room_palette, Strategy(
            "budget_saver", "Budget Saver",
            "A complete look that keeps costs down, favouring great-value pieces.",
            0.65, prefer_tier="budget", upgrade=False, value_mode="cheap"))
        alt = self._build(prefs, detections, room_palette, Strategy(
            "alternative", "Alternative Look",
            "A different take that blends in complementary styles for a more eclectic finish.",
            0.9, compat_style_score=0.85, avoid_skus={i["sku"] for i in best["items"]}))
        return [best, saver, alt]

    # ------------------------------------------------------------------------------
    def _candidates(self, category: str, prefs: Preferences, strat: Strategy, scheme: list[str],
                    room_palette: list[dict] | None) -> list[dict]:
        out = []
        for p in self.by_category.get(category, []):
            s_score = _style_score(p["style"], prefs.style, strat.compat_style_score)
            if s_score == 0 or not _fits(p, prefs):
                continue
            c_score, de = harmony_score(p["colour_hex"], scheme, room_palette)
            true_match = 0.5 * s_score + 0.5 * c_score
            match = true_match
            if p["sku"] in strat.avoid_skus:
                match -= 0.3
            if strat.prefer_tier and p["tier"] == strat.prefer_tier:
                match += 0.1
            out.append({"product": p, "match": match, "true_match": true_match, "style_score": s_score, "colour_score": c_score, "delta_e": de})
        return out

    def _build(self, prefs: Preferences, detections, room_palette, strat: Strategy) -> dict:
        scheme = COLOUR_SCHEMES[prefs.scheme]["colours"]
        area = prefs.length_m * prefs.width_m
        target = prefs.budget * strat.spend_factor
        warnings: list[str] = []

        detected = {d["category"] for d in (detections or []) if d.get("category") and d["confidence"] >= 0.4}
        needs, kept, skipped = [], [], []
        for cat, weight, qty, essential, min_area in ROOM_NEEDS[prefs.room_type]:
            if prefs.keep_existing and cat in detected:
                kept.append({"category": cat, "label": CATEGORIES[cat]})
                continue
            if not essential and area < min_area:
                skipped.append({"category": cat, "label": CATEGORIES[cat],
                                "reason": f"Room is {area:.1f} m²; a {CATEGORIES[cat].lower()} needs about {min_area} m²."})
                continue
            needs.append((cat, weight, qty, essential))

        total_weight = sum(w for _, w, _, _ in needs) or 1
        cands = {cat: self._candidates(cat, prefs, strat, scheme, room_palette) for cat, *_ in needs}
        min_cost = {cat: min((c["product"]["price"] * q for c in cands[cat]), default=0) for cat, _, q, _ in needs}

        order = sorted(needs, key=lambda n: (not n[3], -n[1]))  # essentials first, then by weight
        chosen: dict[str, dict] = {}
        remaining = target
        for i, (cat, weight, qty, essential) in enumerate(order):
            alloc = target * weight / total_weight
            reserve = sum(min_cost[c] for c, _, _, e in order[i + 1:] if e)
            affordable = [c for c in cands[cat] if c["product"]["price"] * qty <= remaining - reserve]
            if not affordable:
                if essential:
                    if cands[cat]:
                        warnings.append(
                            f"Budget too low for a {CATEGORIES[cat].lower()}: the cheapest suitable option is "
                            f"£{min_cost[cat]:,.0f}.")
                    else:
                        warnings.append(f"No {CATEGORIES[cat].lower()} in the catalogue fits this room's dimensions.")
                elif cands[cat]:
                    skipped.append({"category": cat, "label": CATEGORIES[cat],
                                    "reason": "Left out to keep this option within budget."})
                continue

            dearest = max(c["product"]["price"] for c in affordable) * qty

            def value(c):
                cost = c["product"]["price"] * qty
                if strat.value_mode == "cheap":
                    return 1 - cost / max(dearest, 1)
                return 1 - min(1.0, abs(cost - alloc) / max(alloc, 1))

            # Stay close to this category's budget share so high-priority pieces can't starve the rest;
            # the upgrade pass below spends whatever is left over.
            pool = [c for c in affordable if c["product"]["price"] * qty <= alloc * 1.3] or affordable
            w = 0.45 if strat.value_mode == "cheap" else 0.3
            pick = max(pool, key=lambda c: (1 - w) * c["match"] + w * value(c))
            chosen[cat] = {**pick, "qty": qty, "weight": weight, "essential": essential, "alloc": alloc}
            remaining -= pick["product"]["price"] * qty

        # Floor-space check: drop optional floor-standing items (lowest priority first) if overcrowded.
        def footprint():
            return sum(_footprint_m2(c["product"], c["qty"]) for c in chosen.values())

        limit = area * MAX_FOOTPRINT_RATIO
        for cat, weight, qty, essential in sorted(needs, key=lambda n: n[1]):
            if footprint() <= limit:
                break
            if cat in chosen and not essential and chosen[cat]["product"]["category"] in FLOOR_STANDING:
                skipped.append({"category": cat, "label": CATEGORIES[cat],
                                "reason": "Removed to keep enough open floor space to move around comfortably."})
                remaining += chosen[cat]["product"]["price"] * qty
                del chosen[cat]
        if footprint() > limit:
            warnings.append("Furniture would cover more than 45% of the floor; consider a larger room or fewer items.")

        # Upgrade pass: spend leftover budget on better-matching pieces, most important first.
        if strat.upgrade:
            for cat in sorted(chosen, key=lambda c: -chosen[c]["weight"]):
                cur = chosen[cat]
                cur_cost = cur["product"]["price"] * cur["qty"]
                better = [c for c in cands[cat]
                          if c["match"] > cur["match"] + 0.02
                          and c["product"]["price"] * cur["qty"] - cur_cost <= remaining
                          and footprint() - _footprint_m2(cur["product"], cur["qty"])
                          + _footprint_m2(c["product"], cur["qty"]) <= limit]
                if better:
                    up = max(better, key=lambda c: c["match"])
                    remaining -= up["product"]["price"] * cur["qty"] - cur_cost
                    chosen[cat] = {**cur, **{k: up[k] for k in ("product", "match", "true_match", "style_score", "colour_score", "delta_e")}}

        items = [self._item(chosen[cat], prefs, detected) for cat, *_ in needs if cat in chosen]
        return self.summarize({
            "key": strat.key,
            "title": strat.title,
            "description": strat.description,
            "items": items,
            "kept": kept,
            "skipped": skipped,
            "warnings": warnings,
        }, prefs)

    # ------------------------------------------------------------------------------
    @staticmethod
    def summarize(option: dict, prefs: Preferences) -> dict:
        """(Re)compute an option's totals - used after generation and after the user swaps an item."""
        items = option["items"]
        area = prefs.length_m * prefs.width_m
        total = round(sum(i["line_total"] for i in items), 2)
        fp = sum(i["width_cm"] * i["depth_cm"] * i["qty"] / 10_000 for i in items if i["category"] in FLOOR_STANDING)
        option.update({
            "total": total,
            "budget": prefs.budget,
            "remaining": round(prefs.budget - total, 2),
            "within_budget": total <= prefs.budget,
            "footprint_m2": round(fp, 2),
            "footprint_ratio": round(fp / area, 3) if area else 0,
            "match_score": round(sum(i["match_score"] for i in items) / len(items), 3) if items else 0,
        })
        return option

    def alternatives(self, prefs: Preferences, category: str, qty: int, essential: bool,
                     detections: list[dict] | None = None, room_palette: list[dict] | None = None,
                     limit: int = 8) -> list[dict]:
        """Best-matching alternatives for one item (for the "Swap" feature), regardless of budget."""
        scheme = COLOUR_SCHEMES[prefs.scheme]["colours"]
        strat = Strategy("swap", "", "", 1.0)
        detected = {d["category"] for d in (detections or []) if d.get("category") and d["confidence"] >= 0.4}
        cands = sorted(self._candidates(category, prefs, strat, scheme, room_palette),
                       key=lambda c: (-round(c["match"], 2), c["product"]["price"]))[:limit]
        return [self._item({**c, "qty": qty, "essential": essential}, prefs, detected) for c in cands]

    @staticmethod
    def _item(c: dict, prefs: Preferences, detected: set) -> dict:
        p, qty = c["product"], c["qty"]
        style_label = STYLES[p["style"]]["label"]
        scheme_label = COLOUR_SCHEMES[prefs.scheme]["label"]
        reasons = [
            f"Matches your {style_label} style" if p["style"] == prefs.style
            else f"{style_label} piece that complements {STYLES[prefs.style]['label']} interiors",
            f"{p['colour_name']} suits the {scheme_label} palette (ΔE {c['delta_e']:.0f})"
            if c["delta_e"] < 25 else f"{p['colour_name']} adds a contrasting accent to the {scheme_label} palette",
        ]
        if p["category"] in FLOOR_STANDING or p["category"] == "rug":
            reasons.append(f"{p['width_cm']}×{p['depth_cm']} cm fits your {prefs.length_m:g}×{prefs.width_m:g} m room")
        if p["category"] in detected and not prefs.keep_existing:
            reasons.append(f"Replaces the existing {CATEGORIES[p['category']].lower()} detected in your photo")
        return {
            "sku": p["sku"], "name": p["name"], "category": p["category"], "category_label": p["category_label"],
            "style": p["style"], "tier": p["tier"], "colour_name": p["colour_name"], "colour_hex": p["colour_hex"],
            "material": p["material"], "price": p["price"], "qty": qty, "line_total": round(p["price"] * qty, 2),
            "width_cm": p["width_cm"], "depth_cm": p["depth_cm"], "height_cm": p["height_cm"],
            "essential": c["essential"], "match_score": round(c["true_match"], 3),
            "style_score": round(c["style_score"], 2), "colour_score": round(c["colour_score"], 2),
            "reasons": reasons,
        }
