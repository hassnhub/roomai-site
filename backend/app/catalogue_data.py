"""Deterministic generator for the demonstration product catalogue.

The catalogue is synthetic: names are invented and prices are indicative UK retail
ranges (GBP) chosen to reflect typical high-street budget and premium tiers. It is
generated rather than scraped so the project avoids retailer terms-of-service and
copyright issues while still giving the recommender realistic data to work with.
"""
import random

from .domain import CATEGORIES

# (budget price range, premium price range, (width, depth, height) cm budget, premium dims)
CATEGORY_SPECS = {
    "sofa": ((299, 549), (899, 1799), (165, 88, 82), (225, 98, 84)),
    "armchair": ((149, 279), (399, 799), (75, 78, 80), (85, 85, 86)),
    "coffee_table": ((59, 129), (199, 449), (100, 55, 42), (120, 65, 40)),
    "tv_unit": ((79, 169), (249, 599), (140, 40, 50), (180, 45, 55)),
    "rug": ((49, 119), (179, 449), (160, 230, 1), (200, 290, 1)),
    "floor_lamp": ((35, 79), (119, 279), (35, 35, 160), (45, 45, 170)),
    "shelving": ((59, 139), (199, 449), (80, 35, 180), (120, 40, 200)),
    "plant": ((19, 39), (49, 99), (35, 35, 90), (45, 45, 140)),
    "wall_art": ((25, 59), (89, 249), (60, 3, 80), (90, 3, 120)),
    "cushions": ((9, 18), (25, 49), (45, 15, 45), (50, 15, 50)),
    "bed": ((199, 399), (599, 1399), (150, 208, 100), (170, 218, 115)),
    "wardrobe": ((179, 349), (499, 1199), (100, 55, 190), (150, 60, 210)),
    "bedside_table": ((39, 79), (119, 249), (45, 38, 55), (50, 40, 58)),
    "dresser": ((99, 199), (299, 699), (80, 45, 85), (100, 48, 90)),
    "table_lamp": ((19, 45), (69, 149), (25, 25, 45), (30, 30, 55)),
    "curtains": ((29, 59), (89, 199), (230, 2, 230), (280, 2, 260)),
    "mirror": ((35, 79), (119, 299), (60, 3, 90), (80, 3, 160)),
    "dining_table": ((149, 299), (449, 1199), (120, 80, 75), (180, 90, 76)),
    "dining_chair": ((39, 79), (119, 249), (45, 52, 82), (50, 55, 84)),
    "sideboard": ((149, 279), (399, 899), (140, 42, 80), (170, 45, 82)),
    "pendant_light": ((29, 69), (99, 249), (40, 40, 30), (55, 55, 40)),
    "desk": ((79, 179), (249, 649), (110, 55, 75), (140, 70, 75)),
    "office_chair": ((69, 149), (229, 549), (62, 60, 105), (68, 65, 115)),
    "desk_lamp": ((19, 45), (59, 149), (20, 20, 45), (25, 25, 50)),
    "storage_cabinet": ((79, 159), (229, 499), (80, 40, 90), (100, 45, 110)),
}

STYLE_PROFILE = {
    "modern": {
        "collections": ["Nova", "Arc", "Vertex", "Halo"], "price_factor": 1.0,
        "materials": ["lacquered", "chrome", "marble-effect", "matte black"],
        "colours": [("Charcoal", "#3B3B3B"), ("Snow White", "#F4F4F4"), ("Slate Blue", "#5A6E8C"), ("Jet Black", "#1E1E1E")],
    },
    "scandinavian": {
        "collections": ["Malmo", "Fjord", "Lund", "Bergen"], "price_factor": 0.95,
        "materials": ["oak", "birch", "ash", "beech"],
        "colours": [("Light Oak", "#D9B98A"), ("Chalk White", "#F7F5F0"), ("Pale Grey", "#C9C9C4"), ("Sage", "#A9B69A")],
    },
    "industrial": {
        "collections": ["Foundry", "Brooklyn", "Forge", "Rivet"], "price_factor": 1.0,
        "materials": ["reclaimed wood", "black steel", "mango wood", "iron"],
        "colours": [("Cognac", "#8B4A2B"), ("Matte Black", "#2B2B2B"), ("Rust", "#A0522D"), ("Gunmetal", "#6E6E6E")],
    },
    "bohemian": {
        "collections": ["Marrakesh", "Sahara", "Bali", "Tulum"], "price_factor": 0.9,
        "materials": ["rattan", "mango wood", "cane", "acacia"],
        "colours": [("Terracotta", "#C2643A"), ("Mustard", "#D4A33B"), ("Natural Rattan", "#C8A165"), ("Teal", "#2F7C7A")],
    },
    "minimalist": {
        "collections": ["Pure", "Line", "Form", "Mono"], "price_factor": 1.0,
        "materials": ["ash", "matte white", "matte steel", "oak"],
        "colours": [("Pure White", "#FAFAFA"), ("Stone", "#D6D2C8"), ("Soft Black", "#222222"), ("Greige", "#BDB5A8")],
    },
    "traditional": {
        "collections": ["Windsor", "Ashford", "Kensington", "Belgrave"], "price_factor": 1.15,
        "materials": ["mahogany", "walnut", "oak", "brass-trimmed"],
        "colours": [("Mahogany", "#5C2E1E"), ("Cream", "#EFE6D2"), ("Burgundy", "#6D1F2A"), ("Forest Green", "#2F4B3A")],
    },
    "mid_century": {
        "collections": ["Palm Springs", "Sixty", "Atomic", "Retro"], "price_factor": 1.1,
        "materials": ["walnut", "teak", "brass", "oak"],
        "colours": [("Walnut", "#6B4429"), ("Mustard", "#D1A03A"), ("Deep Teal", "#22696B"), ("Burnt Orange", "#C8662E")],
    },
    "japandi": {
        "collections": ["Kyoto", "Zen", "Hinoki", "Wabi"], "price_factor": 1.1,
        "materials": ["ash", "bamboo", "oak", "walnut"],
        "colours": [("Natural Ash", "#CDB89A"), ("Charcoal", "#3A3734"), ("Clay", "#B08D74"), ("Rice White", "#EEE9E0")],
    },
    "coastal": {
        "collections": ["Nantucket", "Harbour", "Cove", "Seabreeze"], "price_factor": 0.95,
        "materials": ["whitewashed pine", "rattan", "driftwood", "painted oak"],
        "colours": [("Shell White", "#F8F8F5"), ("Sand", "#E1D3B8"), ("Navy", "#2C3E63"), ("Sea Blue", "#7FA8C9")],
    },
}

# Upholstered / soft-furnishing categories use the style's textiles instead of hard materials.
TEXTILES = {
    "modern": ["velvet", "bouclé"], "scandinavian": ["wool", "linen"], "industrial": ["canvas", "distressed leather"],
    "bohemian": ["jute", "woven cotton"], "minimalist": ["cotton", "wool"], "traditional": ["chenille", "velvet"],
    "mid_century": ["tweed", "wool"], "japandi": ["linen", "cotton"], "coastal": ["linen", "seagrass"],
}
TEXTILE_CATEGORIES = {"sofa", "armchair", "rug", "cushions", "curtains"}
PLANTS = ["Fiddle Leaf Fig", "Monstera", "Snake Plant", "Olive Tree", "Rubber Plant", "Kentia Palm"]
ART = ["Abstract", "Botanical", "Landscape", "Line Art", "Geometric", "Architectural"]

CATEGORY_CODES = {cat: f"{i:02d}" for i, cat in enumerate(CATEGORIES)}
STYLE_CODES = {s: s[:3].upper() for s in STYLE_PROFILE}


def _price(rng: random.Random, lo: int, hi: int, factor: float) -> float:
    raw = rng.uniform(lo, hi) * factor
    # Round to retail-style prices: £x9 / £x99.
    if raw < 100:
        return float(int(raw) // 5 * 5 + 4) if raw >= 10 else round(raw, 2)
    return float(int(raw) // 10 * 10 + 9)


def generate_products() -> list[dict]:
    rng = random.Random(2026)
    products = []
    for style, profile in STYLE_PROFILE.items():
        for category, (budget_rng, premium_rng, budget_dims, premium_dims) in CATEGORY_SPECS.items():
            for tier_index, (tier, price_rng, dims) in enumerate(
                [("budget", budget_rng, budget_dims), ("premium", premium_rng, premium_dims)]
            ):
                colour_name, colour_hex = profile["colours"][rng.randrange(len(profile["colours"]))]
                material = profile["materials"][rng.randrange(len(profile["materials"]))]
                collection = profile["collections"][(tier_index + len(category)) % len(profile["collections"])]
                label = CATEGORIES[category]
                if category in TEXTILE_CATEGORIES:
                    material = TEXTILES[style][rng.randrange(2)]
                if category == "plant":
                    material = "ceramic pot" if tier == "premium" else "plastic pot"
                    name = f"{collection} {PLANTS[rng.randrange(len(PLANTS))]}"
                elif category == "wall_art":
                    material = "framed print" if tier == "premium" else "canvas print"
                    name = f"{collection} {ART[rng.randrange(len(ART))]} {material.title()}"
                else:
                    name = f"{collection} {material.title()} {label}"
                width, depth, height = (round(d * rng.uniform(0.95, 1.05)) if d > 3 else d for d in dims)
                products.append({
                    "sku": f"RA-{STYLE_CODES[style]}-{CATEGORY_CODES[category]}-{'B' if tier == 'budget' else 'P'}",
                    "name": name,
                    "category": category,
                    "category_label": label,
                    "style": style,
                    "tier": tier,
                    "colour_name": colour_name,
                    "colour_hex": colour_hex,
                    "material": material,
                    "price": _price(rng, *price_rng, profile["price_factor"]),
                    "currency": "GBP",
                    "width_cm": width,
                    "depth_cm": depth,
                    "height_cm": height,
                    "description": (
                        f"A {tier} {label.lower()} from the {collection} collection in {colour_name.lower()} "
                        f"{material}, designed for {style.replace('_', '-')} interiors."
                    ),
                })
    return products
