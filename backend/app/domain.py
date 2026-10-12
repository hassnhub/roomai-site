"""Interior-design domain knowledge: styles, colour schemes, room requirements.

These rules drive the recommender and are deliberately explicit (rather than learned)
so that every recommendation can be explained to the user.
"""

STYLES = {
    "modern": {"label": "Modern", "description": "Clean lines, sleek finishes and a restrained palette with bold accents."},
    "scandinavian": {"label": "Scandinavian", "description": "Light woods, soft textiles and bright, airy, functional spaces."},
    "industrial": {"label": "Industrial", "description": "Raw materials: reclaimed wood, black steel, leather and exposed textures."},
    "bohemian": {"label": "Boho", "description": "Layered patterns, natural fibres, plants and warm, eclectic colour."},
    "minimalist": {"label": "Minimalist", "description": "Only the essentials: uncluttered surfaces and a calm, neutral base."},
    "traditional": {"label": "Classic", "description": "Classic silhouettes, rich woods, symmetry and refined detailing."},
    "mid_century": {"label": "Mid-Century Modern", "description": "Organic curves, tapered legs, walnut and retro accent colours."},
    "japandi": {"label": "Japandi", "description": "Japanese calm meets Scandinavian warmth: natural, low and serene."},
    "coastal": {"label": "Coastal", "description": "Relaxed whites, sandy neutrals, blues and breezy natural textures."},
}

# Floor finish used when visualising each style.
STYLE_FLOORS = {
    "modern": "#8E8A86", "scandinavian": "#D9B98A", "industrial": "#5B4636", "bohemian": "#B9824F",
    "minimalist": "#D6D2C8", "traditional": "#6B3E26", "mid_century": "#A8774F", "japandi": "#CDB89A",
    "coastal": "#E1D3B8",
}

# Friendly names for detector labels shown in the UI.
DISPLAY_NAMES = {
    "couch": "Sofa", "chair": "Chair", "bed": "Bed", "dining table": "Table", "tv": "TV",
    "potted plant": "Plant", "vase": "Vase", "clock": "Clock", "book": "Books", "laptop": "Laptop",
    "keyboard": "Keyboard", "mouse": "Mouse", "cup": "Cup", "bowl": "Bowl", "wine glass": "Glass",
    "refrigerator": "Fridge", "oven": "Oven", "sink": "Sink", "toilet": "Toilet", "monitor": "Monitor",
}

# Styles that blend well together (used when an exact style match is unavailable).
STYLE_COMPATIBILITY = {
    "modern": ["minimalist", "mid_century", "industrial"],
    "scandinavian": ["japandi", "minimalist", "coastal"],
    "industrial": ["modern", "mid_century"],
    "bohemian": ["coastal", "mid_century"],
    "minimalist": ["modern", "japandi", "scandinavian"],
    "traditional": ["coastal", "mid_century"],
    "mid_century": ["modern", "scandinavian", "industrial"],
    "japandi": ["scandinavian", "minimalist"],
    "coastal": ["scandinavian", "bohemian", "traditional"],
}

COLOUR_SCHEMES = {
    "neutral": {"label": "Warm Neutrals", "colours": ["#F5F1EA", "#D9CFC1", "#A89F91", "#6B6259", "#FFFFFF"]},
    "warm": {"label": "Warm & Cosy", "colours": ["#E8B07A", "#C8553D", "#8C4A2F", "#F2D0A4", "#F7E7CE"]},
    "cool": {"label": "Cool Blues & Greys", "colours": ["#A7C4D8", "#4F7CAC", "#2E4A62", "#DDE8EE", "#8FA9A8"]},
    "earthy": {"label": "Earthy & Natural", "colours": ["#8A9A5B", "#C19A6B", "#6F4E37", "#D8C3A5", "#4E5D3A"]},
    "monochrome": {"label": "Monochrome", "colours": ["#111111", "#444444", "#888888", "#CCCCCC", "#F2F2F2"]},
    "pastel": {"label": "Soft Pastels", "colours": ["#F6C6C7", "#CDE7DD", "#FBE7B5", "#D7D2F0", "#BFE0F2"]},
    "bold": {"label": "Bold & Vibrant", "colours": ["#1F4E79", "#E9A23B", "#A83250", "#2A7F62", "#222222"]},
}

ROOM_TYPES = {
    "living_room": "Living Room",
    "bedroom": "Bedroom",
    "dining_room": "Dining Room",
    "home_office": "Home Office",
}

CATEGORIES = {
    "sofa": "Sofa", "armchair": "Armchair", "coffee_table": "Coffee Table", "tv_unit": "TV Unit",
    "rug": "Rug", "floor_lamp": "Floor Lamp", "shelving": "Shelving Unit", "plant": "Indoor Plant",
    "wall_art": "Wall Art", "cushions": "Cushion", "bed": "Bed Frame", "wardrobe": "Wardrobe",
    "bedside_table": "Bedside Table", "dresser": "Chest of Drawers", "table_lamp": "Table Lamp",
    "curtains": "Curtains", "mirror": "Mirror", "dining_table": "Dining Table", "dining_chair": "Dining Chair",
    "sideboard": "Sideboard", "pendant_light": "Pendant Light", "desk": "Desk", "office_chair": "Office Chair",
    "desk_lamp": "Desk Lamp", "storage_cabinet": "Storage Cabinet",
}

# Items that stand on the floor and therefore consume floor area.
FLOOR_STANDING = {
    "sofa", "armchair", "coffee_table", "tv_unit", "floor_lamp", "shelving", "bed", "wardrobe",
    "bedside_table", "dresser", "dining_table", "dining_chair", "sideboard", "desk", "office_chair",
    "storage_cabinet", "plant",
}

# What each room needs: (category, budget weight, quantity, essential, min floor area m²)
ROOM_NEEDS = {
    "living_room": [
        ("sofa", 0.34, 1, True, 0), ("coffee_table", 0.10, 1, True, 0), ("rug", 0.10, 1, False, 0),
        ("tv_unit", 0.10, 1, False, 0), ("armchair", 0.12, 1, False, 12), ("floor_lamp", 0.06, 1, False, 0),
        ("shelving", 0.08, 1, False, 14), ("wall_art", 0.04, 1, False, 0), ("plant", 0.03, 1, False, 0),
        ("cushions", 0.03, 4, False, 0),
    ],
    "bedroom": [
        ("bed", 0.36, 1, True, 0), ("bedside_table", 0.08, 2, True, 0), ("wardrobe", 0.20, 1, False, 9),
        ("rug", 0.07, 1, False, 0), ("dresser", 0.10, 1, False, 11), ("table_lamp", 0.04, 2, False, 0),
        ("curtains", 0.06, 1, False, 0), ("mirror", 0.04, 1, False, 0), ("wall_art", 0.03, 1, False, 0),
        ("plant", 0.02, 1, False, 0),
    ],
    "dining_room": [
        ("dining_table", 0.35, 1, True, 0), ("dining_chair", 0.30, 4, True, 0), ("sideboard", 0.15, 1, False, 10),
        ("pendant_light", 0.08, 1, False, 0), ("rug", 0.06, 1, False, 12), ("wall_art", 0.03, 1, False, 0),
        ("plant", 0.03, 1, False, 0),
    ],
    "home_office": [
        ("desk", 0.30, 1, True, 0), ("office_chair", 0.25, 1, True, 0), ("shelving", 0.15, 1, False, 6),
        ("desk_lamp", 0.06, 1, False, 0), ("storage_cabinet", 0.08, 1, False, 8), ("rug", 0.08, 1, False, 0),
        ("wall_art", 0.04, 1, False, 0), ("plant", 0.04, 1, False, 0),
    ],
}

# Keep at least this share of the floor free for circulation (ergonomic guidance ~40-60%).
MAX_FOOTPRINT_RATIO = 0.45

# COCO class -> furniture category per room type ("*" = any room).
COCO_TO_CATEGORY = {
    "couch": {"*": "sofa"},
    "chair": {"dining_room": "dining_chair", "home_office": "office_chair", "*": "armchair"},
    "bed": {"*": "bed"},
    "dining table": {"living_room": "coffee_table", "home_office": "desk", "*": "dining_table"},
    "tv": {"*": "tv_unit"},
    "potted plant": {"*": "plant"},
    "clock": {"*": "wall_art"},
    "vase": {"*": "plant"},
    "book": {"*": "shelving"},
}

# COCO classes that hint at the room type, with evidence weights.
ROOM_HINTS = {
    "bed": {"bedroom": 3.0},
    "couch": {"living_room": 2.5},
    "tv": {"living_room": 1.5, "bedroom": 0.3},
    "dining table": {"dining_room": 1.5, "home_office": 0.4, "living_room": 0.3},
    "chair": {"dining_room": 0.6, "home_office": 0.5, "living_room": 0.2},
    "laptop": {"home_office": 2.0},
    "keyboard": {"home_office": 2.0},
    "mouse": {"home_office": 1.5},
    "monitor": {"home_office": 1.5},
    "book": {"home_office": 0.4, "living_room": 0.3},
    "wine glass": {"dining_room": 1.0},
    "cup": {"dining_room": 0.3},
    "bowl": {"dining_room": 0.5},
    "potted plant": {"living_room": 0.2},
}
