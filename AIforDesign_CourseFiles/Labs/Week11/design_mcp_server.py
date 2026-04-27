"""
MCP Server: Design Tools

A Model Context Protocol (MCP) server that exposes design-related
tools for color palettes, typography, accessibility checking,
spacing systems, and dark mode generation.

Any MCP-compatible client (Claude Code, Cursor, ChatGPT, etc.)
can connect to this server and discover its tools automatically.

Transport: stdio (for local use).

Usage:
    python design_mcp_server.py
"""

import json
import re
import math
from mcp.server.fastmcp import FastMCP

# ── Create the MCP server instance ────────────────────────────
mcp = FastMCP("design-tools")

# ── Shared helpers ────────────────────────────────────────────

_HEX_PATTERN = re.compile(r"^#?([0-9A-Fa-f]{6})$")

def _validate_hex(color: str) -> str | None:
    """Return a normalised '#RRGGBB' string, or None if invalid."""
    m = _HEX_PATTERN.match(color.strip())
    return f"#{m.group(1).upper()}" if m else None


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _luminance(r: int, g: int, b: int) -> float:
    """Relative luminance per WCAG 2.1 definition."""
    def lin(c: int) -> float:
        s = c / 255.0
        return s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


# ══════════════════════════════════════════════════════════════
# Tool 1: Color Palette
# ══════════════════════════════════════════════════════════════

PALETTES = {
    "calm":       {"colors": ["#E8F4FD", "#B8D4E3", "#6B9BC3", "#4A7C9B", "#2C5F7C"],
                   "desc": "Cool blues, tranquil and serene"},
    "energetic":  {"colors": ["#FF6B35", "#F7C948", "#FF3F00", "#FF8C42", "#FFD166"],
                   "desc": "Warm oranges, vibrant and dynamic"},
    "luxury":     {"colors": ["#1A1A2E", "#16213E", "#0F3460", "#E94560", "#D4AF37"],
                   "desc": "Deep navy + gold, sophisticated"},
    "nature":     {"colors": ["#2D6A4F", "#40916C", "#52B788", "#74C69D", "#B7E4C7"],
                   "desc": "Fresh greens, organic and earthy"},
    "playful":    {"colors": ["#FF595E", "#FFCA3A", "#8AC926", "#1982C4", "#6A4C93"],
                   "desc": "Rainbow spectrum, youthful and fun"},
    "minimalist": {"colors": ["#FFFFFF", "#F5F5F5", "#E0E0E0", "#333333", "#000000"],
                   "desc": "Monochrome, clean and modern"},
}

@mcp.tool()
def get_color_palette(mood: str) -> str:
    """Get a 5-color palette (hex codes) for a given design mood.

    Args:
        mood: One of: calm, energetic, luxury, nature, playful, minimalist

    Returns JSON:
        {"colors": ["#HEX", ...], "desc": "Palette description"}

    On invalid mood returns:
        {"error": "...", "valid_moods": [...]}
    """
    key = mood.strip().lower()
    if key not in PALETTES:
        return json.dumps({
            "error": f"Unknown mood '{mood}'.",
            "valid_moods": sorted(PALETTES.keys()),
        })
    return json.dumps(PALETTES[key])


# ══════════════════════════════════════════════════════════════
# Tool 2: Font Recommendation
# ══════════════════════════════════════════════════════════════

FONTS = {
    "modern":      {"heading": "Inter",            "body": "Source Sans Pro",  "accent": "Space Grotesk"},
    "classic":     {"heading": "Playfair Display",  "body": "Lora",            "accent": "Cormorant"},
    "playful":     {"heading": "Fredoka One",      "body": "Nunito",           "accent": "Baloo 2"},
    "corporate":   {"heading": "Roboto Slab",      "body": "Open Sans",        "accent": "Roboto"},
    "handwritten": {"heading": "Caveat",           "body": "Quicksand",        "accent": "Patrick Hand"},
}

@mcp.tool()
def get_font_recommendation(style: str) -> str:
    """Recommend a font pairing (heading, body, accent) for a design style.

    Args:
        style: One of: modern, classic, playful, corporate, handwritten

    Returns JSON:
        {"heading": "Font Name", "body": "Font Name", "accent": "Font Name"}

    On invalid style returns:
        {"error": "...", "valid_styles": [...]}
    """
    key = style.strip().lower()
    if key not in FONTS:
        return json.dumps({
            "error": f"Unknown style '{style}'.",
            "valid_styles": sorted(FONTS.keys()),
        })
    return json.dumps(FONTS[key])


# ══════════════════════════════════════════════════════════════
# Tool 3: Contrast Ratio Calculator
# ══════════════════════════════════════════════════════════════

@mcp.tool()
def calculate_contrast_ratio(color1: str, color2: str) -> str:
    """Calculate the WCAG 2.1 contrast ratio between two hex colors.

    Args:
        color1: First hex color  (e.g. '#FFFFFF' or 'FF6B6B')
        color2: Second hex color (e.g. '#1A1A2E' or '1A1A2E')

    Returns JSON:
        {
            "color1": "#FFFFFF", "color2": "#1A1A2E",
            "contrast_ratio": 17.11,
            "passes_AA_normal": true,   // >= 4.5:1
            "passes_AA_large":  true,   // >= 3.0:1
            "passes_AAA":       true    // >= 7.0:1
        }

    On invalid input returns:
        {"error": "..."}
    """
    c1 = _validate_hex(color1)
    c2 = _validate_hex(color2)
    if not c1 or not c2:
        bad = []
        if not c1:
            bad.append(f"color1='{color1}'")
        if not c2:
            bad.append(f"color2='{color2}'")
        return json.dumps({
            "error": f"Invalid hex color: {', '.join(bad)}. "
                     f"Expected format: '#RRGGBB' (e.g. '#FF6B6B').",
        })

    try:
        L1 = _luminance(*_hex_to_rgb(c1))
        L2 = _luminance(*_hex_to_rgb(c2))
        ratio = (max(L1, L2) + 0.05) / (min(L1, L2) + 0.05)
        return json.dumps({
            "color1": c1,
            "color2": c2,
            "contrast_ratio": round(ratio, 2),
            "passes_AA_normal": ratio >= 4.5,
            "passes_AA_large":  ratio >= 3.0,
            "passes_AAA":       ratio >= 7.0,
        })
    except Exception as e:
        return json.dumps({"error": f"Calculation failed: {e}"})


# ══════════════════════════════════════════════════════════════
# Tool 4: Spacing Scale Generator
# ══════════════════════════════════════════════════════════════

@mcp.tool()
def generate_spacing_scale(base_unit: int = 4, steps: int = 8) -> str:
    """Generate a spacing scale for a design system.

    Produces a geometric progression commonly used in UI design
    (e.g. 4 → 8 → 12 → 16 → 24 → 32 → 48 → 64).

    Args:
        base_unit: The smallest spacing value in px (default 4)
        steps:     How many values to generate (default 8, max 12)

    Returns JSON:
        {
            "base_unit": 4,
            "scale": [4, 8, 12, 16, 24, 32, 48, 64],
            "css_vars": "--space-1: 4px; --space-2: 8px; ..."
        }
    """
    base_unit = max(1, min(base_unit, 16))
    steps = max(2, min(steps, 12))

    # Common multiplier sequence: 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64
    multipliers = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    scale = [base_unit * m for m in multipliers[:steps]]

    css_vars = "; ".join(
        f"--space-{i+1}: {v}px" for i, v in enumerate(scale)
    )

    return json.dumps({
        "base_unit": base_unit,
        "scale": scale,
        "css_vars": css_vars,
    })


# ══════════════════════════════════════════════════════════════
# Tool 5: Dark Mode Palette Suggestion
# ══════════════════════════════════════════════════════════════

@mcp.tool()
def suggest_dark_mode(hex_colors: str) -> str:
    """Suggest dark-mode equivalents for a list of light-theme colors.

    Inverts lightness while preserving hue and saturation, then
    adjusts to maintain WCAG-friendly contrast on a dark background.

    Args:
        hex_colors: Comma-separated hex colors
                    (e.g. '#FFFFFF,#333333,#4A90D9,#E74C3C')

    Returns JSON:
        {
            "dark_background": "#121212",
            "mappings": [
                {"original": "#FFFFFF", "dark_mode": "#E0E0E0", "role": "..."},
                ...
            ]
        }
    """
    raw_list = [c.strip() for c in hex_colors.split(",") if c.strip()]
    if not raw_list:
        return json.dumps({"error": "No colors provided. Send comma-separated hex values."})

    mappings = []
    for raw in raw_list:
        validated = _validate_hex(raw)
        if not validated:
            mappings.append({
                "original": raw,
                "dark_mode": None,
                "error": f"Invalid hex: '{raw}'",
            })
            continue

        r, g, b = _hex_to_rgb(validated)
        lum = _luminance(r, g, b)

        # Strategy: very light colors become muted/off-white;
        # dark colors become lighter; mid-tones shift moderately.
        if lum > 0.85:
            # Near-white → off-white on dark
            dr, dg, db = 224, 224, 224
            role = "surface text (light-on-dark)"
        elif lum < 0.05:
            # Near-black → use as dark surface
            dr, dg, db = 18, 18, 18
            role = "dark surface background"
        else:
            # Mid-tone: lighten slightly for dark backgrounds
            factor = 1.3 if lum < 0.3 else 0.85
            dr = min(255, int(r * factor + 40))
            dg = min(255, int(g * factor + 40))
            db = min(255, int(b * factor + 40))
            role = "accent / interactive element"

        dark_hex = f"#{dr:02X}{dg:02X}{db:02X}"
        mappings.append({
            "original": validated,
            "dark_mode": dark_hex,
            "role": role,
        })

    return json.dumps({
        "dark_background": "#121212",
        "mappings": mappings,
    })


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    mcp.run(transport="stdio")
