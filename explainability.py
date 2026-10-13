"""
Multi-Attribute Explainability Engine for Shop-the-Look
======================================================
Provides honest, structured visual and attribute-based explanations:
- Color similarity (CIELAB ΔE & dominant palette mapping)
- Silhouette and aspect-ratio compatibility
- Pattern & Texture complexity (spatial gradient energy)
- Confidence-aware rationale (openly acknowledges low-confidence fallbacks)
"""

import math
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image


COLOR_FAMILIES = [
    ("Black", (30, 30, 30)),
    ("White / Cream", (240, 240, 235)),
    ("Grey", (130, 130, 130)),
    ("Navy / Dark Blue", (25, 35, 75)),
    ("Light Blue / Denim", (90, 140, 200)),
    ("Beige / Tan / Khaki", (195, 175, 140)),
    ("Brown / Camel", (120, 80, 50)),
    ("Red / Crimson", (190, 30, 40)),
    ("Burgundy / Maroon", (110, 25, 45)),
    ("Pink / Blush", (230, 140, 165)),
    ("Olive / Forest Green", (60, 95, 55)),
    ("Yellow / Mustard", (210, 175, 50)),
    ("Orange / Rust", (205, 95, 40)),
    ("Purple / Lavender", (125, 75, 145)),
]


def rgb_to_cielab(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """Convert sRGB (0-255) to CIE L*a*b* color space."""
    r = rgb[0] / 255.0
    g = rgb[1] / 255.0
    b = rgb[2] / 255.0

    r = ((r + 0.055) / 1.055) ** 2.4 if r > 0.04045 else r / 12.92
    g = ((g + 0.055) / 1.055) ** 2.4 if g > 0.04045 else g / 12.92
    b = ((b + 0.055) / 1.055) ** 2.4 if b > 0.04045 else b / 12.92

    # sRGB to CIE XYZ (D65 illuminant)
    x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047
    y = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 1.00000
    z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883

    def f(t):
        return t ** (1 / 3) if t > 0.008856 else (7.787 * t) + (16 / 116)

    L = max(0.0, (116.0 * f(y)) - 16.0)
    a = 500.0 * (f(x) - f(y))
    b_lab = 200.0 * (f(y) - f(z))
    return L, a, b_lab


def delta_e_cielab(lab1: Tuple[float, float, float], lab2: Tuple[float, float, float]) -> float:
    """Euclidean ΔE in CIELAB space."""
    return math.sqrt(
        (lab1[0] - lab2[0]) ** 2 +
        (lab1[1] - lab2[1]) ** 2 +
        (lab1[2] - lab2[2]) ** 2
    )


def extract_dominant_colors(
    image: Image.Image,
    k: int = 2,
    is_query_crop: bool = False
) -> List[Tuple[str, Tuple[int, int, int]]]:
    """
    Extract dominant color families from an image.
    For query crops, samples the central region to reduce background clutter (chairs, floors).
    """
    img_rgb = image.convert("RGB")
    if is_query_crop:
        w, h = img_rgb.size
        # Sample central 70% to avoid border backgrounds
        cw, ch = int(w * 0.15), int(h * 0.15)
        if w > 30 and h > 30:
            img_rgb = img_rgb.crop((cw, ch, w - cw, h - ch))

    img_small = img_rgb.resize((50, 50))
    pixels = np.array(img_small).reshape(-1, 3).astype(float)

    # Filter out pure white catalog background (mean > 248) if packshot
    if not is_query_crop:
        non_bg = pixels[pixels.mean(axis=1) < 245]
        if len(non_bg) > 100:
            pixels = non_bg

    # Initialize k cluster centers evenly
    indices = np.linspace(0, len(pixels) - 1, k, dtype=int)
    centers = pixels[indices].copy()

    for _ in range(6):
        dists = np.linalg.norm(pixels[:, None] - centers[None], axis=2)
        labels = dists.argmin(axis=1)
        for j in range(k):
            cluster_pts = pixels[labels == j]
            if len(cluster_pts) > 0:
                centers[j] = cluster_pts.mean(axis=0)

    results = []
    for c in centers:
        rgb = tuple(int(max(0, min(255, v))) for v in c)
        c_lab = rgb_to_cielab(rgb)
        best_name = "Neutral"
        min_de = float("inf")
        for name, ref_rgb in COLOR_FAMILIES:
            ref_lab = rgb_to_cielab(ref_rgb)
            de = delta_e_cielab(c_lab, ref_lab)
            if de < min_de:
                min_de = de
                best_name = name
        results.append((best_name, rgb))

    return results


def compute_texture_complexity(image: Image.Image) -> str:
    """
    Measure gradient energy (Laplacian variance) to classify
    garment as solid/plain vs textured/patterned.
    """
    gray = np.array(image.convert("L").resize((100, 100)), dtype=float)
    dy, dx = np.gradient(gray)
    gradient_magnitude = np.sqrt(dx ** 2 + dy ** 2)
    std_dev = float(np.std(gradient_magnitude))

    if std_dev < 14.0:
        return "Solid / Plain"
    elif std_dev < 24.0:
        return "Subtle Texture / Knit"
    else:
        return "Patterned / Printed"


class ExplainabilityEngine:
    """
    Evaluates similarities between a detected scene crop and candidate catalog product.
    Generates honest, confidence-aware attributions.
    """

    @staticmethod
    def explain(
        query_crop: Image.Image,
        catalog_image: Image.Image,
        cos_similarity: float,
        detected_category: str = "clothing",
        exact_match_threshold: float = 0.226
    ) -> Dict:
        q_colors = extract_dominant_colors(query_crop, k=2, is_query_crop=True)
        c_colors = extract_dominant_colors(catalog_image, k=2, is_query_crop=False)

        q_color_names = [c[0] for c in q_colors]
        c_color_names = [c[0] for c in c_colors]

        # Calculate best pair-wise ΔE between dominant colors
        min_de = float("inf")
        for _, q_rgb in q_colors:
            q_lab = rgb_to_cielab(q_rgb)
            for _, c_rgb in c_colors:
                c_lab = rgb_to_cielab(c_rgb)
                de = delta_e_cielab(q_lab, c_lab)
                if de < min_de:
                    min_de = de

        color_match_pct = max(10.0, round(100.0 - min_de * 1.5, 1))
        shared_colors = list(set(q_color_names).intersection(set(c_color_names)))

        # Texture and pattern
        q_texture = compute_texture_complexity(query_crop)
        c_texture = compute_texture_complexity(catalog_image)
        texture_match = (q_texture == c_texture)

        visual_sim_pct = round(max(0.0, min(100.0, cos_similarity * 100.0)), 1)
        is_exact = cos_similarity >= exact_match_threshold

        # Aspect ratios & silhouette framing
        qw, qh = query_crop.size
        cw, ch = catalog_image.size
        q_ar = qh / max(1, qw)
        c_ar = ch / max(1, cw)
        ar_diff = abs(q_ar - c_ar)

        # ── Honest Multi-Attribute Factor Generation ─────────────────
        explanation_points = []

        if cos_similarity < 0.45:
            # Low confidence fallback: be completely honest with the user
            confidence_level = "Low Confidence Fallback"
            explanation_points.append(
                f"⚠️ **Weak Alignment ({visual_sim_pct}%)**: No direct catalog match found for this detected patch."
            )
            explanation_points.append(
                "Serving distant nearest visual neighbor from inventory based on coarse feature embedding."
            )
            if shared_colors:
                explanation_points.append(f"~ Minor palette overlap in **{', '.join(shared_colors)}**")
            else:
                explanation_points.append(
                    f"✗ Noticeable color divergence: Query is **{q_color_names[0]}** vs. Catalog is **{c_color_names[0]}** (ΔE={min_de:.1f})"
                )
            if ar_diff > 0.6:
                explanation_points.append("✗ Silhouette divergence: Packshot framing differs from query crop.")

            summary_sentence = (
                f"Distant fallback recommendation ({visual_sim_pct}% visual alignment). "
                f"The detected crop exhibits visual ambiguity (folds/layering) with no close match in the 28,000 catalog."
            )
        else:
            # Moderate to High similarity
            confidence_level = "Strong Match" if cos_similarity >= 0.65 else "Moderate Alignment"

            if shared_colors:
                explanation_points.append(
                    f"✓ **Color Match**: Shared tonal palette in **{', '.join(shared_colors)}** (ΔE={min_de:.1f})"
                )
            elif min_de < 28.0:
                explanation_points.append(
                    f"~ **Related Tone**: Harmonious color harmony (**{q_color_names[0]}** query vs. **{c_color_names[0]}** product)"
                )
            else:
                explanation_points.append(
                    f"✗ **Color Shift**: Query dominant is **{q_color_names[0]}**, catalog product is **{c_color_names[0]}**"
                )

            if texture_match:
                explanation_points.append(f"✓ **Texture Match**: Both share **{q_texture}** surface appearance")
            else:
                explanation_points.append(f"~ **Texture Difference**: Query is **{q_texture}** vs. catalog **{c_texture}**")

            if ar_diff < 0.5:
                explanation_points.append(f"✓ **Silhouette Match**: Compatible {detected_category} proportion and vertical framing")
            else:
                explanation_points.append(f"~ **Silhouette**: Stylistic alternative within {detected_category} category")

            summary_sentence = (
                f"Candidate match with {visual_sim_pct}% visual feature alignment "
                f"matching in {detected_category} category and {', '.join(q_color_names)} color palette."
            )

        return {
            "is_exact_match": is_exact,
            "confidence_level": confidence_level,
            "visual_similarity_pct": visual_sim_pct,
            "color_match_pct": color_match_pct,
            "primary_query_color": q_color_names[0] if q_color_names else "Unknown",
            "primary_catalog_color": c_color_names[0] if c_color_names else "Unknown",
            "shared_colors": shared_colors,
            "query_texture": q_texture,
            "catalog_texture": c_texture,
            "detected_category": detected_category,
            "key_factors": explanation_points,
            "summary_text": summary_sentence
        }
