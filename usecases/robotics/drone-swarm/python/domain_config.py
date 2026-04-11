"""
Domain configuration for the FleetPatch drone swarm use case.

Cross-modal DD: TUTOR knowledge originates in SAR (Synthetic Aperture Radar);
PUPIL classifies optical RGB frames from scout drones.

See DESIGN.md §4 for the full domain configuration rationale.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from core.dialogic_distillation.protocols import DomainConfig

# ---------------------------------------------------------------------------
# Primary domain configuration
# ---------------------------------------------------------------------------

DRONE_SWARM_CONFIG = DomainConfig(
    expert_role=(
        "senior SAR (Synthetic Aperture Radar) analyst and search-and-rescue "
        "coordinator with field experience interpreting radar returns alongside "
        "optical footage in disaster response operations"
    ),
    item_noun="optical drone camera frame",
    item_noun_plural="optical drone camera frames",
    classification_noun="survivor presence assessment",
    class_noun="presence class",
    feature_noun="optical surface feature or shadow pattern",
    observation_guidance=(
        "surface reflectivity and shadow patterns, object edge geometry, "
        "debris topology and elevation inconsistencies, panel lift indicators, "
        "irregular silhouettes inconsistent with flat-lying debris, "
        "colour contrast between debris and potential survivor indicators, "
        "non-uniform light reflection patterns on flat surfaces"
    ),
    non_visual_exclusions=(
        "SAR return values, radar backscatter intensity, subsurface conditions, "
        "infrared temperature readings (unless thermal camera is confirmed present "
        "on the specific drone tier), audio cues, seismic sensor data, "
        "time since collapse, weather conditions, rescue team reports"
    ),
    good_vocabulary_examples=[
        (
            "corrugated panel edge shows a shadow gap approximately 5-6 cm wide "
            "along the left edge, present at the centre of the edge but absent "
            "at the corners — consistent with a localised lift over a body mass "
            "rather than uniform tilt from a rubble mound beneath"
        ),
        (
            "panel surface shows non-uniform light reflection: specular highlight "
            "at the near end, diffuse return at the far end — inconsistent with "
            "a flat-lying panel under uniform overhead illumination"
        ),
        (
            "irregular elevation visible at the panel centre: the panel profile "
            "appears convex when viewed from the drone's perspective, whereas a "
            "flat-lying panel would appear uniformly planar"
        ),
    ],
    bad_vocabulary_examples=[
        (
            "SAR data shows a strong double-bounce return beneath the panel "
            "(references non-optical sensor data not available to scout drones)"
        ),
        (
            "thermal hotspot detected beneath the panel "
            "(thermal sensor not present on scout tier)"
        ),
        (
            "survivor confirmed by ground extraction team "
            "(non-visual external confirmation not available at inference time)"
        ),
        (
            "the panel is bridging a human body "
            "(structural inference — must be derived from observable optical "
            "features, not stated as a premise)"
        ),
    ],
    precision_gate=0.90,   # safety-critical: false negatives cost lives
    max_fp=0,              # zero tolerance: every false positive diverts rescue resources
)

# ---------------------------------------------------------------------------
# Tier-specific observability contexts for the KF grounding check
# ---------------------------------------------------------------------------
# These are injected into the grounding check prompt to ensure rules are
# validated against the actual sensor capabilities of each fleet tier.

TIER_OBSERVABILITY: dict[str, str] = {
    "scout": (
        "Camera: 12MP RGB, fixed mount, no stabilisation. "
        "Operational altitude: 30–50 m AGL. "
        "Approximate pixel footprint: 1.5–3 cm per pixel at nadir. "
        "Features requiring sub-5 cm resolution, colour accuracy beyond broad "
        "categories (red / orange / blue / dark / light), or stable hover are "
        "NOT reliably observable on this tier. "
        "Motion blur is present at higher wind speeds."
    ),
    "commander": (
        "Camera: 20MP RGB, stabilised 3-axis gimbal. "
        "Operational altitude: 15–25 m AGL with hover capability. "
        "Approximate pixel footprint: 0.5–1.5 cm per pixel at nadir. "
        "Most surface texture features, shadow gap widths ≥ 2 cm, and "
        "corrugation geometry are observable on this tier."
    ),
}

# ---------------------------------------------------------------------------
# Confusable pairs
# ---------------------------------------------------------------------------

CONFUSABLE_PAIRS = [
    {
        "class_a": "person_under_panel",
        "class_b": "empty_panel",
        "description": (
            "Primary pair. Corrugated or flat metal panel lying on rubble. "
            "Ground truth requires SAR confirmation or physical extraction. "
            "Discriminating features: edge shadow gap geometry, panel convexity."
        ),
        "priority": "critical",
    },
    {
        "class_a": "person_under_emergency_blanket",
        "class_b": "empty_emergency_blanket",
        "description": (
            "Mylar emergency blanket lying on rubble vs. covering a survivor. "
            "Discriminating features: irregular surface topology, edge lift pattern."
        ),
        "priority": "high",
    },
    {
        "class_a": "person_in_void_space",
        "class_b": "debris_void",
        "description": (
            "Dark gap in rubble containing a survivor vs. empty void. "
            "Discriminating features: any skin-tone region, clothing texture, "
            "limb geometry visible within the void."
        ),
        "priority": "high",
    },
    {
        "class_a": "person_partially_buried",
        "class_b": "rubble_mound",
        "description": (
            "Irregular rubble elevation caused by a buried person vs. natural "
            "debris accumulation. Discriminating features: bilateral symmetry "
            "inconsistent with random debris, any exposed body part."
        ),
        "priority": "medium",
    },
]

# ---------------------------------------------------------------------------
# Cross-modal TUTOR prompt template
# ---------------------------------------------------------------------------
# This replaces the standard DD TUTOR prompt when the confirmation comes from
# a different sensor modality than the PUPIL's perception.

CROSS_MODAL_TUTOR_PROMPT = """
You are a {expert_role}.

The optical scout drone classified the image below as: **{pupil_classification}**
(confidence: {pupil_confidence:.2f}).

Ground truth confirmation from {confirmation_modality}: **{ground_truth_class}**
Confirmation details: {confirmation_details}

The optical image is shown below. The scout drone's camera is a 12MP RGB
sensor at 30–50 m altitude. It cannot see SAR returns, thermal signatures,
or subsurface features.

Your task: describe what features are VISIBLE IN THIS OPTICAL IMAGE that
should have led to a correct classification of **{ground_truth_class}**.

Focus only on what a camera can observe: surface geometry, shadow patterns,
edge characteristics, reflectivity, colour, and topology.
Do not reference SAR, thermal, or any sensor the optical drone does not carry.
""".strip()
