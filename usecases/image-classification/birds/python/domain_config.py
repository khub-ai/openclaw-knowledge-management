"""
domain_config.py — Ornithology domain configuration for dialogic distillation.
"""
from core.dialogic_distillation import DomainConfig

BIRDS_CONFIG = DomainConfig(
    expert_role="senior ornithologist and bird identification expert",
    item_noun="bird photograph",
    item_noun_plural="bird photographs",
    classification_noun="species identification",
    class_noun="species",
    feature_noun="field mark or plumage feature",
    observation_guidance=(
        "plumage coloration, wing bars, eye rings, bill shape, "
        "tail pattern, facial markings, body proportions"
    ),
    non_visual_exclusions=(
        "range, habitat, vocalizations, behavior, seasonal context"
    ),
)
