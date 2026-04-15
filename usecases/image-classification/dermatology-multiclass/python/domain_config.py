"""
domain_config.py — Dermatology domain configuration for dialogic distillation.
Identical vocabulary to the 2-way version; N-way adds no new terminology.
"""
from core.dialogic_distillation import DomainConfig

DERM_CONFIG = DomainConfig(
    expert_role="senior dermoscopy expert and knowledge engineer",
    item_noun="dermoscopic image",
    item_noun_plural="dermoscopic images",
    classification_noun="diagnosis",
    class_noun="class",
    feature_noun="dermoscopic feature",
    observation_guidance=(
        "pigment network (typical/atypical), border characteristics, "
        "color variation, globules and dots, regression structures, "
        "blue-white veil, vascular structures, symmetry"
    ),
    non_visual_exclusions=(
        "patient history, symptoms, age, body location, palpation findings"
    ),
    good_vocabulary_examples=[
        "gray-blue structureless areas visible within the lesion body",
        "multiple distinct color zones (brown, tan, gray-blue)",
        "dark brown or black irregularly shaped patches",
    ],
    bad_vocabulary_examples=[
        "regression structures present (clinical term the validator won't use)",
        "atypical pigment network (clinical term the validator won't use)",
        "polychromatic pigmentation pattern",
    ],
)
