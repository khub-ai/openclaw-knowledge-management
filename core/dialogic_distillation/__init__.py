"""
core.dialogic_distillation — Domain-independent library for Dialogic Model Distillation.

Three parties collaborate in a structured dialogue to distill expert knowledge
into explicit, auditable rules that improve a weaker model at inference time
without modifying its weights.

Public API:
  DomainConfig          — dataclass configuring domain-specific vocabulary
  PairInfo              — TypedDict for confusable pair info
  FailureRecord         — TypedDict for failure case input

  run_dialogic_distillation  — multi-round dialogic distillation protocol
  generate_kf_guidance       — KF steering guidance generator

  run_expert_rule_author     — expert authors corrective rule from failure
  run_rule_validator_on_image — binary grounding check on one image
  validate_candidate_rule    — pool gate validation
  validate_candidate_rules_batch — batch pool gate
  run_contrastive_feature_analysis — TP vs FP analysis
  run_rule_spectrum_generator — 4-level specificity spectrum
  run_rule_completer         — fill implicit background conditions
  run_semantic_rule_validator — text-only logic check
  run_rule_reviser           — add tightening pre-condition

  image_block               — Anthropic image content block from file path
  encode_image_b64          — base64-encode an image file
  parse_json_block          — extract JSON from LLM output

  DEFAULT_PRECISION_GATE, DEFAULT_MAX_FP, DEFAULT_MAX_TIGHTENING_ROUNDS
"""

# Protocols and types
from .protocols import DomainConfig, PairInfo, FailureRecord

# Constants
from .constants import (
    DEFAULT_PRECISION_GATE,
    DEFAULT_MAX_FP,
    DEFAULT_MAX_TIGHTENING_ROUNDS,
    DEFAULT_EARLY_EXIT_FP,
)

# Dialogic protocol
from .dialogic import run_dialogic_distillation, generate_kf_guidance

# Agent runner functions
from .agents import (
    run_expert_rule_author,
    run_rule_validator_on_image,
    validate_candidate_rule,
    validate_candidate_rules_batch,
    run_contrastive_feature_analysis,
    run_rule_spectrum_generator,
    run_rule_completer,
    run_semantic_rule_validator,
    run_rule_reviser,
    image_block,
    encode_image_b64,
    parse_json_block,
)
