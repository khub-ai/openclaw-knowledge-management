"""
agents.py — Bird classification agents for the KF dialogic loop.

Infrastructure (call_agent, OpenRouter, ClaudeTutor, image encoding) is
re-used from the dermatology agents module to avoid duplication.

Patching agent functions delegate to core.dialogic_distillation, passing
the BIRDS_CONFIG for ornithological prompt vocabulary.
"""

from __future__ import annotations
import asyncio
from pathlib import Path

# ---------------------------------------------------------------------------
# Re-use call_agent infrastructure from dermatology (via importlib to avoid
# circular import — Python would find THIS file as "agents" via sys.path)
# ---------------------------------------------------------------------------
import importlib.util as _ilu

_DERM_AGENTS_PATH = (
    Path(__file__).resolve().parents[2] / "dermatology" / "python" / "agents.py"
)
_derm_spec = _ilu.spec_from_file_location("derm_agents", _DERM_AGENTS_PATH)
_derm_agents = _ilu.module_from_spec(_derm_spec)
_derm_spec.loader.exec_module(_derm_agents)  # type: ignore[union-attr]

call_agent           = _derm_agents.call_agent
ACTIVE_MODEL         = _derm_agents.ACTIVE_MODEL
reset_cost_tracker   = _derm_agents.reset_cost_tracker
get_cost_tracker     = _derm_agents.get_cost_tracker
get_call_cache_stats = _derm_agents.get_call_cache_stats

# Utilities re-exported from core
from core.dialogic_distillation import (
    encode_image_b64, image_block as _image_block, parse_json_block as _parse_json_block,
)


# ---------------------------------------------------------------------------
# Dialogic patching — delegated to core.dialogic_distillation
# ---------------------------------------------------------------------------
#
# All patching agent functions delegate to the domain-independent core
# library, passing the ornithology DomainConfig for prompt vocabulary.
# The original prompts and logic have been extracted to:
#   core/dialogic_distillation/agents.py
#   core/dialogic_distillation/prompts.py
# ---------------------------------------------------------------------------

from core.dialogic_distillation import agents as _dd_agents
from domain_config import BIRDS_CONFIG as _BIRDS_CONFIG


async def run_expert_rule_author(
    task: dict,
    wrong_prediction: str,
    correct_label: str,
    model_reasoning: str = "",
    model: str = "",
    prior_context: str = "",
) -> tuple[dict, int]:
    """Delegate to core — author a corrective rule from a failure case."""
    return await _dd_agents.run_expert_rule_author(
        task, wrong_prediction, correct_label, config=_BIRDS_CONFIG,
        model_reasoning=model_reasoning, model=model or ACTIVE_MODEL,
        prior_context=prior_context, call_agent_fn=call_agent,
    )


async def run_rule_validator_on_image(
    image_path: str,
    ground_truth: str,
    candidate_rule: dict,
    model: str = "",
) -> tuple[dict, int]:
    """Delegate to core — test whether a rule applies to a single image."""
    return await _dd_agents.run_rule_validator_on_image(
        image_path, ground_truth, candidate_rule, config=_BIRDS_CONFIG,
        model=model or ACTIVE_MODEL, call_agent_fn=call_agent,
    )


async def validate_candidate_rule(
    candidate_rule: dict,
    validation_images: list,
    trigger_image_path: str,
    trigger_correct_label: str,
    model: str = "",
    early_exit_fp: int = 2,
) -> dict:
    """Delegate to core — test a rule against a pool of labeled images."""
    return await _dd_agents.validate_candidate_rule(
        candidate_rule, validation_images, trigger_image_path,
        trigger_correct_label, config=_BIRDS_CONFIG, model=model or ACTIVE_MODEL,
        early_exit_fp=early_exit_fp, call_agent_fn=call_agent,
    )


async def validate_candidate_rules_batch(
    candidate_rules: list[dict],
    validation_images: list,
    trigger_image_path: str,
    trigger_correct_label: str,
    model: str = "",
) -> list[dict]:
    """Delegate to core — batch validate spectrum levels."""
    return await _dd_agents.validate_candidate_rules_batch(
        candidate_rules, validation_images, trigger_image_path,
        trigger_correct_label, config=_BIRDS_CONFIG, model=model or ACTIVE_MODEL,
        call_agent_fn=call_agent,
    )


async def run_contrastive_feature_analysis(
    tp_cases: list[dict],
    fp_cases: list[dict],
    candidate_rule: dict,
    pair_info: dict,
    model: str = "",
) -> tuple[dict, int]:
    """Delegate to core — identify TP/FP discriminating feature."""
    return await _dd_agents.run_contrastive_feature_analysis(
        tp_cases, fp_cases, candidate_rule, pair_info, config=_BIRDS_CONFIG,
        model=model or ACTIVE_MODEL, call_agent_fn=call_agent,
    )


async def run_rule_spectrum_generator(
    candidate_rule: dict,
    tp_cases: list[dict],
    fp_cases: list[dict],
    contrastive_result: dict,
    pair_info: dict,
    model: str = "",
) -> tuple[list[dict], int]:
    """Delegate to core — generate 4-level specificity spectrum."""
    return await _dd_agents.run_rule_spectrum_generator(
        candidate_rule, tp_cases, fp_cases, contrastive_result, pair_info,
        config=_BIRDS_CONFIG, model=model or ACTIVE_MODEL, call_agent_fn=call_agent,
    )


async def run_rule_completer(
    candidate_rule: dict,
    pair_info: dict,
    model: str = "",
) -> tuple[dict, int]:
    """Delegate to core — fill in implicit background conditions."""
    return await _dd_agents.run_rule_completer(
        candidate_rule, pair_info, config=_BIRDS_CONFIG,
        model=model or ACTIVE_MODEL, call_agent_fn=call_agent,
    )


async def run_semantic_rule_validator(
    candidate_rule: dict,
    pair_info: dict,
    model: str = "",
) -> tuple[dict, int]:
    """Delegate to core — text-only semantic check."""
    return await _dd_agents.run_semantic_rule_validator(
        candidate_rule, pair_info, config=_BIRDS_CONFIG,
        model=model or ACTIVE_MODEL, call_agent_fn=call_agent,
    )


async def run_rule_reviser(
    candidate_rule: dict,
    contrastive_result: dict,
    tp_cases: list[dict],
    fp_cases: list[dict],
    pair_info: dict,
    model: str = "",
) -> tuple[dict, int]:
    """Delegate to core — add tightening pre-condition."""
    return await _dd_agents.run_rule_reviser(
        candidate_rule, contrastive_result, tp_cases, fp_cases, pair_info,
        config=_BIRDS_CONFIG, model=model or ACTIVE_MODEL, call_agent_fn=call_agent,
    )
