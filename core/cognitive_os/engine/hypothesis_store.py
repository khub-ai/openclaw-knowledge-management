"""Hypothesis store — lifecycle operations over WorldState.hypotheses.

This module provides the operations that constitute the learning loop at
the hypothesis level: proposing new hypotheses, deduplicating exact
matches, linking canonical competitors, applying evidence-driven
credence updates, decaying stale claims, and pruning abandoned ones.

All functions are **module-level and stateless** — they read and mutate
:class:`WorldState` in place.  No hidden indexes, no class instance
state; the store's entire state lives in ``ws.hypotheses`` plus the
monotonic counter ``ws._next_hypothesis_id``.  This keeps
:class:`WorldState` snapshottable for persistence and testing.

Evidence matching
-----------------
``update_credence_from_events`` dispatches on ``(event_type, claim_type)``
to decide whether a given event supports, contradicts, or is neutral
with respect to each active hypothesis.  Phase 2 implements matchers
for the three claim types that directly consume Events:

* :class:`PropertyClaim`   — EntityStateChanged
* :class:`TransitionClaim` — AgentMoved + action taken
* :class:`CausalClaim`     — trigger evaluated at step t-delay,
                              effect evaluated at step t

Evidence for the remaining claim types comes from sources outside
Events:

* :class:`RelationalClaim` / :class:`StructureMappingClaim`
      — Observer answers (wired in Phase 4+)
* :class:`ConstraintClaim`
      — planner experience (Phase 3+)
* :class:`StrategyClaim`
      — branch-outcome statistics from executed Plans (Phase 3+)

Those matchers return ``None`` (neutral) here and will be extended in
the phases that introduce their evidence sources.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

from .claims import (
    Claim,
    CausalClaim,
    ConstraintClaim,
    PropertyClaim,
    RelationalClaim,
    StrategyClaim,
    StructureMappingClaim,
    TransitionClaim,
)
from .conditions import Condition
from .credence import (
    Credence,
    apply_decay,
    link_competitor,
    unlink_competitor,
    update_on_contradict,
    update_on_support,
)
from .types import (
    AgentDied,
    AgentMoved,
    EntityStateChanged,
    Event,
    Hypothesis,
    ResourceChanged,
    Scope,
    SurpriseEvent,
    WorldState,
)


# ---------------------------------------------------------------------------
# Internal helpers — ID generation, indexing
# ---------------------------------------------------------------------------


def _next_id(ws: WorldState, prefix: str = "h") -> str:
    """Allocate a new, never-before-used hypothesis ID.

    IDs are monotonic — pruned IDs are never re-used.  This matters
    because lattice links (``parent_id`` / ``child_ids``) would
    otherwise silently re-target if a pruned ID were reallocated to a
    different claim.
    """
    n = ws._next_hypothesis_id
    ws._next_hypothesis_id = n + 1
    return f"{prefix}{n}"


def by_canonical_key(ws: WorldState, canonical_key: tuple) -> List[Hypothesis]:
    """Return all active hypotheses whose claim shares the given
    canonical key.  These are structural peers / competitors."""
    return [h for h in ws.hypotheses.values()
            if h.claim.canonical_key() == canonical_key]


def by_full_key(ws: WorldState, full_key: tuple) -> Optional[Hypothesis]:
    """Return the active hypothesis with the given full key, if any.
    At most one should exist (invariant maintained by :func:`propose`)."""
    for h in ws.hypotheses.values():
        if h.claim.full_key() == full_key:
            return h
    return None


# ---------------------------------------------------------------------------
# Public: propose a hypothesis
# ---------------------------------------------------------------------------


def propose(ws:                WorldState,
            claim:              Claim,
            source:             str,
            scope:              Scope,
            step:               int,
            *,
            rationale:          Optional[str] = None,
            expires_at:         Optional[int] = None,
            initial_credence:   Optional[float] = None,
            parent_id:          Optional[str] = None) -> str:
    """Propose a hypothesis; returns the hypothesis ID.

    Dedup / competition pipeline:

    1. **Exact duplicate** (same ``full_key``) — merge evidence by
       treating this proposal as a supporting confirmation of the
       existing hypothesis; return the existing ID unchanged.

    2. **Canonical competitor** (same ``canonical_key``, different
       ``full_key``) — register the new hypothesis and link it
       bidirectionally with every existing competitor via
       ``Credence.competing``.  Evidence that supports one member of
       the group can inform the others (implemented by miners in
       Phase 4).

    3. **Novel** — register a fresh hypothesis with source-prior
       credence.

    Parameters
    ----------
    ws
        WorldState to mutate.
    claim
        The Claim to register.
    source
        Provenance tag, e.g. ``"miner:FutilePattern"``,
        ``"user:correction"``.  Used to look up the initial credence
        prior from :class:`SourcePriors`.
    scope
        Temporal and structural scope of the hypothesis.
    step
        Current step (used as ``created_at`` and ``last_confirmed``).
    rationale
        Optional human-readable explanation, for audit and logging.
    expires_at
        Optional step at which the hypothesis should be re-evaluated.
    initial_credence
        If given, overrides the source-prior lookup.  Rarely used;
        reserved for adapter-seeded claims where the adapter has
        specific confidence that differs from the generic source prior.
    parent_id
        If this proposal is a specialisation of an existing
        hypothesis, the parent's ID.  The function will wire the
        lattice links (``parent.child_ids``, ``new.parent_id``).
    """
    # --- (1) exact duplicate — merge evidence ---
    existing_exact = by_full_key(ws, claim.full_key())
    if existing_exact is not None:
        cfg = _credence_cfg(ws)
        existing_exact.credence = update_on_support(
            existing_exact.credence, step, cfg,
            source_strength=_source_strength(source))
        existing_exact.supporting_steps.append(step)
        return existing_exact.id

    # --- initial credence from source prior (or override) ---
    if initial_credence is None:
        initial_credence = _initial_credence_from_source(ws, source)
    cred = Credence(point=initial_credence, last_confirmed=step)

    # --- (2) canonical competitors — link bidirectionally ---
    canonical_peers = by_canonical_key(ws, claim.canonical_key())
    # These are all competitors of the new hypothesis.

    # --- (3) create and install ---
    h_id = _next_id(ws)
    h = Hypothesis(
        id                  = h_id,
        claim               = claim,
        credence            = cred,
        scope               = scope,
        source              = source,
        supporting_steps    = [step],      # the proposal itself is a datum
        contradicting_steps = [],
        expires_at          = expires_at,
        parent_id           = parent_id,
        child_ids           = [],
        created_at          = step,
        rationale           = rationale,
    )

    # Link competitors
    if canonical_peers:
        peer_ids = tuple(p.id for p in canonical_peers)
        h.credence = replace(h.credence, competing=peer_ids)
        for peer in canonical_peers:
            peer.credence = link_competitor(peer.credence, h_id)

    ws.hypotheses[h_id] = h

    # Wire lattice link up to parent, if any
    if parent_id is not None and parent_id in ws.hypotheses:
        parent = ws.hypotheses[parent_id]
        if h_id not in parent.child_ids:
            parent.child_ids.append(h_id)

    return h_id


# ---------------------------------------------------------------------------
# Public: evidence-driven credence updates
# ---------------------------------------------------------------------------


def update_credence_from_events(ws:     WorldState,
                                events: List[Event],
                                step:   int) -> Dict[str, List[str]]:
    """Scan all active hypotheses against a batch of events.

    For each (hypothesis, event) pair, consult the event→claim matcher
    to decide whether the event supports, contradicts, or is neutral
    toward the hypothesis.  Apply the corresponding credence update.

    Returns a summary dict:

    * ``"newly_committed"`` — hypothesis IDs that crossed the
      commit threshold during this call.
    * ``"newly_demoted"``   — hypothesis IDs that dropped below the
      commit threshold during this call.
    * ``"supported"``       — (h_id, event_step) pairs that received
      supporting evidence.
    * ``"contradicted"``    — (h_id, event_step) pairs that received
      contradicting evidence.

    The planner reads ``newly_demoted`` to decide whether a current
    plan's ``assumptions`` are still valid; the refinement layer reads
    ``contradicted`` to decide whether to propose specialisations.
    """
    cfg = _credence_cfg(ws)
    before_committed = {h_id for h_id, h in ws.hypotheses.items()
                        if h.credence.is_committed(cfg)}

    supported:     List[Tuple[str, int]] = []
    contradicted:  List[Tuple[str, int]] = []

    for h_id, h in list(ws.hypotheses.items()):
        for evt in events:
            verdict = event_evidence_for_claim(evt, h.claim, ws)
            if verdict is True:
                h.credence = update_on_support(
                    h.credence, step, cfg,
                    source_strength=1.0)
                h.supporting_steps.append(getattr(evt, "step", step))
                supported.append((h_id, getattr(evt, "step", step)))
            elif verdict is False:
                h.credence = update_on_contradict(
                    h.credence, step, cfg,
                    strength=1.0)
                h.contradicting_steps.append(getattr(evt, "step", step))
                contradicted.append((h_id, getattr(evt, "step", step)))
            # verdict is None: neutral — no change

    after_committed = {h_id for h_id, h in ws.hypotheses.items()
                       if h.credence.is_committed(cfg)}

    return {
        "newly_committed": sorted(after_committed - before_committed),
        "newly_demoted":   sorted(before_committed - after_committed),
        "supported":       supported,
        "contradicted":    contradicted,
    }


# ---------------------------------------------------------------------------
# Public: decay and pruning
# ---------------------------------------------------------------------------


def apply_staleness_decay_all(ws: WorldState, step: int) -> Dict[str, List[str]]:
    """Apply staleness decay to every active hypothesis.

    Returns the same commit/demote summary as
    :func:`update_credence_from_events` so the caller can detect plans
    that lost their supporting hypotheses to decay rather than
    contradiction.
    """
    cfg = _credence_cfg(ws)
    before_committed = {h_id for h_id, h in ws.hypotheses.items()
                        if h.credence.is_committed(cfg)}
    for h in ws.hypotheses.values():
        h.credence = apply_decay(h.credence, step, cfg)
    after_committed = {h_id for h_id, h in ws.hypotheses.items()
                       if h.credence.is_committed(cfg)}
    return {
        "newly_committed": sorted(after_committed - before_committed),
        "newly_demoted":   sorted(before_committed - after_committed),
    }


def prune_abandoned(ws: WorldState, step: int) -> List[str]:
    """Remove hypotheses whose credence has fallen at or below the
    abandon threshold.  Returns the list of pruned IDs.

    Lattice hygiene: when a pruned hypothesis has children, the
    children's ``parent_id`` is cleared to ``None`` (they become
    top-level rather than orphaned).  When it has parents, the pruned
    ID is removed from ``parent.child_ids``.  Competitor linkage is
    cleaned up in all peers' ``Credence.competing`` lists.
    """
    cfg = _credence_cfg(ws)
    pruned: List[str] = []

    for h_id, h in list(ws.hypotheses.items()):
        if h.credence.is_abandoned(cfg):
            _remove_with_cleanup(ws, h_id)
            pruned.append(h_id)

    return pruned


def _remove_with_cleanup(ws: WorldState, h_id: str) -> None:
    """Delete the hypothesis and repair all structural references."""
    h = ws.hypotheses.pop(h_id, None)
    if h is None:
        return

    # Unlink from parent's child list
    if h.parent_id is not None:
        parent = ws.hypotheses.get(h.parent_id)
        if parent is not None and h_id in parent.child_ids:
            parent.child_ids.remove(h_id)

    # Orphan children (don't cascade delete — they may still be valid)
    for child_id in list(h.child_ids):
        child = ws.hypotheses.get(child_id)
        if child is not None and child.parent_id == h_id:
            child.parent_id = None

    # Unlink from competitors
    for peer_id in h.credence.competing:
        peer = ws.hypotheses.get(peer_id)
        if peer is not None:
            peer.credence = unlink_competitor(peer.credence, h_id)


# ---------------------------------------------------------------------------
# Public: queries
# ---------------------------------------------------------------------------


def committed(ws: WorldState) -> List[Hypothesis]:
    """Return hypotheses whose credence is at or above commit threshold.

    Equivalent to :meth:`WorldState.committed_hypotheses` but available
    at module level so callers can work symmetrically with other
    hypothesis_store operations.
    """
    cfg = _credence_cfg(ws)
    return [h for h in ws.hypotheses.values() if h.credence.is_committed(cfg)]


def contested_groups(ws: WorldState) -> List[List[Hypothesis]]:
    """Group competing hypotheses.  Each returned inner list is a
    set of hypotheses that share a canonical key — these compete for
    the same phenomenon.  Singleton groups (only one hypothesis with
    a given canonical key) are excluded from the result.

    Used by the explorer to find discrimination-worthy exploration
    targets, and by generalisation miners to find parameter-learning
    situations that have converged.
    """
    groups: Dict[tuple, List[Hypothesis]] = {}
    for h in ws.hypotheses.values():
        key = h.claim.canonical_key()
        groups.setdefault(key, []).append(h)
    return [g for g in groups.values() if len(g) > 1]


# ---------------------------------------------------------------------------
# Internal: source strength + initial credence
# ---------------------------------------------------------------------------


def _source_strength(source: str) -> float:
    """How strong a piece of supporting evidence is, based on its source.

    A user correction confirming a hypothesis is stronger evidence than
    a speculative LLM proposal confirming the same claim.  This uses
    the same prior-strength convention as the source priors; a source
    prior of 0.9 means supporting evidence from that source carries
    weight 0.9 in the learning-rate multiplier.
    """
    # Delegate to SourcePriors-equivalent logic; values are in [0,1].
    return max(0.0, min(1.0, _prior_for_source(source)))


def _prior_for_source(source: str) -> float:
    """Wrapper so we can look up priors without importing EngineConfig."""
    kind = source.split(":", 1)[0]
    return {
        "user":      0.95,
        "adapter":   0.80,
        "observer":  0.70,
        "miner":     0.60,
        "analogy":   0.40,
        "llm":       0.30,
        "abductive": 0.25,
    }.get(kind, 0.5)


def _initial_credence_from_source(ws: WorldState, source: str) -> float:
    """Look up initial credence for a source, preferring the engine
    config's :class:`SourcePriors` if available, falling back to
    hard-coded defaults if no config is attached yet.

    The fallback exists because some test and construction paths build
    a :class:`WorldState` without an :class:`EngineConfig`.
    """
    if ws.config is not None and hasattr(ws.config, "source_priors"):
        return ws.config.source_priors.for_source(source)
    return _prior_for_source(source)


def _credence_cfg(ws: WorldState):
    """Return the :class:`CredenceConfig` from ``ws.config``, falling
    back to a default instance when config isn't attached (test paths).
    """
    if ws.config is not None and hasattr(ws.config, "credence"):
        return ws.config.credence
    from .config import CredenceConfig
    return CredenceConfig()


# ---------------------------------------------------------------------------
# Evidence matching — (event, claim) → True / False / None
# ---------------------------------------------------------------------------


def event_evidence_for_claim(evt:   Event,
                             claim: Claim,
                             ws:    WorldState) -> Optional[bool]:
    """Decide whether an event constitutes supporting (True),
    contradicting (False), or neutral (None) evidence for a claim.

    Dispatch table keyed on claim type.  Unknown combinations fall
    through to ``None`` (neutral) — the safe default for claim types
    whose evidence arrives from non-event sources (Observer answers,
    plan outcomes).
    """
    if isinstance(claim, PropertyClaim):
        return _evidence_for_property(evt, claim)
    if isinstance(claim, TransitionClaim):
        return _evidence_for_transition(evt, claim, ws)
    if isinstance(claim, CausalClaim):
        return _evidence_for_causal(evt, claim, ws)
    # RelationalClaim, StructureMappingClaim, ConstraintClaim,
    # StrategyClaim — evidence comes from other channels; neutral here.
    return None


def _evidence_for_property(evt:   Event,
                           claim: PropertyClaim) -> Optional[bool]:
    """PropertyClaim(entity, prop, value) matches EntityStateChanged
    events targeting the same (entity, property).

    * New value matches the claim's value → supporting.
    * New value differs from claim's value → contradicting.
    * Unrelated event → neutral.
    """
    if not isinstance(evt, EntityStateChanged):
        return None
    if evt.entity_id != claim.entity_id:
        return None
    if evt.property != claim.property:
        return None
    # (new value == claim.value) supports; otherwise contradicts
    return evt.new == claim.value


def _evidence_for_transition(evt:   Event,
                             claim: TransitionClaim,
                             ws:    WorldState) -> Optional[bool]:
    """TransitionClaim(action, pre, post) matches events that record
    the action's execution and its observed post-condition.

    Phase 2 implements a narrow matcher: an :class:`AgentMoved` event
    whose action name matches ``claim.action``, evaluated against the
    claim's ``post`` condition in the current WorldState.  The
    ``pre`` side is not re-checked here — the episode runner is
    responsible for only presenting this hypothesis with events where
    the action was actually invoked in a state satisfying ``pre``.

    Richer matchers (resource-changing transitions, entity-state
    transitions) will be layered in when the corresponding action
    event types are introduced.
    """
    # Hook for future extension: transition events not yet modelled in
    # the Event vocabulary fall through as neutral.
    if not isinstance(evt, AgentMoved):
        return None
    # In Phase 2 we don't have action labels on AgentMoved; treat any
    # AgentMoved as evidence only when claim.action is the convention
    # "MOVE" or unset.  Proper action-tagging arrives with the
    # adapter protocol in Phase 4.
    if claim.action not in ("MOVE", "*", ""):
        return None
    # Evaluate the post condition after the move.
    post_truth = claim.post.evaluate(ws)
    if post_truth is None:
        return None
    return post_truth


def _evidence_for_causal(evt:   Event,
                         claim: CausalClaim,
                         ws:    WorldState) -> Optional[bool]:
    """CausalClaim(trigger, effect, min_occurrences, delay) evidence.

    Phase 2 implements the simplest form: when the trigger condition
    is evaluable-and-true in the current WorldState, check whether the
    effect condition is evaluable-and-true.  The ``min_occurrences``
    and ``delay`` parameters are informational in Phase 2 and will be
    used by specialised miners in Phase 4 that track trigger counts
    across the observation history.

    * trigger true + effect true  → supporting.
    * trigger true + effect false → contradicting (hypothesis
      predicted an effect that did not occur).
    * trigger false or unknown    → neutral.

    The ``evt`` argument is currently only used as a "heartbeat" — we
    re-check the trigger/effect on any event, relying on the
    per-step invocation cadence in the runner.
    """
    trig = claim.trigger.evaluate(ws)
    if trig is not True:
        return None
    eff = claim.effect.evaluate(ws)
    if eff is None:
        return None
    return bool(eff)
