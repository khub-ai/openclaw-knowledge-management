# PUPIL Domain Readiness Probe

> A pre-flight check that determines whether a given PUPIL model has sufficient
> visual and verbal capability to benefit from Dialogic Distillation (DD) in a
> given domain — before any DD sessions are run.

**Code**: `core/dialogic_distillation/probe.py`

---

## The Problem

DD works by having an expert (TUTOR) describe visual features that distinguish
a confusable pair, then injecting those descriptions as explicit rules at
inference time. The assumption is that if the VALIDATOR can observe the
described feature, the PUPIL can too — and therefore a rule that activates on
that feature will help the PUPIL.

This assumption is unproven. Two distinct failure modes look identical from the
outside (a DD session that doesn't improve PUPIL accuracy), but have completely
different remedies:

| Failure mode | Cause | Remedy |
|---|---|---|
| **Vocabulary gap** | PUPIL can see the feature but didn't know to attend to it | More rounds of DD |
| **Perception barrier** | PUPIL cannot physically perceive the feature | Different PUPIL model or modality |
| **Comprehension failure** | PUPIL can see the feature but misinterprets the rule text | Simpler rule phrasing |
| **Instability** | PUPIL gives random answers on the same image | PUPIL is unsuitable for this domain |

Running a full DD session to discover a perception barrier wastes time and
API cost. The readiness probe diagnoses capability before DD begins, answering:

> **"Is this PUPIL in the right capability regime for DD to be applicable here?"**

---

## Probe Design: Five Assessment Steps

### Step 1 — TUTOR Expert Descriptions (cached)

The TUTOR model (typically Claude Opus) is asked to describe each probe image
in domain-specific vocabulary, focusing on what visually distinguishes the two
classes. These descriptions are cached — the same TUTOR outputs are reused
across all PUPIL models being tested, so Step 1 costs are incurred only once.

Output: `tutor_descriptions` — one rich description per probe image.

### Step 2 — PUPIL Free Vocabulary (not cached)

The PUPIL model is shown each probe image with a free-description prompt:
*"Describe what you see in this image. Focus on visual features relevant to
identifying the type of surface / lesion / object shown."*

The PUPIL's response is compared word-by-word against the TUTOR's vocabulary.
Vocabulary overlap is scored as the fraction of TUTOR domain terms that appear
in at least one PUPIL description across all images. A low overlap (< 0.20)
indicates a vocabulary gap — the PUPIL can't spontaneously generate the
expert terms, which is exactly the condition DD can address.

Output: `vocabulary_overlap` score ∈ [0, 1] and `pupil_vocab_sample` list.

### Step 3 — Feature Detection Queries (PUPIL only, not cached)

The TUTOR generates a set of specific yes/no feature detection queries, ordered
from easy to hard (configurable, default 12 queries). For example:

- Easy: *"Is the road surface visibly wet? Look for any darkening of the
  pavement or visible moisture."*
- Medium: *"Is there a uniform specular sheen covering the entire road surface,
  or only parts of it?"*
- Hard: *"Can you see individual aggregate particles (small stones embedded in
  the asphalt) clearly through any surface film?"*

The TUTOR-generated queries are cached. The PUPIL's binary answers (yes/no) are
compared against the expected answer for each image. Each query is applied to
all probe images, producing a per-feature detection rate and an overall
perception score.

The perception score is the fraction of feature queries the PUPIL answers
correctly. This directly measures what the VALIDATOR-based grounding check
cannot: whether **the PUPIL specifically** can see the features that expert
rules will refer to.

Output: `feature_profile` (per-feature dict) and `perception_score` ∈ [0, 1].

### Step 4 — Rule Comprehension Delta (PUPIL only, not cached)

The PUPIL is run twice on each probe image:

1. **Zero-shot**: classify the image with no rule, just the two class names
2. **Rule-aided**: classify with the seed rule (if provided) or the best feature
   description from Step 3 injected as a system-level instruction

The difference in accuracy between rule-aided and zero-shot is the
**rule comprehension delta**. A positive delta confirms that injecting
expert vocabulary actually changes the PUPIL's behaviour — the mechanism
DD relies on. A near-zero or negative delta indicates that the PUPIL ignores
injected instructions, making DD futile regardless of how well the rules are
authored.

Output: `zero_shot_accuracy`, `rule_aided_accuracy`, `rule_comprehension_delta`.

### Step 5 — Consistency (PUPIL only, not cached)

The PUPIL is run `_CONSISTENCY_REPEATS` times (default 3) on the same subset
of probe images with the same prompt. The consistency score is the fraction of
runs that agree with the majority answer. A consistency score below 0.50 means
the PUPIL is essentially random on this domain, which no amount of rule
injection will fix.

Output: `consistency_score` ∈ [0, 1].

---

## Verdict Logic

```
VERDICT_GO      = "go"
VERDICT_PARTIAL = "partial"
VERDICT_NO_GO   = "no-go"

Thresholds:
  _GO_PERCEPTION_MIN         = 0.60   # can observe most features
  _GO_RULE_COMPREHENSION_MIN = 0.15   # rule injection improves accuracy ≥ 15pp
  _GO_CONSISTENCY_MIN        = 0.75   # ≥ 75% consistent answers

  _NOGO_PERCEPTION_MAX       = 0.30   # perception barrier — no-go regardless
  _NOGO_CONSISTENCY_MAX      = 0.50   # too random — no-go regardless
```

Decision tree:

```
if perception_score < 0.30 OR consistency_score < 0.50:
    verdict = "no-go"           # hard floors — perception barrier or instability

elif (perception_score >= 0.60 AND
      rule_comprehension_delta >= 0.15 AND
      consistency_score >= 0.75):
    verdict = "go"              # all three dimensions cleared

else:
    verdict = "partial"         # DD may work with heavier rule engineering
```

The report also includes:
- `weak_points` — list of specific issues (e.g. "low hard-feature detection",
  "rule injection not improving accuracy")
- `recommendations` — actionable suggestions tailored to the diagnosis

---

## Caching Strategy

TUTOR and VALIDATOR outputs are **cached** at two levels:

1. **In-memory cache** (`_PROBE_MEM_CACHE`) — persists for the lifetime of the
   Python process. A second `probe()` call with the same model + images reuses
   all cached TUTOR/VALIDATOR responses at zero API cost.

2. **Disk cache** (enabled by setting `LLM_CACHE_DIR` environment variable) —
   responses are pickled to `$LLM_CACHE_DIR/probe/<sha256_key>.pkl`. Persist
   across process restarts and batch probe runs overnight.

PUPIL outputs are **never cached** — the whole point is to observe the PUPIL's
actual behaviour fresh. Caching PUPIL outputs would defeat the probe's purpose.

Cache key:
```python
def _probe_cache_key(model, role, image_hash, prompt):
    raw = f"{model}\x00{role}\x00{image_hash}\x00{prompt}"
    return hashlib.sha256(raw.encode()).hexdigest()
```

`image_hash` is the MD5 of the raw image bytes (from `ProbeImage.image_hash`).
This means caches are invalidated automatically if the image file changes.

### Clearing the cache

```python
from core.dialogic_distillation.probe import clear_probe_cache

clear_probe_cache()           # clear in-memory only
clear_probe_cache(disk=True)  # clear in-memory + all disk .pkl files
```

---

## Cost Tracking

Per-role token usage and estimated USD cost are accumulated separately for
TUTOR, VALIDATOR, and PUPIL. Costs are estimated from model name strings using
a built-in pricing table (Anthropic April 2026 rates; Together/OpenRouter open
weights).

```python
from core.dialogic_distillation.probe import get_probe_costs, reset_probe_costs

reset_probe_costs()

report = await probe(...)

costs = get_probe_costs()
# {
#   "TUTOR":     {"role": "TUTOR", "input_tokens": 4210, "output_tokens": 820,
#                 "api_calls": 10, "cost_usd": 0.0684},
#   "VALIDATOR": {"role": "VALIDATOR", ...},
#   "PUPIL":     {"role": "PUPIL", ...},
# }
```

Typical cost profile for 10 probe images (5 per class):

| Role | API calls | Typical cost |
|---|---|---|
| TUTOR (Opus 4.6) | ~10 | ~$0.05–0.10 |
| VALIDATOR (Sonnet 4.6) | ~12 | ~$0.01–0.03 |
| PUPIL (Qwen3-VL-8B) | ~70 | ~$0.002–0.01 |

Cached TUTOR/VALIDATOR calls show `cost_usd = 0.0` (served from cache).

---

## API Reference

### `ProbeImage`

```python
@dataclass
class ProbeImage:
    path: str                          # absolute path to JPEG
    true_class: str                    # ground-truth class label
    difficulty: str = "medium"         # "easy" | "medium" | "hard"
    notes: str = ""                    # optional human annotation

    @property
    def image_hash(self) -> str: ...   # MD5 of image bytes
```

### `ProbeRoleCosts`

```python
@dataclass
class ProbeRoleCosts:
    role: str
    input_tokens:  int   = 0
    output_tokens: int   = 0
    api_calls:     int   = 0
    cost_usd:      float = 0.0
```

### `probe()`

```python
async def probe(
    pupil_model:     str,
    tutor_model:     str,
    validator_model: str,
    domain_config:   DomainConfig,
    probe_images:    list[ProbeImage],
    pair_info:       dict,
    seed_rule:       dict | None = None,
    call_agent_fn:   Callable | None = None,
    n_feature_queries: int = 12,
    console:         Console | None = None,
) -> dict:
```

Returns a `report` dict with keys:

| Key | Type | Description |
|---|---|---|
| `verdict` | str | `"go"` / `"partial"` / `"no-go"` |
| `perception_score` | float | Fraction of feature queries answered correctly |
| `feature_profile` | dict | Per-feature detection rates |
| `vocabulary_overlap` | float | Fraction of TUTOR terms in PUPIL vocabulary |
| `zero_shot_accuracy` | float | PUPIL accuracy without rule injection |
| `rule_aided_accuracy` | float | PUPIL accuracy with rule injected |
| `rule_comprehension_delta` | float | `rule_aided - zero_shot` |
| `consistency_score` | float | Fraction of runs giving consistent answer |
| `weak_points` | list[str] | Identified capability gaps |
| `recommendations` | list[str] | Actionable next steps |
| `costs` | dict | Per-role cost summaries |
| `tutor_descriptions` | list[str] | Expert descriptions per image |
| `pupil_vocab_sample` | list[str] | Sample PUPIL description excerpts |
| `n_images` | int | Number of probe images processed |

### `save_report()` / `load_report()`

```python
def save_report(report: dict, path: str | Path) -> None: ...
def load_report(path: str | Path) -> dict: ...
```

### Cache management

```python
def clear_probe_cache(disk: bool = False) -> None: ...
def get_probe_costs() -> dict[str, dict]: ...
def reset_probe_costs() -> None: ...
```

---

## Usage Example

```python
import asyncio
from pathlib import Path
from core.dialogic_distillation.probe import (
    probe, ProbeImage, get_probe_costs, reset_probe_costs,
    save_report, clear_probe_cache,
)
from usecases.image_classification.road_surface.python.domain_config import (
    ROAD_SURFACE_CONFIG,
)

# Collect probe images (5–15 per class recommended)
probe_images = [
    ProbeImage("images/dry_001.jpg",   "Dry Road",   difficulty="easy"),
    ProbeImage("images/dry_002.jpg",   "Dry Road",   difficulty="medium"),
    ProbeImage("images/dry_003.jpg",   "Dry Road",   difficulty="hard"),
    ProbeImage("images/wet_001.jpg",   "Wet Road",   difficulty="easy"),
    ProbeImage("images/wet_002.jpg",   "Wet Road",   difficulty="medium"),
    ProbeImage("images/wet_003.jpg",   "Wet Road",   difficulty="hard"),
]

pair_info = {
    "pair_id":   "dry_vs_wet",
    "class_a":   "Dry Road",
    "class_b":   "Wet Road",
    "friction_a": "dry",
    "friction_b": "wet",
}

reset_probe_costs()

report = asyncio.run(probe(
    pupil_model     = "qwen/qwen3-vl-8b-instruct",
    tutor_model     = "claude-opus-4-6",
    validator_model = "claude-sonnet-4-6",
    domain_config   = ROAD_SURFACE_CONFIG,
    probe_images    = probe_images,
    pair_info       = pair_info,
))

print(f"Verdict: {report['verdict']}")
print(f"Perception score: {report['perception_score']:.2f}")
print(f"Rule comprehension delta: {report['rule_comprehension_delta']:+.2f}")
print(f"Consistency: {report['consistency_score']:.2f}")
print(f"Weak points: {report['weak_points']}")

costs = get_probe_costs()
for role, c in costs.items():
    print(f"  {role}: {c['api_calls']} calls, ${c['cost_usd']:.4f}")

save_report(report, "probe_dry_vs_wet_qwen3vl8b.json")

# Second PUPIL model — TUTOR/VALIDATOR responses served from cache
report2 = asyncio.run(probe(
    pupil_model     = "llava-hf/llava-1.5-7b-hf",
    tutor_model     = "claude-opus-4-6",
    validator_model = "claude-sonnet-4-6",
    domain_config   = ROAD_SURFACE_CONFIG,
    probe_images    = probe_images,
    pair_info       = pair_info,
))
# TUTOR/VALIDATOR cost_usd ≈ 0 for second run (served from cache)
```

---

## Probe vs Patchability Framework

The readiness probe and the patchability framework (`docs/patchability.md`) are
complementary:

| | Patchability framework | Readiness probe |
|---|---|---|
| **Focus** | Is DD applicable to this *domain*? | Is DD applicable to this *PUPIL model* in this domain? |
| **Method** | Expert assessment + zero-shot B score | Automated 5-step structured test |
| **Time** | ~1 hour manual | ~10 minutes automated |
| **Output** | Go/no-go for domain experiment | Go/partial/no-go for specific PUPIL model |
| **Dimension covered** | G, V, B, E (domain-level) | Perception (G at PUPIL level), comprehension, consistency |

Run patchability first to decide whether to invest in a domain at all.
Run the readiness probe to select which PUPIL model to use (or to confirm
your chosen PUPIL is suitable) before spending on full DD sessions.

---

## Open Questions

1. **Optimal probe image count**: How many images are needed for a reliable
   verdict? Preliminary estimate is 5–15 per class. Too few → unreliable
   feature detection rates. Too many → unnecessary cost.

2. **Feature query generation quality**: The TUTOR-generated queries in Step 3
   may not cover all discriminating features. A manual review of the generated
   queries is recommended for high-stakes probe runs.

3. **Seed rule sensitivity**: The rule comprehension delta in Step 4 depends on
   the quality of the seed rule. If no seed rule is provided, the probe uses the
   best feature description from Step 3, which may be weaker than a real
   DD-authored rule.

4. **PUPIL model randomness**: Some open-weight models have high temperature
   by default. Step 5 (consistency) is sensitive to sampling temperature. If
   a `temperature=0` option is available for the PUPIL, using it gives a cleaner
   upper-bound consistency signal.
