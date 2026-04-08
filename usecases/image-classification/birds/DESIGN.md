# Birds Use Case — Developer Design Spec

This document is the developer-facing reference for the birds dialogic patching
experiment. For the user-facing walkthrough, see [README.md](README.md). For the
cross-domain design spec covering both birds and dermatology, see
[../DESIGN.md](../DESIGN.md).

---

## Contents

1. [Repository Layout](#1-repository-layout)
2. [Data](#2-data)
3. [API Keys](#3-api-keys)
4. [How to Run](#4-how-to-run)
5. [Output Files](#5-output-files)
6. [Design Issues and Known Limitations](#6-design-issues-and-known-limitations)
7. [Available Confusable Pairs](#7-available-confusable-pairs)
8. [Architecture Notes](#8-architecture-notes)

---

## 1. Repository Layout

```
usecases/image-classification/
  birds/
    README.md                          # user-facing walkthrough
    DESIGN.md                          # this file
    knowledge_base/
      bronzed_cowbird_vs_shiny_cowbird.json   # registered rules from completed session
    python/
      patch.py                         # dialogic patching loop (main entry point)
      agents.py                        # bird-vocabulary LLM agent calls
      results_baseline_qwen3vl8b_cowbird.json  # saved zero-shot baseline
      patch_rules_birds_test.json      # rules registered in test session
      patch_session_birds_test.json    # full session record from test run
      tutor_inbox/                     # human-in-the-loop request files
      tutor_outbox/                    # human-in-the-loop response files
  python/                              # shared dataset + pair registry (not birds-specific)
    dataset.py                         # CUBDataset loader
    confusable_pairs.py                # CONFUSABLE_PAIRS registry
    agents.py                          # shared call_agent / OpenRouter infrastructure
    ...
  src/
    dataset.py                         # alternative dataset interface (src/ variant)
    confusable_pairs.py
    ...
  assets/
    birds/
      bronzed_cowbird.jpg              # species reference images (used in README)
      shiny_cowbird.jpg
      bronzed_0019_fixed.jpg           # failure images from test session
      bronzed_0061_fixed.jpg
      bronzed_0081_unfixed.jpg
      shiny_0080_unfixed.jpg
      ...
```

`birds/python/agents.py` does **not** import from `image-classification/python/agents.py`
via the normal module system. It loads the dermatology agents at runtime via
`importlib.util.spec_from_file_location` to avoid Python resolving "agents" as the
birds file itself during its own loading. See §8 for details.

---

## 2. Data

### CUB-200-2011 dataset

| Property | Value |
|---|---|
| Default location | `C:\_backup\ml\data\_tmp\CUB_200_2011\CUB_200_2011` |
| Download | [Caltech Perona Lab](https://data.caltech.edu/records/65de6-vp158) |
| Size | ~1.1 GB, 11,788 images |
| Classes | 200 North American bird species |

The default path is set in `image-classification/python/dataset.py`:

```python
DEFAULT_DATA_DIR = Path(r"C:\_backup\ml\data\_tmp\CUB_200_2011\CUB_200_2011")
```

Override at runtime with `--data-dir /your/path/to/CUB_200_2011`.

**Expected directory structure inside `CUB_200_2011/`:**

```
images/
  001.Black_footed_Albatross/
  002.Laysan_Albatross/
  ...
  026.Bronzed_Cowbird/
  027.Shiny_Cowbird/
  ...
image_class_labels.txt
images.txt
train_test_split.txt
classes.txt
```

The loader reads `images.txt`, `image_class_labels.txt`, and `train_test_split.txt`
to build the dataset. No pre-processing is required.

### What the test session used

The saved baseline (`results_baseline_qwen3vl8b_cowbird.json`) contains absolute
Windows paths. If your data is in a different location, re-generate the baseline
rather than editing the file:

```bash
python patch.py \
  --cheap-model "qwen/qwen3-vl-8b-instruct" \
  --expert-model "claude-sonnet-4-6" \
  --pair bronzed_cowbird_vs_shiny_cowbird \
  --max-per-class 3 \
  --data-dir /your/CUB_200_2011
```

---

## 3. API Keys

The script looks for keys at `P:/_access/Security/api_keys.env` (a private file on
the developer's machine). If that file doesn't exist, keys must be set as environment
variables before running.

| Key | Used for |
|---|---|
| `ANTHROPIC_API_KEY` | Expert model (Claude) and validator model (Claude) calls |
| `OPENROUTER_API_KEY` | Cheap model calls (Qwen3-VL-8B, and any other OpenRouter model) |
| `OPENAI_API_KEY` | Optional — only needed if using an OpenAI model as cheap or expert |

Set them in your shell before running:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENROUTER_API_KEY=sk-or-...
```

Or add them to a `.env` file and source it. The script does not auto-load a `.env`
in the working directory — only the `P:/` path is auto-loaded.

---

## 4. How to Run

All commands run from `usecases/image-classification/birds/python/`.

### 4.1 Full patch loop from scratch (fresh baseline + patching)

```bash
python patch.py \
  --cheap-model "qwen/qwen3-vl-8b-instruct" \
  --expert-model "claude-sonnet-4-6" \
  --validator-model "claude-sonnet-4-6" \
  --pair bronzed_cowbird_vs_shiny_cowbird \
  --max-per-class 3 \
  --max-val-per-class 4 \
  --max-authoring-per-class 4 \
  --max-confirm-per-class 0 \
  --patch-rules my_rules.json \
  --output my_session.json
```

This runs the zero-shot baseline, patches all failures, and saves results. Skipping
`--skip-rerun` means the patched model is tested on each failure as rules are registered.

### 4.2 Load saved baseline, run patch loop only

```bash
python patch.py \
  --failures-from results_baseline_qwen3vl8b_cowbird.json \
  --cheap-model "qwen/qwen3-vl-8b-instruct" \
  --expert-model "claude-sonnet-4-6" \
  --validator-model "claude-sonnet-4-6" \
  --pair bronzed_cowbird_vs_shiny_cowbird \
  --max-per-class 3 \
  --max-val-per-class 4 \
  --max-authoring-per-class 4 \
  --max-confirm-per-class 0 \
  --patch-rules my_rules.json \
  --output my_session.json
```

This skips the zero-shot baseline step and re-uses saved failures. Useful for
repeating the patching without burning API budget on re-running the cheap model.

### 4.3 Rerun only — apply saved rules to saved failures

```bash
python patch.py \
  --rerun-only \
  --failures-from results_baseline_qwen3vl8b_cowbird.json \
  --cheap-model "qwen/qwen3-vl-8b-instruct" \
  --expert-model "claude-sonnet-4-6" \
  --patch-rules patch_rules_birds_test.json
```

This applies the registered rules to the saved failure images and reports how many
are fixed. No expert calls are made. The expert model is still required as an
argument (used if needed for model routing) but is not called.

### 4.4 Dry run — author and validate but do not register

```bash
python patch.py \
  --failures-from results_baseline_qwen3vl8b_cowbird.json \
  --cheap-model "qwen/qwen3-vl-8b-instruct" \
  --expert-model "claude-sonnet-4-6" \
  --validator-model "claude-sonnet-4-6" \
  --pair bronzed_cowbird_vs_shiny_cowbird \
  --dry-run \
  --output dry_run_session.json
```

Runs the full loop (authoring, completion, semantic validation, held-out gate) but
does not write rules to `--patch-rules` and does not perform the per-failure rerun.

### 4.5 Human-in-the-loop (claude-tutor expert)

```bash
python patch.py \
  --failures-from results_baseline_qwen3vl8b_cowbird.json \
  --cheap-model "qwen/qwen3-vl-8b-instruct" \
  --expert-model "claude-tutor" \
  --validator-model "claude-sonnet-4-6" \
  --pair bronzed_cowbird_vs_shiny_cowbird \
  --patch-rules tutor_session_rules.json \
  --output tutor_session.json
```

When `--expert-model claude-tutor` is set, rule authoring requests are written to
`tutor_inbox/<id>.json` and the script blocks until the corresponding
`tutor_outbox/<id>.txt` appears. The human (or another Claude Code session) reads
the inbox file — which includes the image — and writes the response.

**Inbox file format** (`tutor_inbox/<id>.json`):
```json
{
  "agent_id": "expert_rule_author",
  "system_prompt": "...",
  "text_content": "...",
  "image_paths": ["tutor_inbox/<id>_img0.jpg"]
}
```

**Outbox response**: a plain text file containing the JSON the agent would normally
return (e.g. `{"rule": "...", "feature": "...", "favors": "...", ...}`).

Only rule authoring (Steps 1a, contrastive analysis, spectrum generation) is routed
to the tutor. All pool validation (RULE_VALIDATOR calls) still uses `--validator-model`
via API.

### 4.6 Key flags reference

| Flag | Default | Description |
|---|---|---|
| `--cheap-model` | required | Pupil model for zero-shot and rerun classification |
| `--expert-model` | required | Expert model for rule authoring, completion, spectrum |
| `--validator-model` | = expert-model | Validator for pool gate checks (can be cheaper) |
| `--failures-from` | none | Load failures from a saved baseline JSON instead of running zero-shot |
| `--pair` | all pairs | Limit to a single pair ID |
| `--max-per-class` | 3 | Zero-shot baseline images per class |
| `--max-val-per-class` | 8 | Held-out pool size per class (never shown to expert) |
| `--max-authoring-per-class` | 8 | Authoring pool per class (contrastive analysis) |
| `--max-confirm-per-class` | 16 | Confirmation pool per class (fresh seed; 0 disables) |
| `--min-precision` | 0.75 | Minimum precision for held-out gate; FP is also capped at 1 |
| `--data-dir` | `DEFAULT_DATA_DIR` | CUB-200-2011 root directory |
| `--patch-rules` | `patch_rules_birds.json` | Output file for registered rules |
| `--output` | `patch_session_birds.json` | Full session record output |
| `--skip-rerun` | false | Skip the per-failure rerun step |
| `--rerun-only` | false | Apply saved rules to saved failures; no patching |
| `--dry-run` | false | Run full loop but do not register rules |

---

## 5. Output Files

### `patch_rules_<name>.json`

Array of registered rule objects. Each entry:

```json
{
  "id": "r_001",
  "condition": "...",    // full precondition string for injection into prompt
  "action": "Classify as Bronzed Cowbird. Rule: ...",
  "favors": "Bronzed Cowbird",
  "pair_id": "bronzed_cowbird_vs_shiny_cowbird",
  "source": "expert:claude-sonnet-4-6",
  "triggered_by": "bronzed_cowbird_vs_shiny_cowbird",
  "preconditions": ["...", "...", ...],
  "rule_text": "When ..."
}
```

This file is also the input to `--patch-rules` on `--rerun-only` runs.

### `patch_session_<name>.json`

Full session record. Top-level fields:

```json
{
  "cheap_model": "qwen/qwen3-vl-8b-instruct",
  "expert_model": "claude-sonnet-4-6",
  "validator_model": "claude-sonnet-4-6",
  "patch_rules_file": "patch_rules_birds_test.json",
  "dry_run": false,
  "total_tasks": 6,
  "failures_before": 4,
  "rules_authored": 4,
  "rules_accepted": 2,
  "rules_registered": 2,
  "patch_records": [...]
}
```

Each `patch_records` entry contains: `task_id`, `wrong_prediction`, `correct_label`,
`candidate_rule` (full JSON from expert), `semantic_validation`, `validation`
(held-out gate results), `spectrum_history`, `accepted`, `registered`, `rule_id`.

### `results_baseline_<name>.json`

Zero-shot baseline output. Contains `summary` and `tasks` array. Each task:

```json
{
  "task_id": "...",
  "pair_id": "bronzed_cowbird_vs_shiny_cowbird",
  "image_path": "C:\\...\\Bronzed_Cowbird_0019_796242.jpg",
  "correct_label": "Bronzed Cowbird",
  "predicted_label": "Shiny Cowbird",
  "correct": false
}
```

`image_path` is an absolute path — if running on a different machine, re-generate
the baseline or update paths accordingly.

### `knowledge_base/<pair>.json`

Curated, version-controlled subset of rules from completed sessions. The
`patch_rules_*.json` files in `python/` are session artifacts; `knowledge_base/`
holds the canonical registered rules for a pair. Format mirrors the dermatology
knowledge base: `pair`, `rules` array, each rule with `id`, `rule`, `feature`,
`favors`, `confidence`, `preconditions`, `held_out_gate`, `source`, `authored_from`,
`verified_by`, `created_at`.

---

## 6. Design Issues and Known Limitations

### 6.1 `agents.py` import chain

`birds/python/agents.py` re-uses the call_agent infrastructure from
`dermatology/python/agents.py` rather than duplicating it. It loads the dermatology
module at import time via `importlib.util.spec_from_file_location("derm_agents", ...)`.

Similarly, `birds/python/patch.py` loads `birds/python/agents.py` via importlib
rather than a normal `import agents` statement — because `image-classification/python/`
is on `sys.path` and contains its own `agents.py`, which would shadow the bird version.

If you move files or add `agents.py` to another directory on the path, this may break.
The fix is always to use explicit absolute path loading via importlib, not to rely on
`sys.path` ordering.

### 6.2 Absolute paths in saved baseline

`results_baseline_qwen3vl8b_cowbird.json` contains absolute Windows paths
(`C:\\_backup\\ml\\data\\...`). Running `--rerun-only --failures-from` on a different
machine will fail when the script tries to open those paths. Re-run the zero-shot step
first to generate a fresh baseline, or patch the paths in the JSON.

### 6.3 Held-out pool is sampled from train split only

The pool validation (held-out gate) draws images from `split="train"`. Test-split
images are never used for validation — but they also can't be used to pre-screen FP
edge cases. The Shiny_Cowbird_0080 false positive in the test session is an example:
r_002 fires on it as a FP, but it was in the test split and therefore not in the
pool. The gate caught FP ≤ 1 on training images; it could not prevent all possible
FP cases.

### 6.4 Spectrum search reliability

The 4-level spectrum generates rule variants at different specificity levels. The most
general level (L1, 1 precondition) sometimes fires on 0 images even in the trigger
pool — it is too general to distinguish. The most specific level (L4) sometimes
over-tightens until the rule no longer fires on its own trigger image. This happened
for failure 2 (Bronzed_0061) and failure 4 (Shiny_0080) in the test session, causing
both to be rejected. A future improvement would be to try a wider range of precondition
counts or to use the semantic validator's reliability ratings to guide which conditions
to include at each level.

### 6.5 Rule counter resets each session

`_rule_counter` starts at 0 every run. Rule IDs (`r_001`, `r_002`, ...) will
collide across sessions if you load a previous `--patch-rules` file and run again.
The system does not merge IDs — it appends new rules to the file with fresh IDs
starting from 1. Fix: either use separate output file names per session, or manually
edit the counter before merging rule files.

### 6.6 `--max-confirm-per-class 0` disables confirmation pool

The confirmation pool (a second independent pool drawn with a different random seed)
is meant to give a final precision check after the spectrum selects a winner. Setting
`--max-confirm-per-class 0` skips it to reduce API cost in development runs. The
test session used `--max-confirm-per-class 0`. For a production run, use the default
(16) or at least 4.

### 6.7 `fires_on_trigger=True` is required for registration

A rule that does not fire on the trigger image — the specific image it was authored
for — is rejected even if it achieves perfect precision on the pool. This is intentional:
a rule that can't recognize the case it was designed for is not yet calibrated. The
completion step sometimes over-tightens rules past this threshold (as in failure 2
and 4 of the test session).

### 6.8 Semantic validation is advisory only

The semantic REVISE verdict does not block registration. The validator may flag a
precondition as "unreliable" or "context_dependent" and recommend revising the rule,
but the loop proceeds to pool validation regardless. This is intentional — semantic
validation catches obvious errors but real-image pool validation is the ground truth.
If you want to enforce semantic ACCEPT before proceeding, you'd need to add that gate
in `patch.py:run_patch_loop`.

### 6.9 Windows console encoding

The script forces UTF-8 stdout/stderr at startup to work around Windows cp1252.
All status output uses ASCII symbols ("OK"/"WRONG") rather than Unicode (✓/✗). If
you redirect output to a file on Windows, ensure the receiving file is opened with
UTF-8 encoding.

---

## 7. Available Confusable Pairs

Pair IDs for `--pair`:

| Pair ID | Class A | Class B | CUB class IDs |
|---|---|---|---|
| `bronzed_cowbird_vs_shiny_cowbird` | Bronzed Cowbird | Shiny Cowbird | 26 vs 27 |
| `american_crow_vs_fish_crow` | American Crow | Fish Crow | 29 vs 30 |
| `red-faced_cormorant_vs_pelagic_cormorant` | Red-faced Cormorant | Pelagic Cormorant | 24 vs 25 |
| `black-billed_cuckoo_vs_yellow-billed_cuckoo` | Black-billed Cuckoo | Yellow-billed Cuckoo | 31 vs 33 |
| `brewer_sparrow_vs_clay-colored_sparrow` | Brewer Sparrow | Clay-colored Sparrow | 115 vs 117 |
| `chipping_sparrow_vs_tree_sparrow` | Chipping Sparrow | Tree Sparrow | 116 vs 130 |
| `california_gull_vs_herring_gull` | California Gull | Herring Gull | 59 vs 62 |
| `herring_gull_vs_ring-billed_gull` | Herring Gull | Ring-billed Gull | 62 vs 64 |
| `common_raven_vs_white-necked_raven` | Common Raven | White-necked Raven | 107 vs 108 |
| `loggerhead_shrike_vs_great_grey_shrike` | Loggerhead Shrike | Great Grey Shrike | 111 vs 112 |
| `indigo_bunting_vs_blue_grosbeak` | Indigo Bunting | Blue Grosbeak | 14 vs 54 |
| `common_tern_vs_forsters_tern` | Common Tern | Forster's Tern | 144 vs 146 |
| `caspian_tern_vs_elegant_tern` | Caspian Tern | Elegant Tern | 143 vs 145 |
| `northern_waterthrush_vs_louisiana_waterthrush` | Northern Waterthrush | Louisiana Waterthrush | 183 vs 184 |
| `least_flycatcher_vs_western_wood_pewee` | Least Flycatcher | Western Wood-Pewee | 39 vs 102 |

Only `bronzed_cowbird_vs_shiny_cowbird` has been run through the full patching loop.
The others have class definitions in `confusable_pairs.py` and can be used with any
of the `patch.py` commands above.

---

## 8. Architecture Notes

### Call chain

```
patch.py
  └── bird_agents (birds/python/agents.py, loaded via importlib)
        └── derm_agents (dermatology/python/agents.py, loaded via importlib)
              └── call_agent()  ──► Anthropic API  (Claude models)
                                ──► OpenRouter API  (Qwen, etc.)
                                ──► claude-tutor    (filesystem inbox/outbox)
```

`patch.py` also imports directly from the shared `image-classification/` layer:

```python
from dataset import load as load_cub, DEFAULT_DATA_DIR, CUBDataset
from confusable_pairs import CONFUSABLE_PAIRS, ConfusablePair
```

These resolve from `image-classification/python/` via `sys.path` insertion.

### sys.path insertion order

`patch.py` inserts four directories into `sys.path` at startup, in reverse priority
order (last inserted = highest priority):

```python
for _p in (str(_KF_ROOT), str(_UC_DIR / "src"), str(_UC_DIR / "python"), str(_HERE)):
    sys.path.insert(0, _p)
```

`_HERE` (`birds/python/`) ends up at position 0 — highest priority. This is why
`from dataset import ...` resolves from `image-classification/python/dataset.py`
only as long as `birds/python/` doesn't contain its own `dataset.py`. If you add
a `dataset.py` in `birds/python/`, it will shadow the shared one.

### Rule injection format (patched classifier)

When the cheap model is re-run with rules active, each registered rule is injected
into the system prompt as a numbered mandatory instruction block:

```
MANDATORY RULES — if a rule's pre-conditions are met, you MUST apply it:

RULE 1:
  IF: [all preconditions]
  THEN: [action / classification]

...
```

The model is instructed to output `"rule_fired": "<rule_id or null>"` so the script
can track which rule caused the prediction change.
