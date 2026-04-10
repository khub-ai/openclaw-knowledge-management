# Dermatology Use Case — Developer Design Spec

This document is the developer-facing reference for the dermatology dialogic patching
experiment. For the user-facing walkthrough, see [README.md](README.md). For the
cross-domain design spec covering both dermatology and birds, see
[../DESIGN.md](../DESIGN.md).

---

## Contents

1. [Repository Layout](#1-repository-layout)
2. [Data](#2-data)
3. [API Keys](#3-api-keys)
4. [How to Run — Dialogic Patch Loop](#4-how-to-run--dialogic-patch-loop)
5. [How to Run — KF Ensemble Harness](#5-how-to-run--kf-ensemble-harness)
6. [Output Files](#6-output-files)
7. [Design Issues and Known Limitations](#7-design-issues-and-known-limitations)
8. [Available Pairs](#8-available-pairs)
9. [Architecture Notes](#9-architecture-notes)

---

## 1. Repository Layout

```
usecases/image-classification/dermatology/
  README.md                               # user-facing walkthrough
  DESIGN.md                               # this file
  assets/
    ISIC_0024315.jpg                      # melanoma failure images (Cases 1–3)
    ISIC_0024333.jpg
    ISIC_0024400.jpg
    ISIC_0024336.jpg                      # BCC/BKL failure images (Cases 4–5)
    ISIC_0024420.jpg
  knowledge_base/
    melanoma_vs_melanocytic_nevus.json    # batch-imported rules (pre-loop)
    basal_cell_carcinoma_vs_benign_keratosis.json
    actinic_keratosis_vs_benign_keratosis.json
  python/
    patch.py                              # dialogic patching loop (main entry point)
    agents.py                             # all LLM agent calls + OpenRouter/Claude/tutor routing
    harness.py                            # KF ensemble pipeline (Observer/Mediator/Verifier)
    dataset.py                            # HAM10000 dataset loader
    rules.py                              # RuleEngine (used by harness)
    tools.py                              # tool definitions for the ensemble agents
    migrate_rules.py                      # batch-import knowledge_base/*.json → rules.json
    rules.json                            # live rule store used by the KF ensemble harness
    patch_rules_clean.json                # patch loop rules (isolated from rules.json)
    results_baseline_qwen3vl8b_zeroshot.json    # Qwen3-VL-8B zero-shot baseline
    results_baseline_gpt4o_zeroshot.json        # GPT-4o zero-shot baseline
    results_baseline_maverick_zeroshot.json     # Llama-4 Maverick zero-shot baseline
    distill_dialogic.py                   # three-party dialogic distillation (PUPIL/TUTOR/KF)
    distill_dialogic_session.json         # transcript of dialogic distillation run
    patch_session_single_test.json        # canonical completed session (mel/nev + tutor)
    patch_session_*.json                  # other session records
    tutor_inbox/                          # human-in-the-loop request files
    tutor_outbox/                         # human-in-the-loop response files
```

The `agents.py` file is the shared infrastructure layer for the entire
`image-classification/` use case. Both the birds `agents.py` and the ensemble
`harness.py` depend on it. It handles:
- Anthropic API calls (Claude models)
- OpenRouter API calls (Qwen3-VL-8B, GPT-4o-mini, etc.)
- `claude-tutor` filesystem inbox/outbox routing
- Call caching and cost tracking

---

## 2. Data

### HAM10000 dataset

| Property | Value |
|---|---|
| Default location | `C:\_backup\ml\data\DermaMNIST_HAM10000` |
| Download | [ISIC archive](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) or [Kaggle](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection) |
| Size | ~2.7 GB, 10,015 images |
| Metadata | `HAM10000_metadata.csv` |

Override at runtime with `--data-dir /your/path`.

**Expected directory structure:**

```
DermaMNIST_HAM10000/
  HAM10000_images_part_1/
    ISIC_0024306.jpg
    ISIC_0024307.jpg
    ...
  HAM10000_images_part_2/
    ISIC_0029306.jpg
    ...
  HAM10000_metadata.csv
```

The loader reads `HAM10000_metadata.csv` to get per-image diagnosis labels (`dx`)
and reserves a deterministic 20% of each class as the test split (by `lesion_id`).
No separate split file is required.

### Class sizes (approximate, after train/test split)

| Class | Dx code | Total | Approx train | Approx test |
|---|---|---|---|---|
| Melanoma | `mel` | 1,113 | ~890 | ~223 |
| Melanocytic Nevus | `nv` | 6,705 | ~5,364 | ~1,341 |
| Basal Cell Carcinoma | `bcc` | 514 | ~411 | ~103 |
| Benign Keratosis | `bkl` | 1,099 | ~879 | ~220 |
| Actinic Keratosis | `akiec` | 327 | ~262 | ~65 |

> Note: the loaded sizes above (mel=123, nv=1081, etc.) are after the dataset filters
> to images that can be found on disk. If both parts are present and complete, totals
> are closer to the full dataset counts.

### Failure images referenced in experiments

The five images used in the worked examples are in `assets/` and also in
`HAM10000_images_part_1/`. The `assets/` copies are only for documentation; the
loader uses the `HAM10000_images_part_1/part_2/` paths.

---

## 3. API Keys

The script reads keys from `P:/_access/Security/api_keys.env` (developer-specific
private file). If that file doesn't exist, set environment variables directly:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENROUTER_API_KEY=sk-or-...
export OPENAI_API_KEY=sk-...          # only needed for OpenAI models
```

| Key | Used for |
|---|---|
| `ANTHROPIC_API_KEY` | Claude models (expert, validator, ensemble harness) |
| `OPENROUTER_API_KEY` | Cheap model calls (Qwen3-VL-8B and other OpenRouter models) |
| `OPENAI_API_KEY` | GPT-4o, o4-mini (optional) |

---

## 4. How to Run — Dialogic Patch Loop

All `patch.py` commands run from `dermatology/python/`.

### 4.1 Canonical experiment (Qwen3-VL-8B + claude-tutor expert)

This reproduces the completed mel/nev session from the README worked example:

```bash
python patch.py \
  --failures-from results_baseline_qwen3vl8b_zeroshot.json \
  --cheap-model "qwen/qwen3-vl-8b-instruct" \
  --expert-model "claude-tutor" \
  --validator-model "claude-sonnet-4-6" \
  --pair melanoma_vs_melanocytic_nevus \
  --max-val-per-class 4 \
  --max-authoring-per-class 4 \
  --max-confirm-per-class 8 \
  --patch-rules patch_rules_clean.json \
  --output patch_session_my_run.json
```

When `--expert-model claude-tutor` is used, the script pauses after authoring each
rule request and waits for you to write a response to `tutor_outbox/<id>.txt`. See
§4.6 for the human-in-the-loop protocol.

### 4.2 Fully automated (API expert + API validator)

```bash
python patch.py \
  --failures-from results_baseline_qwen3vl8b_zeroshot.json \
  --cheap-model "qwen/qwen3-vl-8b-instruct" \
  --expert-model "claude-sonnet-4-6" \
  --validator-model "claude-sonnet-4-6" \
  --pair melanoma_vs_melanocytic_nevus \
  --max-per-class 3 \
  --max-val-per-class 4 \
  --max-authoring-per-class 4 \
  --max-confirm-per-class 0 \
  --patch-rules patch_rules_my_run.json \
  --output patch_session_my_run.json
```

This requires no human interaction. `--max-confirm-per-class 0` disables the final
confirmation pool to reduce cost during development.

### 4.3 Fresh baseline + patch (no saved baseline)

```bash
python patch.py \
  --cheap-model "qwen/qwen3-vl-8b-instruct" \
  --expert-model "claude-sonnet-4-6" \
  --validator-model "claude-sonnet-4-6" \
  --pair melanoma_vs_melanocytic_nevus \
  --max-per-class 3 \
  --patch-rules patch_rules_my_run.json \
  --output patch_session_my_run.json
```

Omitting `--failures-from` runs the zero-shot baseline first.

### 4.4 Dry run (author and validate, do not register)

```bash
python patch.py \
  --failures-from results_baseline_qwen3vl8b_zeroshot.json \
  --cheap-model "qwen/qwen3-vl-8b-instruct" \
  --expert-model "claude-sonnet-4-6" \
  --validator-model "claude-sonnet-4-6" \
  --pair melanoma_vs_melanocytic_nevus \
  --dry-run \
  --output patch_session_dry_run.json
```

### 4.5 Run a different pair

Replace `--pair melanoma_vs_melanocytic_nevus` with any of:
- `--pair basal_cell_carcinoma_vs_benign_keratosis`
- `--pair actinic_keratosis_vs_benign_keratosis`

Or omit `--pair` entirely to run all three pairs in sequence.

### 4.6 Human-in-the-loop (claude-tutor) protocol

When `--expert-model claude-tutor`, rule authoring is routed to the filesystem:

1. The script writes `tutor_inbox/<id>.json` containing the system prompt, image
   path, and the agent's question.
2. The script prints the inbox path and blocks, polling every 2 seconds.
3. You read the inbox file (and the referenced image), write your response as plain
   text to `tutor_outbox/<id>.txt`, and save.
4. The script detects the outbox file, reads it, and continues.

**Inbox file shape:**
```json
{
  "agent_id": "expert_rule_author",
  "system_prompt": "You are a senior dermoscopy clinician...",
  "text_content": "The model predicted Melanocytic Nevus. The correct answer is Melanoma...",
  "image_paths": ["tutor_inbox/<id>_img0.jpg"]
}
```

**Outbox response:** a plain text file containing the JSON the agent would return,
for example:
```json
{
  "rule": "When a dermoscopic lesion shows regression structures...",
  "feature": "regression_structures",
  "favors": "Melanoma",
  "confidence": "high",
  "preconditions": ["..."],
  "rationale": "..."
}
```

Only rule authoring steps (EXPERT_RULE_AUTHOR, contrastive analysis, spectrum
generation) are routed to the tutor. All pool validation (RULE_VALIDATOR) uses
`--validator-model` via API — those are high-volume binary checks, not creative
judgment calls.

### 4.7 Key flags reference

| Flag | Default | Description |
|---|---|---|
| `--cheap-model` | required | Pupil model for zero-shot baseline and rerun |
| `--expert-model` | required | Expert model for rule authoring steps |
| `--validator-model` | = expert-model | Pool validation calls (can be a cheaper model) |
| `--failures-from` | none | Load failures from a saved baseline JSON |
| `--pair` | all pairs | Limit to one pair |
| `--max-per-class` | 3 | Zero-shot baseline images per class |
| `--max-val-per-class` | 8 | Held-out pool size per class |
| `--max-authoring-per-class` | 8 | Authoring pool per class (contrastive analysis) |
| `--max-confirm-per-class` | 16 | Confirmation pool per class (0 disables) |
| `--min-precision` | 0.75 | Minimum precision for pool gate; FP also capped at 1 |
| `--data-dir` | `DEFAULT_DATA_DIR` | HAM10000 root directory |
| `--patch-rules` | `patch_rules.json` | Output file for registered rules |
| `--output` | `patch_session.json` | Full session record |
| `--skip-rerun` | false | Skip the per-failure rerun step |
| `--dry-run` | false | Author and validate rules but do not register |

---

## 5. How to Run — KF Ensemble Harness

The ensemble harness (`harness.py`) is a separate pipeline from the patch loop.
It runs the full Observer/Mediator/Verifier KF ensemble on images from the dataset.
It requires a populated `rules.json` (loaded from `knowledge_base/` via `migrate_rules.py`).

### 5.1 Migrate knowledge base rules into rules.json

```bash
python migrate_rules.py              # import all knowledge_base/*.json → rules.json
python migrate_rules.py --dry-run    # preview without saving
```

Run this once before using the harness. It is safe to re-run — it won't duplicate rules.

### 5.2 Run the ensemble on a single pair

```bash
python harness.py \
  --pair melanoma_vs_melanocytic_nevus \
  --max-per-class 3 \
  --mode test \
  --output results_mel_nev.json
```

`--mode test` uses frozen rules (no rule learning). `--mode train` enables rule
extraction from outcomes (online learning — introduces confound, use carefully).

### 5.3 Zero-shot baseline (no KF rules)

```bash
python harness.py \
  --pair melanoma_vs_melanocytic_nevus \
  --max-per-class 3 \
  --baseline zero_shot \
  --output results_baseline.json
```

### 5.4 Run all three pairs

```bash
python harness.py \
  --all \
  --max-per-class 3 \
  --mode test \
  --output results_all.json
```

### 5.5 Resume an interrupted run

```bash
python harness.py \
  --pair melanoma_vs_melanocytic_nevus \
  --max-per-class 3 \
  --mode test \
  --output results_mel_nev.json \
  --resume
```

`--resume` skips any task_id already present in `--output`.

---

## 6. Output Files

### `patch_rules_<name>.json`

Rule store for a patch session. Unlike `rules.json` (which the ensemble harness
uses), patch rules are stored in a flat array format:

```json
{
  "version": 2,
  "rules": [
    {
      "id": "r_001",
      "condition": "[Patch rule — melanoma_vs_melanocytic_nevus] ...",
      "action": "Classify as Melanoma. Rule: ...",
      "favors": "Melanoma",
      "pair_id": "melanoma_vs_melanocytic_nevus",
      "source": "expert:claude-tutor",
      "triggered_by": "melanoma_vs_melanocytic_nevus_ISIC_0024315",
      "preconditions": ["...", "...", ...],
      "rule_text": "When ..."
    },
    ...
  ]
}
```

`patch_rules_clean.json` is the canonical registered-rule file for the completed
mel/nev tutor session (2 rules: r_001 and r_002). Never pre-populate it before a
new run — the script appends to it on each registration.

### `patch_session_<name>.json`

Full session record. Key fields:

```json
{
  "cheap_model": "qwen/qwen3-vl-8b-instruct",
  "expert_model": "claude-tutor",
  "validator_model": "claude-sonnet-4-6",
  "patch_rules_file": "patch_rules_clean.json",
  "total_tasks": 6,
  "failures_before": 3,
  "rules_authored": 3,
  "rules_accepted": 2,
  "rules_registered": 2,
  "cross_pair_events": [...],
  "patch_records": [...]
}
```

`cross_pair_events` lists cases where a rule registered for one failure fired on a
different failure. In the canonical session, r_002 (authored for ISIC_0024333)
fired on ISIC_0024400 and fixed it.

`patch_session_single_test.json` is the canonical completed session (mel/nev, all
three failures resolved via 2 rules + 1 cross-pair generalization).

### `results_baseline_<name>_zeroshot.json`

Zero-shot baseline from the harness. Contains a `tasks` array with per-image results.
Three baseline files are saved:
- `results_baseline_qwen3vl8b_zeroshot.json` — Qwen3-VL-8B (11/18 correct, 61%)
- `results_baseline_gpt4o_zeroshot.json` — GPT-4o (used in early experiments)
- `results_baseline_maverick_zeroshot.json` — Llama-4 Maverick

Use `--failures-from results_baseline_qwen3vl8b_zeroshot.json` to reproduce the
canonical session without re-running the cheap model.

### `rules.json`

The live rule store used by the ensemble harness (not the patch loop). Populated via
`migrate_rules.py` from `knowledge_base/`. The patch loop never writes to `rules.json`
— it uses `patch_rules_<name>.json` files.

---

## 7. Design Issues and Known Limitations

### 7.1 Two separate rule stores

`rules.json` (used by `harness.py` via `rules.py` / `RuleEngine`) and
`patch_rules_<name>.json` (used by `patch.py`) are separate formats and separate
files. They are not synchronized. If you want rules from a completed patch session
to be available to the ensemble harness, you must manually copy them into
`knowledge_base/<pair>.json` and re-run `migrate_rules.py`.

### 7.2 No rerun-only mode in dermatology patch.py

Unlike the birds `patch.py`, the dermatology version does not have `--rerun-only`.
To test whether registered rules fix the failures, re-run the loop with
`--failures-from` pointing to the baseline file and inspect the `patch_records`
in the output session JSON — or run the harness with the rules migrated.

### 7.3 AK/BKL pair is architecturally unresolvable

The actinic_keratosis_vs_benign_keratosis pair contains lichenoid keratosis (LPLK)
images in the BKL class that lack the standard seborrheic keratosis markers. These
images are visually similar to actinic keratosis at the SK-feature level, making it
impossible to write a discriminating rule without introducing FPs. Use `--sk-only` in
the harness to restrict the BKL class to histology-confirmed seborrheic keratosis
images only:

```bash
python harness.py --pair actinic_keratosis_vs_benign_keratosis --sk-only --mode test
```

This excludes the LPLK variants and gives a cleaner experimental result, at the cost
of reduced BKL class size.

### 7.4 train/test split is deterministic but not standard

The loader reserves 20% of images per `lesion_id` group as the test split using a
fixed seed. This is not the official HAM10000 train/test split (the dataset does not
ship with one). Results from this code are not directly comparable to published
HAM10000 benchmarks that define their own splits.

### 7.5 claude-tutor requires polling — no timeout

The script polls `tutor_outbox/` every 2 seconds and blocks indefinitely. There is
no timeout. If you close the session without writing a response file, the script
will hang. To interrupt: `Ctrl+C`, then clean up any partial inbox files before
the next run.

### 7.6 Online learning confound in harness --mode train

`--mode train` enables the post-task rule extraction pass — the harness adds new
rules after each classification. In a multi-image run, later images may benefit from
rules learned during earlier images in the same run. This means `--mode train` results
are not a clean measure of the pre-loaded rule set's effectiveness. Always use
`--mode test` for evaluation.

### 7.7 Expert cost is paid per failure, not per rule

Each failure processed by the patch loop makes multiple API calls:
- 1 EXPERT_RULE_AUTHOR call (for rule authoring)
- 1 RULE_COMPLETER call
- 1 SEMANTIC_RULE_VALIDATOR call
- N RULE_VALIDATOR calls against the held-out pool (one per pool image, per spectrum level)
- If contrastive analysis is triggered: 1 more EXPERT call + N more validator calls

A single failure can generate 20–40 API calls in total if the spectrum search is
triggered. With `claude-sonnet-4-6` as validator, cost per failure is approximately
$0.05–$0.20 depending on pool size and spectrum depth. Use `--max-val-per-class 4`
and `--max-confirm-per-class 0` for development runs.

### 7.8 Batch import rules are weaker than loop-authored rules

Rules in `knowledge_base/*.json` were pre-encoded from dermoscopy references, not
authored by the dialogic loop. They lack `held_out_gate` results and `authored_from`
fields. They are useful as a starting-point rule set for the ensemble harness, but
they have not been through the same validation pipeline as loop-authored rules.

### 7.9 Semantic validation is advisory

The semantic REVISE verdict does not block registration. A rule that gets REVISE
from the semantic validator still proceeds to pool validation. If you want to enforce
a stricter gate, add a check on `semantic_validation["verdict"] == "accept"` in
`patch.py:run_patch_loop` before the held-out gate.

### 7.10 Rule counter resets each session

`_rule_counter` starts at 0 on each run. If you point `--patch-rules` at an existing
file from a prior session, new rules will get IDs `r_001`, `r_002`, etc. — which
may collide with existing entries. Use a new output file name per session, or manually
offset the counter in the JSON before appending.

---

## 8. Available Pairs

| Pair ID | Class A | Dx code | Class B | Dx code |
|---|---|---|---|---|
| `melanoma_vs_melanocytic_nevus` | Melanoma | `mel` | Melanocytic Nevus | `nv` |
| `basal_cell_carcinoma_vs_benign_keratosis` | Basal Cell Carcinoma | `bcc` | Benign Keratosis | `bkl` |
| `actinic_keratosis_vs_benign_keratosis` | Actinic Keratosis | `akiec` | Benign Keratosis | `bkl` |

Only `melanoma_vs_melanocytic_nevus` has been run to completion through the full
dialogic patching loop. `basal_cell_carcinoma_vs_benign_keratosis` has been partially
run (2 failures; 1 rule registered, 1 rejected). `actinic_keratosis_vs_benign_keratosis`
is architecturally blocked (see §7.3).

---

## 9. Architecture Notes

### Call chain

```
patch.py (or harness.py)
  └── agents.py
        ├── call_agent() ──► Anthropic API   (Claude models)
        │                ──► OpenRouter API   (Qwen3-VL-8B, etc.)
        │                ──► OpenAI API       (GPT-4o, o4-mini)
        │                ──► claude-tutor     (filesystem inbox/outbox)
        └── cost tracker, call cache
```

### Agent roles used in patch.py

| Agent role | Function | Who calls it |
|---|---|---|
| `EXPERT_RULE_AUTHOR` | Author initial corrective rule from failure image | Expert model / claude-tutor |
| `RULE_COMPLETER` | Add implicit background conditions to rule | Expert model |
| `SEMANTIC_RULE_VALIDATOR` | Rate each precondition; advise ACCEPT/REVISE/REJECT | Validator model |
| `RULE_VALIDATOR` (per image) | Binary: do this image's visible features meet all preconditions? | Validator model |
| `CONTRASTIVE_ANALYSIS` | Identify feature that separates TP from FP pool images | Expert model |
| `RULE_SPECTRUM_GENERATOR` | Generate 4 versions of rule at different specificity levels | Expert model |

### Rule injection format (patched classifier in patch.py)

Rules are injected into the cheap model's system prompt as mandatory instructions:

```
MANDATORY RULES — if a rule's pre-conditions are met, you MUST apply it:

RULE 1:
  IF: <condition string (full precondition list)>
  THEN: <action string (classification + confidence + rule text)>

...
```

The model outputs `"rule_fired": "<rule_id or null>"` in its response JSON.

### Held-out pool mechanics

- Pool is sampled from `split="train"` images only (test images never used for validation).
- Default seed 42 for the held-out pool, seed 123 for the confirmation pool.
- The pool is always balanced: equal numbers from both classes.
- A rule passes if: `precision >= min_precision` AND `FP <= 1` AND `fires_on_trigger == True`.
- All three conditions must hold; precision alone is not sufficient.

---

## 10. Three-Party Dialogic Distillation (`distill_dialogic.py`)

A separate experiment demonstrating that multi-round dialog is necessary for
effective knowledge distillation. See [README.md §10](README.md#10-three-party-dialogic-learning--why-it-works)
for the full user-facing writeup.

### The grounding problem

Single-shot elicitation (ask the tutor for a rule, test it) fails because the
tutor and validator use different visual vocabulary for the same image. The tutor
writes "irregular pigment network with abrupt cutoff"; the validator sees
"relatively symmetric oval shape with corona pattern." Both descriptions are
valid — they describe the same image at different levels of abstraction. But the
rule's preconditions must match the *validator's* vocabulary to fire.

### Three-party solution

`distill_dialogic.py` implements a multi-round protocol:

```
PUPIL failure ──► TUTOR authors rule (Round 1)
                       │
                       ▼
                  KF: grounding check
                  (validator tests preconditions on trigger image)
                       │
               ┌───────┴───────┐
               │ fires         │ does not fire
               ▼               ▼
          pool gate       KF: show TUTOR
                          validator's observations
                          + steering guidance
                               │
                               ▼
                          TUTOR: refine rule (Round 2+)
                               │
                               └──► KF: grounding check (loop)
```

KF's steering moves:
1. **Vocabulary alignment**: "use the validator's phrases, not dermoscopic abstractions"
2. **Specificity coaching**: "you had N preconditions — consolidate to 2-3"
3. **Strategy pivot**: at round 3+, suggest trying a different visual signal

### Usage

```bash
python distill_dialogic.py
python distill_dialogic.py --max-rounds 4 --tutor-model claude-opus-4-6
python distill_dialogic.py --failure-ids ISIC_0024410,ISIC_0024647
```

### Evidence: single-shot vs dialogic (same 4 failures)

| Image | Single-shot | Dialogic | Rounds | Pool precision |
|---|---|---|---|---|
| ISIC_0024410 | not grounded | **ACCEPTED** | 3 | 1.00 |
| ISIC_0024647 | not grounded | **ACCEPTED** | 1 | 0.83 |
| ISIC_0024911 | not grounded | **ACCEPTED** | 1 | 1.00 |
| ISIC_0025128 | not grounded | grounded, pool failed | 1 | 0.67 |

Single-shot: 0/4 grounded. Dialogic: 4/4 grounded, 3/4 accepted.

**Expanded test** (60 images, 30/class): 3 accepted rules applied to Qwen3-VL-8B
achieved 55/60 (91.7%, +36.7pp over zero-shot baseline). Comparable to the
original failure-driven patch loop (56/60, 93.3%).

### Output

`distill_dialogic_session.json` contains the full transcript: every round's
tutor rule, KF grounding check, validator observations, KF steering guidance,
and pool results. This is the primary evidence artifact.
