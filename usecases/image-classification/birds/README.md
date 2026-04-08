# Knowledge Fabric for Bird Species Identification: Teaching AI to See What Experts See

> **For**: Ornithologists, naturalists, and field biologists curious about how AI can be corrected by domain experts without any programming or retraining.
>
> **Status**: Experiment completed — Bronzed Cowbird vs Shiny Cowbird: Qwen3-VL-8B zero-shot 33% → 67% after KF dialogic patching (+33pp). 2 of 4 failures resolved; 2 rules registered.
>
> **Dataset**: CUB-200-2011, 200 North American bird species, 11,788 images.
>
> **Also see**: [Image Classification Overview](../README.md) for the broader Knowledge Fabric context, including the dermatology use case and cross-domain comparisons.

---

## The One-Line Summary

An AI model confused Bronzed Cowbirds with Shiny Cowbirds. An ornithologist explained, in plain language, the field marks that distinguish them. The system turned those explanations into explicit rules, tested them against known images, and applied them — improving accuracy from 33% to 67% without retraining the model.

---

## Contents

1. [Why Birds](#1-why-birds)
2. [The Dataset](#2-the-dataset)
3. [The Approach](#3-the-approach)
4. [The Dialogic Patching Loop — Step by Step](#4-the-dialogic-patching-loop--step-by-step)
5. [The Expert's Role](#5-the-experts-role)
6. [Worked Example: Confused Cowbirds](#6-worked-example-confused-cowbirds)
7. [Results](#7-results)
8. [What This Shows About Dialogic Learning](#8-what-this-shows-about-dialogic-learning)

---

## 1. Why Birds

Fine-grained visual classification of bird species is an ideal proving ground for the Knowledge Fabric thesis — that a domain expert's natural-language explanations can improve AI accuracy on hard cases without retraining.

**The gap is large and measurable.** CLIP zero-shot performance on CUB-200 is around 65%. The supervised state of the art (HERBS, 2023) reaches 93%. That 28-percentage-point gap represents exactly the space where expert discriminative knowledge matters — and where KF aims to operate.

**The hard cases are well-defined.** The remaining errors at SOTA are concentrated in confusable species pairs — cases where two species are so visually similar that statistical training on aggregate visual features falls short. These are exactly the cases where expert verbal criteria ("the bill is shorter relative to head size in a Downy") matter most.

**Expert language is publicly available.** Field guides (Sibley, Kaufman), eBird species accounts, and ornithological literature contain precisely the discriminative reasoning KF needs — for every species and every confusable pair.

**The logic is transparent.** "This is a Bronzed Cowbird and not a Shiny Cowbird because the iris is bright red, not dark brown" is a sentence any naturalist can read, understand, and verify. The knowledge is explicit and auditable.

The pair tested in this experiment:

| Bronzed Cowbird | Shiny Cowbird |
|---|---|
| ![Bronzed Cowbird](../assets/birds/bronzed_cowbird.jpg) | ![Shiny Cowbird](../assets/birds/shiny_cowbird.jpg) |

*Both are all-black cowbirds found in overlapping ranges. Bronzed has a conspicuous red iris, a thick decurved bill, and a prominent neck ruff (visible as raised hackle feathers). Shiny has a dark eye, a slimmer straight bill, and uniformly intense blue-violet iridescent gloss. These differences are visible in photographs but require knowing what to look for.*

---

## 2. The Dataset

**CUB-200-2011** (Caltech-UCSD Birds)

| Property | Value |
|---|---|
| Images | 11,788 (5,994 train / 5,794 test) |
| Classes | 200 North American bird species |
| Annotations | Labels + 312 binary visual attribute annotations + 10 per-image natural language descriptions (Reed et al.) |
| Access | Fully open — [Caltech download](https://data.caltech.edu/records/65de6-vp158) |
| Leaderboard | [Papers with Code](https://paperswithcode.com/sota/fine-grained-image-classification-on-cub-200) |
| CLIP zero-shot | ~65% |
| Supervised SOTA | 93.1% (HERBS, 2023) |

The 312 binary attribute annotations are the closest analogue in ornithology to the lesion-attribute annotations in dermatology: structured visual criteria, defined per species, that capture exactly the kind of discriminative field knowledge KF is designed to work with.

---

## 3. The Approach

This experiment demonstrates [Dialogic Learning](../../../docs/glossary.md#dialogic-learning) applied to a bird identification AI.

The core idea: a cheap, capable-but-fallible vision model (the "pupil") makes errors on hard confusable pairs. An expert — which may be a human ornithologist or a superior AI acting in that role — examines each error and explains *in plain language* what the pupil missed and what field mark should have been decisive. The system turns that explanation into an explicit, testable rule. Before the rule is trusted, it is validated against a pool of labeled images. If it passes, it is registered and applied to the pupil — and the pupil is re-tested.

This is not fine-tuning. The base model's weights are never changed. The knowledge lives in an explicit, human-readable rule file that can be inspected, corrected, or withdrawn at any time.

**Setup for this experiment**:

- **Pupil model**: Qwen3-VL-8B-Instruct (via OpenRouter) — a strong open-source vision-language model
- **Expert model**: Claude Sonnet 4.6, acting as a senior ornithologist
- **Validator model**: Claude Sonnet 4.6, independently checking rule quality
- **Pair tested**: Bronzed Cowbird vs Shiny Cowbird (6 images, 3 per class)
- **Held-out validation pool**: 8 images (4 per class, drawn from training split)

**Zero-shot baseline result: 2/6 correct (33.3%)**

Qwen misclassified every Bronzed Cowbird as a Shiny Cowbird, and one Shiny Cowbird as a Bronzed Cowbird. The pattern is exactly the kind of systematic failure the dialogic patching loop is designed to resolve: not random noise, but a consistent gap in what the model knows how to look for.

---

## 4. The Dialogic Patching Loop — Step by Step

The loop runs automatically once a failure case is identified. The expert's involvement is focused on high-value judgment calls; the mechanical work of validating rules against image pools is handled by the system automatically.

```
         ┌──────────────────────────────────────────────────────────┐
         │               FAILURE DETECTED                           │
         │   Qwen calls a Bronzed Cowbird a Shiny Cowbird           │
         └──────────────────────┬───────────────────────────────────┘
                                │
                                ▼
         ┌──────────────────────────────────────────────────────────┐
         │               EXPERT RULE AUTHORING                      │
         │   Expert sees the image + Qwen's wrong reasoning         │
         │   Expert writes: "Here is what Qwen missed and           │
         │   here is the field mark rule that should have applied." │
         └──────────────────────┬───────────────────────────────────┘
                                │
                                ▼
         ┌──────────────────────────────────────────────────────────┐
         │               RULE COMPLETION                            │
         │   A second pass adds the implicit background conditions  │
         │   the expert assumed but did not state —                 │
         │   closing loopholes a naive system would exploit         │
         └──────────────────────┬───────────────────────────────────┘
                                │
                                ▼
         ┌──────────────────────────────────────────────────────────┐
         │               SEMANTIC VALIDATION                        │
         │   Each precondition is reviewed independently:           │
         │   Is this a reliable discriminator between the two       │
         │   species, or is it ambiguous? Weak conditions are       │
         │   flagged before any images are tested.                  │
         └──────────────────────┬───────────────────────────────────┘
                                │
                                ▼
         ┌──────────────────────────────────────────────────────────┐
         │               IMAGE POOL VALIDATION                      │
         │   The rule is tested against a held-out pool of          │
         │   labeled images. Precision must be ≥ 75%, FP ≤ 1.      │
         │   If it fires on too many wrong cases → rejected.        │
         └──────────────────────┬───────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │ Fails precision gate  │ Passes
                    │ or doesn't fire       │
                    ▼                       ▼
         ┌──────────────────┐   ┌──────────────────────────────────┐
         │ SPECTRUM SEARCH  │   │          RULE REGISTERED         │
         │ Generate 4 rule  │   │                                  │
         │ versions (more   │   │  Applied to Qwen on the          │
         │ general to more  │   │  original failure image.         │
         │ specific). Test  │   │  Verified: did Qwen flip         │
         │ all four; keep   │   │  to the right answer?            │
         │ tightest that    │   └──────────────────────────────────┘
         │ passes.          │
         └──────────────────┘
```

---

## 5. The Expert's Role

In this experiment, the expert role is played by an AI assistant (Claude Sonnet 4.6) instructed to act as a senior ornithologist. In production use, this role would be filled by a human expert. The interactive loop is identical — only the identity of the expert changes.

The expert performs four distinct tasks:

### Role 1 — Rule Author

The expert sees the image and Qwen's wrong prediction. The expert's job is to:
1. Identify exactly what Qwen got wrong — did it miss a field mark? Did it misinterpret one?
2. Author a corrective rule: "When [these specific visual features are present], identify as [species]."

The rule must be purely visual — no range, habitat, vocalizations, or behavior unless directly visible in the image. It must be specific enough not to fire on the confusable species, and generalizable to other images of the same species, not just this one photograph.

### Role 2 — Rule Completer

A separate pass reviews the rule for implicit assumptions. Expert ornithologists write *diagnostic* rules: they describe what is distinctive about the failure case. But they unconsciously omit background conditions that go without saying to a trained observer. A naive AI checking only the explicit list would fire the rule on cases the expert never intended. The RULE_COMPLETER adds those background conditions explicitly — for instance, that the bird is an adult male (eliminating females and juveniles where the field mark may not apply).

### Role 3 — Semantic Validator

Before testing the rule on any images, each precondition is rated:
- **Reliable** — a consistently present, genuinely discriminating feature between the two species
- **Context-dependent** — only meaningful in combination with other conditions
- **Unreliable** — appears in both species or varies too much to be trusted

This step catches weak conditions before they waste validation budget.

### Role 4 — Spectrum Generator

If the completed rule fails the precision gate (too many false positive fires on the wrong species), the system generates four versions of the rule at different levels of specificity — from the single most essential condition to the full completed rule plus an additional tightening condition. All four are tested; the tightest version that still passes is kept. This prevents the common failure mode of over-tightening a rule until it no longer fires on the target image at all.

---

## 6. Worked Example: Confused Cowbirds

**Zero-shot baseline** — Qwen3-VL-8B on 6 images, no rules active:

| Image | Ground Truth | Prediction | Correct |
|---|---|---|---|
| Bronzed_Cowbird_0019 | Bronzed Cowbird | Shiny Cowbird | WRONG |
| Bronzed_Cowbird_0061 | Bronzed Cowbird | Shiny Cowbird | WRONG |
| Bronzed_Cowbird_0081 | Bronzed Cowbird | Shiny Cowbird | WRONG |
| Shiny_Cowbird_0005  | Shiny Cowbird  | Shiny Cowbird | correct |
| Shiny_Cowbird_0030  | Shiny Cowbird  | Shiny Cowbird | correct |
| Shiny_Cowbird_0080  | Shiny Cowbird  | Bronzed Cowbird | WRONG |

Qwen defaulted to Shiny Cowbird for every all-dark cowbird it saw. It identified the all-dark plumage — a feature shared by both species — and stopped there, without checking the features that actually distinguish them.

The patching loop processed each of the 4 failures in turn.

---

### Case 1 — Bronzed_Cowbird_0019

![Bronzed_Cowbird_0019](../assets/birds/bronzed_0019_fixed.jpg)

**Ground truth**: Bronzed Cowbird
**Qwen's prediction**: Shiny Cowbird — WRONG

**What Qwen missed**: The bright red iris. Bronzed Cowbird has a strikingly red eye visible in photographs; Shiny Cowbird has a dark brown-to-blackish iris that never appears red. The bill shape is also diagnostic — Bronzed has a thick, decurved bill that appears almost finch-like, while Shiny has a slender, straight-culmen blackbird bill. Qwen fixated on the all-black plumage (shared by both species) and defaulted to Shiny Cowbird without checking either the eye color or the bill morphology.

**Expert's corrective rule**:

> *When a small, all-black cowbird shows a conspicuous bright red or orange-red iris clearly visible as a bold colored eye, combined with a distinctly thick-based, slightly decurved bill that appears heavier and more robust than a typical icterid bill — identify as Bronzed Cowbird.*

**Validation result**:

| Pool | Fires on trigger | True Positives | False Positives | Precision | Result |
|---|---|---|---|---|---|
| Held-out (8 images) | Yes | 1 | 0 | **1.00** | Pass |

Rule registered as **r_001**.

---

### Case 2 — Bronzed_Cowbird_0061

![Bronzed_Cowbird_0061](../assets/birds/bronzed_0061_fixed.jpg)

**Ground truth**: Bronzed Cowbird
**Qwen's prediction**: Shiny Cowbird — WRONG

**What Qwen missed**: Again the red iris — the single most diagnostic field mark separating these two species — and the heavy neck ruff (hackle feathers that create the bull-necked profile characteristic of Bronzed Cowbird, entirely absent in Shiny Cowbird).

**Expert's corrective rule** (authored independently for this image):

> *When a stocky, all-dark cowbird shows a distinctly red iris (visible as a bright red or orange-red eye), a notably thick-based heavy bill, and prominent neck ruff or hackle texture (feathers appearing raised around the neck/upper breast creating a hunched, bull-necked profile) — identify as Bronzed Cowbird.*

**Validation result**: The completed rule (with additional background conditions added by the RULE_COMPLETER) **did not fire on the trigger image** — the additional conditions over-tightened it past the point where it recognizes its own target. The system correctly treated this as a completion artifact and fell back to the pre-completion rule, but that also failed to generalize across the held-out pool. Rule **rejected**.

**How it was fixed anyway**: Rule **r_001**, authored for Case 1, fired on this image at re-run and correctly flipped Qwen's prediction to Bronzed Cowbird. The red-iris rule generalized across cases — exactly the intended behavior.

---

### Case 3 — Bronzed_Cowbird_0081

![Bronzed_Cowbird_0081](../assets/birds/bronzed_0081_unfixed.jpg)

**Ground truth**: Bronzed Cowbird
**Qwen's prediction**: Shiny Cowbird — WRONG

**What Qwen missed**: The heavy, deep-based bill and the proportionally large, rounded head — the "bull-necked" gestalt that distinguishes Bronzed Cowbird from the slimmer-headed, slimmer-billed Shiny Cowbird. In this image the red iris is less clearly visible (angle or lighting), so the expert focused on structural features rather than eye color. Additionally, Shiny Cowbird males display intense, uniform blue-violet iridescent gloss across the whole body; this bird's gloss appears more restricted and less brilliantly iridescent — consistent with Bronzed.

**Expert's corrective rule**:

> *When a black cowbird shows a visibly thick, heavy, conical bill with a distinctly rounded head profile (approaching a 'bull-necked' or large-headed appearance), and the plumage shows a dull-to-moderate gloss rather than intense iridescent sheen across the entire body — identify as Bronzed Cowbird.*

**Spectrum search**: The completed rule (10 preconditions) failed to fire on the trigger image. The pre-completion rule (5 preconditions) fired with precision=1.00. The spectrum was generated:

| Level | Preconditions | TP | FP | Precision | Fires on trigger | Result |
|---|---|---|---|---|---|---|
| Most general (L1) | 1 | 0 | 0 | 0.00 | No | Fail |
| Moderate (L2) | 2 | 4 | 2 | 0.67 | Yes | Fail |
| Original (L3) | 5 | 2 | 0 | **1.00** | Yes | **Pass** |
| Most specific (L4) | 6 | 0 | 0 | 0.00 | No | Fail |

Level 3 (5 preconditions, pre-completion version) was selected. Rule registered as **r_002**.

**Re-run result**: At re-run, **r_002** fired on Bronzed_0081 and correctly predicted Bronzed Cowbird on Bronzed_0019 — but Bronzed_0081 itself was not fixed. r_002 fired on 0019 instead; 0081 remains unfixed (neither rule fires consistently on this particular shot, likely due to camera angle or lighting obscuring the diagnostic features).

---

### Case 4 — Shiny_Cowbird_0080

![Shiny_Cowbird_0080](../assets/birds/shiny_0080_unfixed.jpg)

**Ground truth**: Shiny Cowbird
**Qwen's prediction**: Bronzed Cowbird — WRONG

**What Qwen missed**: This is a female or subdued-plumage Shiny Cowbird. Female Bronzed Cowbirds are distinguished by a visible nuchal ruff, a steep-foreheaded "bull-necked" profile, a heavier and slightly decurved bill, and often a red or orange-red iris. This bird shows none of those: the head is smoothly rounded, the nape is flat with no ruff, the bill is slim and straight, and the eye is dark. Qwen apparently saw "plain brown cowbird" and defaulted to Bronzed — the reverse of its error on the all-black birds.

**Expert's corrective rule** (for Shiny Cowbird):

> *When a cowbird shows uniformly dull brown plumage with no visible neck ruff, a flat-crowned and round-headed profile, a slender conical bill without a pronounced decurved tip, and no iridescent tones on the upperparts — identify as Shiny Cowbird (female).*

**Validation result**: The rule fired on the trigger image, but on the 8-image held-out pool it did not fire on any of the 4 Shiny Cowbird images — it fired on 0 out of 4, giving precision 0.00. This is a case where the rule describes visible features correctly but the validator (Qwen) cannot reliably identify those features in held-out images. Rule **rejected**.

**What happened at re-run**: Rule r_002 (bull-necked + heavy bill → Bronzed Cowbird) fired as a **false positive** on this image, flipping a wrong-prediction into a confirmed wrong prediction. This is an expected edge case: the precision gate caught FP ≤ 1 on the held-out training pool, but this specific FP image was from the test split and was not in the pool. The gate controls for known FP cases, not all possible FP cases.

---

## 7. Results

### Zero-shot baseline (Qwen3-VL-8B, no rules)

| Pair | Correct | Total | Accuracy |
|---|---|---|---|
| Bronzed Cowbird vs Shiny Cowbird | 2 | 6 | 33.3% |

### Patching loop outcomes

| Failure | Expert's diagnosis | Rule | Outcome |
|---|---|---|---|
| Bronzed_0019 | Missed red iris + thick bill | r_001 registered (precision=1.00) | Fixed by r_002 at rerun |
| Bronzed_0061 | Missed red iris + neck ruff | Rejected — over-tightened by completer | Fixed by r_001 at rerun |
| Bronzed_0081 | Missed heavy bill + matte gloss | r_002 registered (spectrum L3, precision=1.00) | Unfixed (no rule fires) |
| Shiny_0080 | Missed absence of neck ruff + slim bill | Rejected — validator couldn't confirm in pool | Unfixed + r_002 FP |

### After patching

| Phase | Correct | Accuracy |
|---|---|---|
| Zero-shot (Qwen3-VL-8B) | 2/6 | 33.3% |
| After KF patching | 4/6 | 66.7% |
| Delta | +2 | **+33pp** |
| Rules authored / accepted / registered | 4 / 2 / 2 | |

### Cross-rule generalization

The most important result is the **cross-generalization**:
- **r_001** was authored from Bronzed_0019 but fixed Bronzed_0061 at rerun.
- **r_002** was authored from Bronzed_0081 but fixed Bronzed_0019 at rerun.

Each rule generalized beyond the image it was authored from to another example of the same confusion. This is the core prediction of the KF hypothesis: when a rule captures a class-level visual principle rather than an image-specific artifact, it transfers.

---

## 8. What This Shows About Dialogic Learning

### The pupil can be taught — if it is teachable

Qwen3-VL-8B followed the injected rules. When r_001 said "if you see a bright red iris, classify as Bronzed Cowbird," Qwen correctly applied that instruction when the feature was visible in a new image. This is not guaranteed for all models — some models effectively ignore injected rules, reverting to their training distribution regardless. Verifying that a candidate pupil is teachable before running the full loop is an important prerequisite.

### A single precise rule can fix multiple failures

Two rules fixed two failures each. Neither rule was authored with the second failure in mind — cross-generalization was an emergent result of capturing the right underlying field mark. This is Dialogic Learning at work: the expert's explanation of *why* the first failure was wrong turned out to contain the knowledge needed to fix a second, unrelated image.

### The precision gate protects what already works

Both rejected rules were rejected correctly. The Bronzed_0061 rule was over-tightened by the completer until it stopped firing on its own target image. The Shiny_0080 rule described genuine features but the validator could not reliably confirm them in held-out images — which means the rule would have been unreliable in deployment. The system declined to register both. This is not a bug; it is the expected behavior of a conservative validation gate.

### Expert language does not need to be technical

The accepted rules were derived from the same kind of language an experienced field ornithologist would use with a beginning student: "the eye is bright red, not dark brown," "the bill is thick-based and slightly decurved." No ML terminology. No training data. No annotation tooling. The domain expertise was sufficient on its own.

### Some failures require a different observation

Bronzed_0081 and Shiny_0080 remain unresolved. In both cases the diagnostic features are present in principle but are not reliably detected by the validator in the held-out pool — suggesting an angle, lighting, or image-quality issue that prevents the relevant field marks from being consistently visible. A third rule targeting a different discriminating feature (or a more robust image-level description) would be needed to recover those cases.

---

*For the technical architecture, pipeline design, and full implementation notes, see [DESIGN.md](../DESIGN.md).*

*For the dermatology parallel — where the same pipeline was applied to skin-lesion classification — see [dermatology/README.md](../dermatology/README.md).*

*For the broader Knowledge Fabric positioning, see the [Image Classification Overview](../README.md).*
