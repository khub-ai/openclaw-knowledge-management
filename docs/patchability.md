# DD Patchability: When Does Dialogic Distillation Work?

> A theory of DD applicability — predicting before an experiment whether a
> new domain will yield a meaningful result from Dialogic Distillation.

---

## The Core Question

DD has been demonstrated empirically in birds and dermatology. Both showed
large accuracy gains (+50pp) from a small number of expert-authored rules.
But these were selected opportunistically. Before investing in a new domain
experiment — staging images, sourcing experts, running sessions — it would be
valuable to know in advance whether DD is likely to work there.

This document proposes a principled framework for that prediction.

---

## The Fundamental Condition

DD works when the PUPIL has **latent perceptual capability that expert
vocabulary can unlock**, but is not currently using that capability because
it does not know what features to attend to.

DD fails when the PUPIL **genuinely cannot perceive** the relevant features —
not a vocabulary problem, but a hard sensory or resolution barrier.

These two failure modes look identical from the outside (low classification
accuracy) but have completely different causes and completely different
responses to DD:

| Failure type | Root cause | DD outcome |
|---|---|---|
| **Vocabulary gap** | PUPIL can see the features but lacks the terms or significance framing | DD works — unlocks latent capability |
| **Perception barrier** | Features are physically unobservable at this modality or resolution | DD fails — no latent capability to unlock |

The grounding check already implicitly tests this: a rule that consistently
fails grounding signals that the PUPIL cannot observe what the expert
described. But this diagnosis happens *after* a DD session, not before.
The goal here is a pre-session diagnostic.

---

## The Four Patchability Dimensions

### Dimension 1: Grounding Probe Rate (G)

**What it measures**: What fraction of expert-described features does the KF
validator confirm as observable in the PUPIL's modality?

**How to measure**: Ask an expert to describe 5 representative images from
the domain in 2–3 sentences each. Run each description through the KF
grounding check. Count the fraction of individual feature claims confirmed
as observable.

| G value | Interpretation |
|---|---|
| G > 0.70 | Expert features are observable — DD likely to work |
| 0.40–0.70 | Partial observability — DD may work with rule refinement |
| G < 0.30 | Expert features largely unobservable — DD likely to fail |

**Example**: SAR imagery direct classification scores G ≈ 0.05 — radar
backscatter features are physically invisible to an optical VLM. Dermoscopy
scores G ≈ 0.85 — regression areas, globules, and pigment network are all
visible to a camera.

---

### Dimension 2: Vocabulary Divergence (V)

**What it measures**: How different is the PUPIL's spontaneous description
vocabulary from an expert's description vocabulary for the same image?

**How to measure**: Show the same 5 images to both Qwen3-VL-8B (free
description prompt: "describe what you see in this image") and a domain
expert. Compare the resulting descriptions — manually or via embedding
cosine distance.

| V level | Interpretation |
|---|---|
| High | Expert uses domain-specific terms PUPIL never generates — large gap, high DD potential |
| Medium | Some overlap, some expert-specific terms — moderate gap |
| Low | PUPIL already describes images in terms close to expert vocabulary — small marginal gain |

**Key insight**: High V is necessary but not sufficient. V must be high *and*
G must be high. A large vocabulary gap over unobservable features (SAR direct
classification) yields high V but G ≈ 0, so DD still fails.

---

### Dimension 3: Zero-Shot Baseline Accuracy (B)

**What it measures**: How well does the PUPIL perform on the confusable pair
without any DD intervention?

**How to measure**: Run Qwen3-VL-8B zero-shot on a held-out set of 20 images
per class from the confusable pair. Record accuracy.

| B range | Interpretation |
|---|---|
| B > 0.90 | Near-saturated — DD adds little; domain may not need it |
| 0.40–0.75 | Genuine confusion — meaningful failure rate with room for improvement — **sweet spot** |
| 0.30–0.40 | Poor performance — possible perception barrier rather than vocabulary gap |
| B < 0.30 | Below random or near-random — strong signal of perception barrier |

**Caution**: Very low B (< 0.30) can indicate either a very hard vocabulary
gap or a perception barrier. Distinguish using the G score — if G is also
low, it is a perception barrier; if G is high, it is a hard vocabulary gap
that DD can address.

---

### Dimension 4: Expert Convergence (E)

**What it measures**: Can a domain expert articulate a clear, observable
distinction between the confusable pair within a few minutes of looking at
images? Do multiple experts agree?

**How to assess**: Qualitative. Show 3 pairs of confusable images to one or
two domain experts and ask: "What would you look for to distinguish these?"
If the expert responds immediately with specific visual features, E is high.
If the expert hedges, says "it depends," or requires clinical context beyond
the image, E is low.

| E level | Interpretation |
|---|---|
| High | Expert articulates clear observable distinctions confidently — DD will produce stable rules |
| Medium | Expert can distinguish but with caveats or requires multiple features simultaneously |
| Low | Expert disagrees with themselves or requires non-visual context — unstable vocabulary to distil |

---

## The Combined Patchability Assessment

```
High patchability:   G > 0.70  AND  V high  AND  0.40 < B < 0.75  AND  E high

Low patchability (any of):
  - G < 0.30           (perception barrier — DD cannot help)
  - B > 0.90           (already solved — DD not needed)
  - V low              (no vocabulary gap — nothing to distil)
  - E low              (no stable expert vocabulary to distil)
```

This is not a numerical formula — it is a structured checklist. Each
dimension can eliminate a domain from consideration independently, or raise
confidence that DD will work.

---

## Domain Assessment: Confirmed Results

### Birds (CUB-200-2011)

| Dimension | Score | Evidence |
|---|---|---|
| G | High (~0.85) | Eye rings, bill curvature, wing bar position all clearly observable |
| V | High | Qwen says "bird"; ornithologist says "thick-based decurved bill with iridescent sheen on hood" |
| B | 33% zero-shot | Bronzed vs Shiny Cowbird — well below chance (two classes), genuine confusion |
| E | High | Field identification keys are explicit, stable, widely agreed |

**Result**: High patchability. Confirmed: 33% → 83% after 2-round DD (+50pp).

---

### Dermatology (HAM10000)

| Dimension | Score | Evidence |
|---|---|---|
| G | High (~0.85) | Regression, globules, pigment network, blue-grey veil all observable via camera |
| V | High | Qwen says "dark skin lesion"; dermoscopist says "regression with peppering and disorganised network" |
| B | 50–67% zero-shot | Melanoma vs nevus — meaningful failure rate |
| E | High | Dermoscopy has a formal structured vocabulary (ABCD rule, Menzies method) |

**Result**: High patchability. Confirmed: 50% → 100% pilot, 55% → 93% expanded.

---

## Domain Assessment: Predicted

### Road Surface Conditions (RSCD) — dry vs wet

| Dimension | Score | Notes |
|---|---|---|
| G | Medium-high | Reflective sheen, aggregate darkening, water film are camera-observable; temperature/physics not |
| V | High | Qwen says "grey road"; pavement engineer says "uniform specular sheen across aggregate matrix with darkened interstitial zones" |
| B | Unknown — to be measured | Visual difference is subtle; likely 50–70% zero-shot |
| E | High | Pavement friction research has clear visual indicator vocabulary |

**Prediction**: High patchability. Primary value: dry vs wet is a genuinely
hard confusable pair visually. Secondary value: RSCD provides 600K labeled
images for pool validation at scale.

**Note on RSCD actual content**: The 1-million image version contains three
friction classes only — dry, wet, water (standing water). No ice or snow
classes exist in this release. The dry vs wet pair is the primary DD target;
wet vs water is a secondary pair with more obvious visual distinction.

---

### Road Surface Conditions (RSCD) — wet vs water

| Dimension | Score | Notes |
|---|---|---|
| G | High | Specular reflections, mirror-like surface, light reflections from surroundings all observable |
| V | Medium | The visual difference is fairly obvious even to a non-expert |
| B | Likely 70–80% | Water (standing water) is visually more distinct than wet |
| E | High | Clear expert vocabulary available |

**Prediction**: Medium patchability — less interesting than dry vs wet because
B is likely already moderate and V is lower.

---

### Drone Swarm / Person Under Panel

| Dimension | Score | Notes |
|---|---|---|
| G | Medium (scout tier), High (commander tier) | Edge shadow gap at ≥5cm observable at scout altitude; ridge curvature marginal |
| V | Very high | Qwen says "metal panel on debris"; SAR analyst says "non-uniform edge shadow gap indicating localised panel lift over body mass" |
| B | Very low (predicted ~10–20%) | Panel occludes person completely; zero training examples of this class |
| E | High | SAR analysts can clearly articulate the optical correlates |

**Prediction**: High patchability — the very low B and very high V make this
an especially strong DD candidate. The G score at scout tier requires careful
grounding (some criteria must be simplified for 12MP / 30–50m altitude).

---

### SAR Imagery (direct optical VLM on SAR images)

| Dimension | Score | Notes |
|---|---|---|
| G | Very low (~0.05) | Radar backscatter is physically invisible to an optical VLM |
| V | Very high | Complete modality gap |
| B | Near random | VLM has no grounding for SAR pixel values |
| E | High | SAR analysts have rich vocabulary |

**Prediction**: Very low patchability for direct DD. The modality barrier
overrides the vocabulary gap. High V with near-zero G is the signature of
a perception barrier masquerading as a vocabulary gap. See cross-modal DD
(FleetPatch use case) for the correct approach.

---

### Candlestick Patterns (finance)

| Dimension | Score | Notes |
|---|---|---|
| G | High | All pattern features (body length, wick ratios, gap positions) are camera-observable |
| V | High | Qwen says "red and green bars"; trader says "bearish engulfing with upper shadow exceeding body length 2.5×" |
| B | Unknown — estimated 50–65% | 103 named patterns; many visually similar; existing mAP ~0.61 |
| E | Medium-high | Pattern vocabulary is explicit but some patterns are contested between analysts |

**Prediction**: High patchability. 103 named patterns with explicit rules and
stable vocabulary is a strong DD candidate. See `usecases/` for planned
experiment.

---

## Pre-Session Diagnostic Protocol

Before committing to a full DD experiment on a new domain, run this 1-hour
diagnostic:

```
1. Collect 5 representative confusable-pair images (any available source)

2. Expert description probe (15 min):
   Ask one domain expert to describe each image in 2–3 sentences.
   Focus on what distinguishes the two classes.

3. PUPIL description probe (10 min):
   Run Qwen3-VL-8B on the same 5 images.
   Free description prompt: "Describe what you see in this image in detail."

4. Grounding check (15 min):
   Feed expert descriptions through KF grounding check.
   Count fraction of feature claims confirmed observable.
   → Compute G

5. Vocabulary divergence (5 min):
   Compare expert vs PUPIL descriptions manually.
   → Assess V qualitatively (high / medium / low)

6. Zero-shot accuracy (10 min):
   Run Qwen3-VL-8B on 20 confusable-pair images with classification prompt.
   → Compute B

7. Expert convergence (5 min):
   Did the expert answer confidently and specifically?
   → Assess E qualitatively (high / medium / low)

8. Apply patchability matrix:
   G > 0.70, V high, 0.40 < B < 0.75, E high → proceed with DD experiment
   Any elimination condition met → reconsider or redesign
```

Total time: ~1 hour, ~25 images, one expert consultation.
Expected output: go/no-go decision with written rationale.

---

## Theoretical Basis

The patchability framework maps onto a simple latent capability model:

```
PUPIL accuracy = f(latent_perceptual_capability, vocabulary_alignment)

latent_perceptual_capability = g(sensor_resolution, modality_match, training_distribution)
vocabulary_alignment         = h(expert_vocabulary ∩ PUPIL_vocabulary)

DD intervention: increases vocabulary_alignment
DD cannot increase: latent_perceptual_capability
```

G measures whether latent perceptual capability exists for the expert's
features. V measures the gap in vocabulary alignment. B measures the
combined current performance. DD can increase V toward 1.0 — but only if
G is already high enough that the features exist in the perceptual signal.

This also explains why the grounding check is architecturally central to DD
rather than just a quality filter: it is the mechanism by which vocabulary
alignment and perceptual capability are kept separate, preventing rules that
sound expert-grounded but require perception the PUPIL doesn't have.

---

## Open Questions

1. **Can G be measured automatically** without running the full grounding
   check? An automated G-probe could make the diagnostic faster.

2. **Is the B sweet spot (0.40–0.75) a universal constant** or domain-
   dependent? A domain with extreme class imbalance might have a different
   useful range.

3. **Does patchability predict the number of DD rounds needed**, or just
   whether DD works at all? High V + high G might predict faster convergence
   (fewer grounding rounds) than medium V + medium G.

4. **Multi-domain transfer**: if DD rules are authored for domain A, do they
   help in domain B with similar visual structure? The patchability framework
   does not yet address cross-domain rule transfer.

5. **Expert quality sensitivity**: the framework assumes expert convergence (E)
   but does not model what happens when the expert is only partially qualified
   or when expert vocabulary is contested within the field.
