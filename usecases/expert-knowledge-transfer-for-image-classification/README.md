# Use Case 02: Expert Knowledge Transfer via Natural Language
## The "Excel Moment" for AI — Teaching a Model Without Machine Learning

---

> **Status**: Experiment complete — results in Section 8
> **Theme**: KF as a knowledge authoring tool for domain experts with no AI expertise

---

## 1. The Problem

A domain expert — an ornithologist, a clinician, a compliance officer, a materials scientist — possesses discriminative knowledge that no model has been trained on. The knowledge exists in their head, in field guides, in diagnostic manuals, in internal training materials. Today, transferring that knowledge to an AI system requires:

- Collecting hundreds or thousands of labeled examples
- Running a fine-tuning job (GPU hours, ML expertise, cost)
- Waiting days to weeks for results
- Iterating blindly without being able to inspect what the model "learned"
- Starting over if the domain shifts

None of this is accessible to a domain expert who knows nothing about AI.

**Knowledge Fabric proposes a different path**: the expert describes, in plain natural language, *why* one case belongs to class A and not class B. KF extracts those explanations into structured, verified, reusable rules. The model applies those rules to new cases immediately. The knowledge is portable — a file that can be shared, audited, and transferred to a different model or organization.

This is the **Excel moment for AI**: just as Excel gave non-programmers the ability to do computation that previously required software engineers, KF gives domain experts the ability to teach AI systems that previously required ML teams.

---

## 2. The Experiment Domain: Fine-Grained Bird Species Classification

### Why birds

Fine-grained visual classification of bird species is an ideal proving ground for this thesis:

1. **The gap is large and measurable**: CLIP zero-shot performance on CUB-200 is ~65%. Supervised SOTA (HERBS, 2023) is ~93%. The 28-percentage-point gap represents what expert discriminative knowledge is worth — and it is the space KF aims to close via NL teaching rather than labeled training data.

2. **The hard cases are well-defined**: The ~7% error rate at SOTA is concentrated in confusable species pairs — Downy vs. Hairy Woodpecker, Herring vs. Thayer's Gull, Orchard vs. Baltimore Oriole. These are exactly the cases where expert verbal criteria ("the bill is shorter relative to head size") matter most and where visual training on aggregate statistics fails.

3. **Expert NL reasoning is publicly available**: Field guides (Sibley, Kaufman, National Audubon), eBird species accounts, and ornithological literature contain precisely the discriminative reasoning KF needs as teaching input — for every species, for every confusable pair. No live expert is required for the experiment; documented expertise substitutes.

4. **The teaching is believable**: "This is a Hairy Woodpecker rather than a Downy because the bill is as long as the head depth, not shorter" is a sentence a non-expert can read, understand, and verify. The knowledge is transparent.

5. **Public benchmark with live leaderboard**: [Papers with Code — CUB-200](https://paperswithcode.com/sota/fine-grained-image-classification-on-cub-200) tracks ~70 published methods. Results are directly comparable.

### Dataset: CUB-200-2011

| Property | Value |
|---|---|
| Full name | Caltech-UCSD Birds-200-2011 |
| Images | 11,788 (5,994 train / 5,794 test) |
| Classes | 200 North American bird species |
| Annotations | Labels + 312 binary attribute annotations + 10 per-image NL descriptions (Reed et al.) |
| Access | Fully open — [Caltech download](https://data.caltech.edu/records/65de6-vp158) |
| Leaderboard | [Papers with Code](https://paperswithcode.com/sota/fine-grained-image-classification-on-cub-200) |
| CLIP zero-shot | ~65% |
| Supervised SOTA | 93.1% (HERBS, 2023) |

**Supplementary sources for expert NL teaching material**:
- Sibley Guide to Birds (standard ornithological reference — all 200 species covered)
- eBird species accounts at allaboutbirds.org — free, per-species, include identification tips and confusion species
- 312 binary attribute annotations (already in dataset) — structured visual criteria per species

---

## 3. Experiment Design

### The teaching session (simulated)

For each of the ~30 most confusable species pairs identified in the literature, construct a KF teaching session using language sourced from field guides and eBird accounts:

```
EXPERT TEACHING INPUT (example):

"A Hairy Woodpecker looks very similar to a Downy Woodpecker, but can be
distinguished by the following features:

1. Bill length: The Hairy's bill is approximately as long as its head is deep.
   The Downy's bill is noticeably shorter — roughly half the head depth.

2. Body size: The Hairy is larger overall, roughly the size of a Robin.
   The Downy is sparrow-sized.

3. Outer tail feathers: The Hairy has clean white outer tail feathers with
   no spots or bars. The Downy typically shows black spots on those feathers.

4. When in doubt: If you cannot judge bill length, look for the clean tail
   feathers on the Hairy."
```

KF's role: extract from this text the structured, verifiable rules that distinguish the two species; generate executable classification guidance; store both in the knowledge base with provenance linking back to the source.

### Conditions

| Condition | Description |
|---|---|
| **Baseline: zero-shot** | Frontier vision-language model (e.g., GPT-4V or Claude with vision), no fine-tuning, no KF |
| **Baseline: few-shot** | Same model with 3–5 labeled example images per species, no NL explanation |
| **KF: NL teaching only** | KF teaching session using field-guide text, no additional labeled images |
| **KF: NL teaching + few-shot** | KF teaching session + 3–5 labeled examples |
| **Reference: supervised SOTA** | HERBS or equivalent — requires full labeled training set |

### Primary metric

**Accuracy on confusable-pair test set**: Construct a held-out test set restricted to the ~30 most visually similar species pairs. Report per-pair and aggregate accuracy across conditions.

This metric is more diagnostic than overall accuracy because:
- Overall accuracy is dominated by easy cases (eagle vs. hummingbird) where all conditions perform well
- Confusable pairs expose what discriminative knowledge actually contributes
- The improvement narrative is cleaner: "KF closes most of the gap between zero-shot and supervised SOTA on the hard cases, using only NL teaching — no training data"

**Secondary metrics**:
- Accuracy on full CUB-200 test set
- Number of NL teaching sessions required to reach each accuracy level
- Knowledge base reuse rate: how many times is a given rule applied across different test images

---

## 4. The KF Role in This Experiment

This experiment is designed to stress-test three specific KF capabilities:

### 4.1 Rule extraction from NL description

The expert provides prose description of discriminative features. KF must:
- Parse the description into discrete, independently applicable rules
- Identify which rules are decisive vs. corroborating
- Detect and flag contradictions or ambiguities that require clarification
- Store rules in a form that is both human-readable and machine-applicable

### 4.2 Verified tool generation

For each extracted rule, KF generates an executable tool — in this domain, a visual feature extraction prompt or scoring function — and verifies it against the few labeled examples provided (or held-out validation images). Rules that cannot be verified against evidence are flagged, not silently stored.

### 4.3 Cross-species knowledge accumulation

The knowledge base should compound: rules learned for the Downy/Hairy pair (bill length relative to head) may generalize to other woodpecker confusions. KF should:
- Detect when a new rule is a specialization or extension of an existing rule
- Propose generalizations to the expert for confirmation
- Track which rules transfer and which are species-specific

---

## 5. The Narrative for the Excel Moment

The experiment should be designed to produce a specific before/after story:

**Before KF**: An ornithology professor wants to teach a computer vision system to correctly distinguish Downy from Hairy Woodpeckers — a confusion that even intermediate birders make. She knows nothing about machine learning. Her options are: (a) collect 500 labeled photos and hire an ML team, or (b) use the AI as-is and accept ~65% accuracy on this pair.

**With KF**: The professor opens a teaching session. She types — or dictates — the same explanation she gives students in her ornithology course. Twenty minutes later, the system has extracted her criteria, verified them against a handful of examples, and stored them as reusable rules. The model now performs substantially better on Downy/Hairy images. She exports the knowledge base and shares it with a colleague at another institution running a different model. It works there too.

**What KF claims**: The gap between "domain expert has knowledge" and "AI system has knowledge" can be closed with natural language, without ML expertise, without labeled training data at scale, and without model retraining.

---

## 6. Implementation Decisions

1. **Vision model**: GPT-4V. KF injects extracted rules into the system prompt or structured reasoning chain at inference time; no fine-tuning occurs. API key stored at the standard location on P drive.

2. **Teaching session format**: Teaching input is sourced from textually documented expert opinion — field guides (Sibley, Kaufman), eBird/All About Birds species accounts, and ornithological literature. These sources contain precisely the discriminative NL reasoning needed ("bill length relative to head depth", "clean vs. spotted outer tail feathers"). Visual nuance that cannot be verbalized is out of scope for this experiment; the thesis is that documented textual expertise is sufficient.

3. **Verification mechanism**: An expert is assumed available to converse with KF during the teaching and verification loop — the same SOLVER → SUPERVISOR → HUMAN escalation pattern used in the ARC-AGI-2 use case. When KF extracts a rule and is uncertain, it surfaces the rule to the expert for confirmation or correction before storing it. This makes the verification mechanism interactive rather than purely automated.

4. **Benchmark framing**: Report on both — the full CUB-200 test set for direct leaderboard comparison, and a curated confusable-pair subset for the narrative. The confusable-pair subset is the primary story; the full-set score provides credibility.

---

## 8. Experiment Results

*Run date: 2026-03-29. Model: GPT-4o. 15 confusable pairs, 20 test images per class per condition (40 images/pair). 118 KF rules in knowledge base, auto-accepted from field-guide text (no expert verification). Results file: `results/results_20260329T104850.json`.*

### 8.1 Overall accuracy

| Condition | Accuracy | vs. Zero-shot |
|---|---|---|
| Zero-shot | 78.0% (468/600) | — |
| Few-shot (3 train images/class) | 82.8% (497/600) | +4.8pp |
| KF-taught (NL rules injected) | 72.5% (434/599) | **−5.5pp** |

KF-taught underperformed zero-shot overall. Per-pair results reveal why: KF rules worked well for some pairs and caused systematic failures in others.

### 8.2 Per-pair breakdown

| Pair | Zero-shot | Few-shot | KF-taught | KF delta |
|---|---|---|---|---|
| American Crow vs Fish Crow | 68% | 68% | 49% | −19pp |
| Black-billed Cuckoo vs Yellow-billed Cuckoo | 78% | 88% | 85% | +7pp |
| Brewer Sparrow vs Clay-colored Sparrow | 82% | 95% | 50% | −32pp |
| Bronzed Cowbird vs Shiny Cowbird | 88% | 95% | **98%** | **+10pp** |
| California Gull vs Herring Gull | 45% | 40% | 50% | +5pp |
| Caspian Tern vs Elegant Tern | 85% | 92% | 88% | +3pp |
| Chipping Sparrow vs Tree Sparrow | 88% | 90% | 42% | **−45pp** |
| Common Raven vs White-necked Raven | 88% | 92% | 85% | −3pp |
| Common Tern vs Forster's Tern | 42% | 68% | 38% | −5pp |
| Herring Gull vs Ring-billed Gull | 87% | 93% | 87% | +0pp |
| Indigo Bunting vs Blue Grosbeak | 90% | 95% | 88% | −3pp |
| Least Flycatcher vs Western Wood-Pewee | 72% | 78% | 70% | −3pp |
| Loggerhead Shrike vs Great Grey Shrike | 80% | 55% | 72% | −8pp |
| Northern Waterthrush vs Louisiana Waterthrush | 75% | 72% | 75% | +0pp |
| Red-faced Cormorant vs Pelagic Cormorant | 82% | 95% | **92%** | **+10pp** |

### 8.3 What worked

KF rules produced measurable gains where visual rules were clear, diagnostic, and applicable to the images as photographed:

- **Red-faced Cormorant vs Pelagic Cormorant (+10pp)**: Rules about bare facial skin extent and bill color are unambiguous and visible in most images. GPT-4o applied them reliably.
- **Bronzed Cowbird vs Shiny Cowbird (+10pp)**: Plumage differences (bronzy-brown vs. glossy black-blue body on male, buff vs. grayish on female) are consistent and visible.
- **Black-billed Cuckoo vs Yellow-billed Cuckoo (+7pp)**: Bill color and undertail spot pattern are both visible features well described in the teaching material.

### 8.4 What failed — and why

The three largest KF failures share a common root: **rules that do not match what is visible in the images**.

**Chipping Sparrow vs Tree Sparrow (−45pp)**: The most damaging failure. CUB-200-2011 class 130 is the *Eurasian Tree Sparrow* (*Passer montanus*). The knowledge base was built using field-guide text about the *American Tree Sparrow* (*Spizelloides arborea*) — a completely different species. The injected rules systematically pushed predictions toward Chipping Sparrow for images that actually showed Eurasian Tree Sparrows. This is a category error introduced by auto-accepting rules without expert verification.

**Brewer Sparrow vs Clay-colored Sparrow (−32pp)**: Fine-grained sparrow rules describe subtle plumage differences (malar stripe, supercilium contrast, crown pattern) that are not consistently discernible in CUB photographs taken at variable angles and distances. The rules may also have bled across pair boundaries, since Brewer, Clay-colored, and Chipping Sparrows are all involved in overlapping confusable pairs.

**American Crow vs Fish Crow (−19pp)**: The most visually similar pair in the dataset (cosine sim 0.996). Most diagnostic features reference voice ("Fish Crow gives a nasal *uh-uh*"), range, and body size context — none of which are available from a single photograph. Rules that inject non-visual criteria confuse rather than help a vision model.

### 8.5 Interpretation

The experiment result is mixed but instructive. The key finding is not that KF rules fail to help — it is that **incorrectly specified or non-visual rules cause systematic harm**, while well-grounded visual rules reliably help.

The auto-acceptance mode used here (no HUMAN verification step) is the source of the failures:

1. The Tree Sparrow taxonomic error would have been caught immediately by any expert reviewing the extracted rules.
2. Rules referencing voice or range would be rejected by an expert as inapplicable to a visual task.
3. The sparrow rules flagging low-confidence features could be assigned lower weight or eliminated.

This is precisely the validation loop KF is designed to support: SOLVER extracts rules, SUPERVISOR presents them to HUMAN, HUMAN rejects non-applicable rules before they reach the knowledge base. Skipping this step produced the failures observed. The corollary: for pairs where rules were visually grounded and accurate, KF consistently outperformed zero-shot.

### 8.6 Path forward

1. **Re-run with expert verification**: Run the interactive teaching session (`kf_teacher.py` without `auto_accept=True`) and reject rules that reference non-visual features. Predict this would recover most of the lost ground.
2. **Fix Tree Sparrow taxonomy**: Update the teaching file to correctly describe Eurasian Tree Sparrow (*P. montanus*) — rufous crown, bold black cheek patch, white cheek with black spot. Predict this pair would jump from 42% to well above zero-shot.
3. **Add confidence-weighted rule injection**: Rules with `confidence: low` should be presented as weak priors, not directives.

---

## 7. References

- CUB-200-2011 dataset: [Caltech Perona Lab](https://www.vision.caltech.edu/datasets/cub_200_2011/)
- Papers with Code leaderboard: [Fine-Grained Classification on CUB-200](https://paperswithcode.com/sota/fine-grained-image-classification-on-cub-200)
- HERBS (SOTA, 2023): [arXiv 2303.06442](https://arxiv.org/abs/2303.06442)
- Reed et al. NL descriptions: [GitHub icml2016](https://github.com/reedscot/icml2016)
- Concept Bottleneck Models (related work): [NeurIPS 2020](https://proceedings.mlr.press/v119/koh20a.html)
- Beyond CBMs (NeurIPS 2024): [proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/file/9a439efaa34fe37177eba00737624824-Paper-Conference.pdf)
- CLIP zero-shot on CUB: [GWU study 2024](https://blogs.gwu.edu/pless/2024/06/10/comparative-study-of-clips-zero-shot-accuracy-on-cub-dataset/)
- eBird species accounts: [allaboutbirds.org](https://www.allaboutbirds.org)
