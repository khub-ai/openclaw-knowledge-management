# FleetPatch: Cross-Modal Knowledge Propagation for Heterogeneous Drone Swarms

> **New to Dialogic Distillation?** This use case involves multiple sensor
> modalities, heterogeneous hardware tiers, and fleet-scale knowledge
> propagation — it is among the more advanced applications in this
> repository. If you prefer to start with something simpler and
> self-contained, the
> [bird species identification](../../image-classification/birds/README.md)
> and [dermatology](../../image-classification/dermatology/README.md) use
> cases demonstrate the same core DD mechanism on single-camera,
> single-domain classification tasks with publicly available datasets and
> results you can reproduce in under an hour.
>
> **Status**: Design and scenario specification complete. Simulation
> prototype in planning. No physical hardware required.
>
> **Also see**: [Robotics Use Case Overview](../README.md) for the broader
> context of Knowledge Fabric in embodied systems.

---

## The One-Line Summary

A SAR drone found what 38 optical drones confidently missed; a human expert
explained why in plain language; DD turned that explanation into a patch and
instantly updated every drone in the fleet — including retroactively
reclassifying 90 minutes of already-captured footage — without retraining a
single model.

---

## Contents

1. [Start Here: The Simple Version](#1-start-here-the-simple-version)
2. [The Full Scenario: Heterogeneous Drone Swarm](#2-the-full-scenario-heterogeneous-drone-swarm)
3. [What DD Does Here](#3-what-dd-does-here)
4. [Without DD vs With DD](#4-without-dd-vs-with-dd)
5. [Why This Is Hard Without DD](#5-why-this-is-hard-without-dd)
6. [Fleet Architecture](#6-fleet-architecture)
7. [The Three Roles of DD in This Context](#7-the-three-roles-of-dd-in-this-context)
8. [Simulation Setup](#8-simulation-setup)
9. [Getting Started](#9-getting-started)

---

## 1. Start Here: The Simple Version

Before the drone scenario, consider a simpler version of the same problem.

A leisure company operates 50 public swimming pools, each fitted with AI
cameras that watch for drowning. The AI was trained on thousands of hours of
pool footage and works well for most situations. Management is confident.

One afternoon, a teenager gets into difficulty in the deep end. The AI sees a
person quietly in the water and scores it as normal swimming — 0.94 confidence.
The lifeguard on duty is watching another section. The teenager survives, but
barely.

The investigation reveals the problem. Real drowning is silent. There is no
splashing, no waving. The victim goes vertical, head tilted back with the mouth
just at the water surface, arms pressing down rather than stroking, body still
except for a slow bob. Experienced lifeguards call it the **instinctive drowning
response**. The AI had never been trained on it.

A senior lifeguard trainer — thirty years on the job — looks at the footage and
describes immediately what she sees:

> *"Head back, mouth at water line. Arms wide, pressing down — not pulling
> through. Vertical. Not moving forward. That is a drowning person. Any one of
> these alone means nothing. All together means someone is about to go under."*

That description becomes a rule. Not a retrained neural network — a rule,
written in plain language, checked to ensure every feature in it is observable
by a poolside camera, then broadcast to all 50 pools simultaneously. Under a
minute. No engineer required.

Within the same afternoon, the system retroactively reprocesses footage from
all 49 other pools over the past month. It finds two prior incidents where the
same silent pattern appeared seconds before the lifeguard intervened — both
previously scored as normal swimming.

**This is DD as an instant patch.** An expert describes what the AI should have
seen. The description becomes a rule. The rule reaches every camera in the
network immediately. The archived footage is re-examined with fresh eyes.

The drone swarm scenario below is the same mechanism operating at greater
scale, across more complex hardware, with one additional element: the expert's
knowledge comes from a *different sensor* than the cameras that need to act on
it.

---

## 2. The Full Scenario: Heterogeneous Drone Swarm

### Setup

A magnitude 6.8 earthquake has collapsed a 12-story residential building.
Visibility is poor — dust and intermittent smoke. A 40-drone swarm is deployed:

| Tier | Count | Sensor | Compute | Role |
|---|---|---|---|---|
| Scout | 38 | 12MP RGB camera | Cortex-M MCU | Cover ground, classify continuously |
| Commander | 2 | RGB + SAR payload | Jetson Orin NX + Qwen3-VL-8B | Deep sensing, TUTOR-capable |

**SAR** (Synthetic Aperture Radar) sees through smoke, darkness, and debris by
emitting radar pulses and measuring reflected energy. Metal structures —
including the human body — reflect strongly. A person buried under rubble
appears as a coherent bright return in the SAR image, invisible to optical
cameras.

The optical scouts cover 100% of the search area continuously. The SAR
commanders cover roughly 15% per pass, at higher cost and lower frequency.

### The failure

After 90 minutes of operation, 7 survivors have been located — all in
accessible surface positions. Commander drone C1 makes a pass over Grid Zone 7
and its SAR image shows a strong, spatially coherent double-bounce return
beneath what appears optically to be a corrugated metal roof panel lying flat
on rubble. The SAR analyst at the ground station flags it as a high-probability
survivor.

Scout drone S17 passed over the same coordinates 22 minutes earlier. Its
classification: **no person detected, confidence 0.94.**

The optical image shows a corrugated metal panel, approximately 2×3 metres,
lying across a rubble mound. No visible skin, limbs, or clothing. To the
classifier, this is unambiguously empty debris. The survivor — buried beneath
the panel — is completely invisible.

### Why the AI fails here

A metal panel lying flat on level rubble sits uniformly: consistent corrugation
ridges, no edge elevation, flush to the substrate. When a person is underneath,
the panel is tented by the body. One or more edges lift 4–8 cm off the
substrate, creating an irregular shadow line. The corrugation ridges near the
panel centre show a slight upward bow inconsistent with gravity.

These features are visible in the optical image. The AI had simply never been
trained to look for them. And at 0.94 confidence, the frame was never flagged
for human review.

### The DD loop

The SAR analyst is shown S17's optical frame alongside the SAR confirmation and
asked: *"What should the optical scout have seen?"*

The analyst responds:

> *"A panel lying flat on rubble has no edge gap — it sits flush. When someone
> is underneath, the panel bridges the body and at least one edge lifts. Look
> for a shadow gap under the edge — present at some points along the edge, absent
> at others. Uniform tilt from a rubble mound looks different: the gap is even
> all the way along. A person creates a localised lift, not a uniform tilt."*

This description becomes a candidate rule. The KF grounding check verifies that
each criterion is observable by an optical camera at operational altitude. One
criterion — ridge curvature near the panel centre — is flagged as ambiguous at
typical drone resolution and removed. Pool validation against 24 archived frames
(6 confirmed survivor-under-panel, 18 confirmed empty panel) yields precision
1.0 after one contrastive tightening round.

**Rule accepted. Fleet broadcast. 47 seconds.**

### The outcome

All 38 scouts retroactively reprocess their 90-minute archive against the new
rule. Three additional survivors are identified from frames already captured.
The next week, the system flags a live incident on a different mission 11
seconds before a human analyst would have noticed.

---

## 3. What DD Does Here

This scenario is more complex than the pool example because the expert's
knowledge originates in a different sensor modality than the cameras that must
act on it. DD plays three distinct roles:

**Patch** — the same role as in the pool scenario. A deployed classifier has a
blind spot. An expert fills it. The fix reaches the entire fleet instantly,
without retraining.

**Synthesizer** — the expert's rule is not a simple translation. It integrates
knowledge from three sources that no single system previously held together:
what the SAR return reveals about physical reality (a person is underneath),
what that physical reality causes to appear optically (the panel tents, the edge
lifts), and what the optical sensor can actually resolve at operational altitude
(edge shadow gap yes, ridge curvature no). The resulting rule did not exist in
any of these sources individually.

**Propagator** — the rule propagates simultaneously to 38 heterogeneous
hardware tiers that could not have been updated by a weight-update approach
without separate retraining pipelines per architecture. The rule is a natural
language string, architecture-agnostic, deployable over any communication
channel, applicable retroactively to archived frames.

---

## 4. Without DD vs With DD

| Dimension | Without DD | With DD |
|---|---|---|
| Analyst workload | Small team reviews AI-flagged queue | Same |
| Novel presentation — visible | Missed confidently, never queued | Caught — rule fixes it in minutes |
| Novel presentation — buried | Missed until SAR re-pass (hours later) | Fixed after first SAR detection |
| Time to fleet-wide update | 6–24 hours (retrain, validate, deploy) | Under 2 minutes |
| Retroactive reprocessing | Not possible | Immediate, same session |
| Expert knowledge reach | One analyst's shift | Every drone, every future mission |
| Heterogeneous fleet update | Separate pipeline per model architecture | Single rule broadcast |
| Auditability | Score only | Full rule trace per classification |
| Rule revocation | Full retrain required | Delete from rule pool |

The "without DD" operation is not chaos. It is a well-organised triage system
with one structural failure: it cannot recover from confident errors on
presentations the classifier has never seen, and when a novel presentation is
discovered, there is no rapid path to propagate the fix fleet-wide.

---

## 5. Why This Is Hard Without DD

The fundamental obstacles without DD:

**The cold start problem.** No labeled examples exist for "person under metal
panel, visible only as edge tenting." Collecting enough examples to retrain
would take days. In a SAR operation, hours matter.

**The confidence trap.** The AI scores 0.94 on the wrong answer. Frames below
a confidence threshold get human review. Frames above it do not. A novel
presentation that closely resembles a known negative class will always score
high confidence — and will always be invisible to the human review queue.

**The modality gap.** SAR knowledge does not transfer to optical classifiers by
training on more optical data. The information that "there is a person under
this panel" does not exist in optical pixels until a human expert articulates
the cross-modal correlate. There is no supervised learning path that closes this
gap without that articulation step.

**The heterogeneous deployment problem.** Scout drones run MobileNetV3.
Commander drones run Qwen3-VL-8B. A weight update for one architecture provides
no benefit to the other. Two separate retraining pipelines, two validation
cycles, two deployment procedures — all while survivors remain buried.

**Retroactive blindness.** Every frame captured before the novel presentation
was identified is permanently classified by the old model. The accumulated
observations of 38 drones over 90 minutes cannot be revisited.

---

## 6. Fleet Architecture

```
Ground Station
┌──────────────────────────────────────────────────────┐
│  SAR Analyst (TUTOR)                                 │
│  KF Grounding Validator (Claude Sonnet)              │
│  Rule Pool (broadcast over mesh)                     │
│  Semantic Map (coordinates → hazard class)           │
└──────────┬───────────────────────────────────────────┘
           │ mesh network
    ┌──────┴──────┐
    │  Commander  │  ×2   SAR + RGB + Jetson Orin
    │  (TUTOR-    │       Qwen3-VL-8B
    │   capable)  │       Sees through smoke/debris
    └──────┬──────┘
           │ rule broadcast (47 seconds fleet-wide)
    ┌──────┴──────┐
    │   Scout     │  ×38  RGB only + Cortex-M MCU
    │   (PUPIL)   │       Lightweight classifier
    │             │       Covers 100% of search area
    └─────────────┘
```

The commander tier acts as the TUTOR-capable node: it has the compute and
sensor diversity to identify novel presentations and to articulate cross-modal
rules. The scout tier is the PUPIL fleet: cheap, numerous, and continuously
updated by rules broadcast from the commander tier and ground station.

---

## 7. The Three Roles of DD in This Context

| Role | What it does | Pool scenario equivalent |
|---|---|---|
| **Patch** | Fixes a blind spot in a deployed classifier, instantly, fleet-wide | Yes — identical |
| **Synthesizer** | Composes new knowledge from SAR data, expert reasoning, and sensor capability constraints — produces an artifact no single source contained | No — pool scenario is single-modality |
| **Propagator** | Broadcasts architecture-agnostic rules to heterogeneous hardware tiers; applies retroactively to archived frames | Partial — pool scenario has homogeneous cameras |

The pool scenario demonstrates Patch. The drone swarm scenario demonstrates
all three, and introduces the cross-modal synthesis as the novel technical
contribution.

---

## 8. Simulation Setup

No physical hardware is required to demonstrate this use case. The recommended
simulation stack:

**For visual fidelity (DD rule validation):**
- **AirSim** (Unreal Engine, photorealistic RGB rendering) — best visual quality
  for testing whether Qwen3-VL-8B can observe DD rule criteria in rendered images
- Synthetic SAR images can be approximated using available SAR simulation tools
  or replaced with real archived SAR data for the TUTOR loop

**For physics and flight dynamics:**
- **Gazebo + PX4 SITL** — most mature multi-drone simulation; ROS2 native;
  40-drone swarm feasible on a single workstation
- Supports mesh network simulation between drone nodes

**For the full integrated demo:**
- AirSim handles rendering → optical classification by Qwen3-VL-8B
- Gazebo + PX4 handles flight dynamics and swarm coordination
- DD loop runs on host machine, consumes rendered frames, broadcasts rules
- Ground station exposed as MCP server: `navigate_to()`, `update_hazard_map()`,
  `broadcast_rule()`, `reprocess_archive()`

See [DESIGN.md](DESIGN.md) for the full integration architecture.

---

## 9. Getting Started

The DD loop at the core of this use case uses the same library as the birds
and dermatology experiments:

```bash
# Prerequisites
pip install anthropic           # TUTOR and KF validator
pip install transformers        # Qwen3-VL-8B PUPIL

# Run a standalone DD session (no simulator required)
# Provide: a failure image, a confirmation image, a pool of labeled frames
cd usecases/robotics/drone-swarm/python
python run_dd_session.py \
    --failure-image path/to/panel_frame.jpg \
    --confirmation "SAR confirmed survivor under panel at these coordinates" \
    --pool-dir path/to/labeled_pool/ \
    --tutor-model claude-opus-4-6 \
    --validator-model claude-sonnet-4-6 \
    --pupil-model qwen/qwen3-vl-8b-instruct
```

For the full swarm simulation, see [DESIGN.md](DESIGN.md) §5 for setup
instructions.

The domain configuration for this use case is in
[`python/domain_config.py`](python/domain_config.py).
