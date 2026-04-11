# FleetPatch Design Specification

This is the developer-facing design specification for the drone swarm cross-modal
fleet patching use case. The user-facing scenario and motivation are in
[README.md](README.md).

---

## 1. Core Contribution

This use case demonstrates a novel application of Dialogic Distillation (DD)
where:

1. **The TUTOR's knowledge originates in a different sensor modality** than the
   PUPIL's perception. A SAR analyst authors rules that optical cameras must
   act on — a cross-modal synthesis step that has no equivalent in standard DD.

2. **The PUPIL is a heterogeneous fleet**, not a single model. Rules must be
   grounded against multiple hardware tiers with different sensors, resolutions,
   and compute budgets simultaneously.

3. **Retroactive reprocessing** — applying a newly minted rule to the swarm's
   archived frame buffer — is a first-class operation rather than an optional
   bonus.

4. **The rule is the unit of fleet synchronisation.** Instead of distributing
   weight updates (architecture-specific, multi-megabyte, slow), the system
   distributes natural language rules (architecture-agnostic, ~200 tokens,
   sub-minute).

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GROUND STATION                           │
│                                                                 │
│   SAR Analyst ──── TUTOR dialogue ──── Rule candidate          │
│                                              │                  │
│   KF Grounding Validator (Claude Sonnet) ────┤                  │
│     • Per-tier observability check           │                  │
│     • Removes unobservable criteria          │                  │
│                                              ↓                  │
│   Pool Validator (parallel asyncio) ─── Rule accepted?         │
│     • Precision gate: ≥ 0.90               │                   │
│     • Max FP: 0                             │                   │
│     • Contrastive tightening if needed      │                   │
│                                              ↓                  │
│   Rule Pool (KF artifact store) ◄─── Rule registered           │
│     • Version-controlled                    │                   │
│     • Tier-specific observational variants  │                   │
│     • Revocable                             │                   │
│                                              ↓                  │
│   Broadcast Engine ──────────────── Mesh network               │
│   Archive Reprocessor ──────────── Frame buffer query          │
│                                                                 │
│   Semantic Hazard Map                                           │
│     (coordinates → hazard_class, confidence, rule_id)          │
└─────────────────────────────────────────┬───────────────────────┘
                                          │
                     ┌────────────────────┼──────────────────────┐
                     │                   mesh                     │
          ┌──────────▼──────┐                    ┌───────────────▼──────┐
          │  COMMANDER ×2   │                    │    SCOUT ×38         │
          │                 │                    │                      │
          │ SAR payload     │                    │ RGB 12MP             │
          │ RGB 20MP        │                    │ Cortex-M MCU         │
          │ Jetson Orin NX  │                    │ Lightweight PUPIL    │
          │ Qwen3-VL-8B     │                    │ Rule-injected ctx    │
          │                 │                    │                      │
          │ Detects novel   │                    │ Covers 100% area     │
          │ presentations   │                    │ Continuous sweep     │
          │ via SAR         │                    │ Reports low-conf     │
          └─────────────────┘                    └──────────────────────┘
```

---

## 3. DD Protocol Adaptations

The core DD protocol (from `core/dialogic_distillation/`) requires two
adaptations for this use case:

### 3.1 Cross-modal TUTOR prompt

Standard DD: *"The PUPIL misclassified this image. What visual features should
it have attended to?"*

Cross-modal adaptation: *"The PUPIL's optical camera classified this zone as
[class]. SAR data confirms the ground truth is [class]. The optical image is
shown below. What features visible in the optical image correlate with what the
SAR data reveals, such that a camera-only system could have reached the correct
classification?"*

The TUTOR is not asked to describe SAR features — they are asked to bridge from
SAR knowledge to optical observables. This is a distinct cognitive task that
the prompt must frame explicitly.

### 3.2 Multi-tier grounding check

Standard DD runs one grounding check against the validator model.

Multi-tier adaptation: the grounding check runs once per hardware tier, with
tier-specific context injected:

```python
TIER_CONTEXTS = {
    "scout": (
        "You are evaluating whether an optical camera with 12MP resolution "
        "at 30-50m altitude can observe the described feature. Features "
        "requiring sub-centimetre detail, colour accuracy beyond broad "
        "categories, or direct contact with the subject are not observable."
    ),
    "commander": (
        "You are evaluating whether a 20MP optical camera at 15-25m altitude "
        "with stable hover capability can observe the described feature."
    ),
}
```

A rule criterion that fails grounding for the scout tier but passes for the
commander tier is retained in a tier-specific variant. The scout receives a
simplified version; the commander receives the full version.

### 3.3 Retroactive reprocessing

After a rule is accepted, the archive reprocessor runs the rule against the
swarm's frame buffer:

```python
async def reprocess_archive(
    rule: Rule,
    frame_buffer: FrameBuffer,
    lookback_seconds: int = 5400,  # 90 minutes
    tier: str = "scout",
) -> list[ReclassifiedFrame]:
    frames = frame_buffer.query(
        max_age_seconds=lookback_seconds,
        tier=tier,
        original_class="no_person",          # only re-examine negatives
        confidence_min=0.70,                  # confident negatives only
    )
    results = await asyncio.gather(*[
        apply_rule_to_frame(rule, frame, tier=tier)
        for frame in frames
    ])
    return [r for r in results if r.reclassified]
```

This is a first-class operation in the protocol, not a post-hoc step. The
retroactive results are reported to the ground station and added to the semantic
hazard map immediately.

---

## 4. Domain Configuration

```python
# python/domain_config.py

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from core.dialogic_distillation.protocols import DomainConfig

DRONE_SWARM_CONFIG = DomainConfig(
    expert_role=(
        "senior SAR (Synthetic Aperture Radar) analyst and search-and-rescue "
        "coordinator with field experience interpreting radar returns alongside "
        "optical footage in disaster response operations"
    ),
    item_noun="optical drone camera frame",
    item_noun_plural="optical drone camera frames",
    classification_noun="survivor presence assessment",
    class_noun="presence class",
    feature_noun="optical surface feature or shadow pattern",
    observation_guidance=(
        "surface reflectivity and shadow patterns, object edge geometry, "
        "debris topology and elevation inconsistencies, panel lift indicators, "
        "irregular silhouettes inconsistent with flat-lying debris, "
        "thermal anomalies if thermal camera available"
    ),
    non_visual_exclusions=(
        "SAR return values, radar backscatter intensity, subsurface conditions, "
        "infrared temperature readings (unless thermal camera confirmed present), "
        "audio, seismic sensors, time of day, weather conditions"
    ),
    good_vocabulary_examples=[
        (
            "corrugated panel edge shows a shadow gap approximately 5-6cm wide "
            "along the left edge, present at the centre of the edge but absent "
            "at the corners — consistent with localised lift over a body mass"
        ),
        (
            "irregular elevation at panel centre: corrugation ridges in the "
            "central third bow slightly upward relative to ridges at the periphery"
        ),
        (
            "panel surface shows non-uniform light reflection: specular highlight "
            "present at one end, diffuse return at the other, inconsistent with "
            "a flat-lying panel under uniform illumination"
        ),
    ],
    bad_vocabulary_examples=[
        "SAR data shows a strong double-bounce return (uses non-optical modality)",
        "thermal hotspot detected beneath the panel (thermal not available on scouts)",
        "survivor confirmed by ground team (non-visual external confirmation)",
        "panel is bridging a body (structural inference, not optical observation)",
    ],
    precision_gate=0.90,   # safety-critical: false negatives cost lives
    max_fp=0,              # zero tolerance for false positives in this domain
)

# Tier-specific observability constraints injected into grounding check
TIER_OBSERVABILITY = {
    "scout": (
        "Camera: 12MP RGB. Altitude: 30-50m AGL. No stabilisation. "
        "Pixel footprint: approximately 1.5-3cm per pixel. "
        "Features requiring sub-5cm resolution are NOT observable."
    ),
    "commander": (
        "Camera: 20MP RGB. Altitude: 15-25m AGL. Stabilised gimbal. "
        "Pixel footprint: approximately 0.5-1.5cm per pixel. "
        "Most surface texture features observable."
    ),
}
```

---

## 5. Simulation Stack

### 5.1 Option A: AirSim + Gazebo (recommended)

AirSim provides photorealistic Unreal Engine rendering for visual classification
tasks. Gazebo + PX4 SITL provides realistic flight dynamics and swarm
coordination. The two communicate via ROS2.

```
AirSim (Unreal Engine)          Gazebo + PX4 SITL
  ├── RTX rendered frames   ←→   ├── Flight dynamics
  ├── RGB camera simulation      ├── 40-drone swarm
  ├── Scene with debris,         ├── Mesh network sim
  │   rubble, metal panels       └── ROS2 bridge
  └── Ground truth labels
        ↓
  Qwen3-VL-8B (host GPU)
  + DD rule injection
        ↓
  Classification output
        ↓
  Semantic map update
        ↓
  Path planner (Nav2)
        ↓
  Velocity commands → PX4
```

**Hardware requirement**: RTX 4090 (24GB VRAM) or two GPUs. Qwen3-VL-8B
requires ~16GB VRAM; AirSim RTX rendering requires ~8GB VRAM. Both can run on
a single 24GB card at reduced rendering quality, or separated across two GPUs.

### 5.2 Option B: Gazebo only (lower fidelity, easier setup)

Replace AirSim with Gazebo's built-in camera sensor. Lower visual fidelity —
the sim-to-real gap for VLM classification is larger — but the DD loop and
fleet propagation are fully demonstrable. Suitable for validating the
architecture before investing in AirSim integration.

### 5.3 Synthetic SAR approximation

Full SAR simulation (NVIDIA SimRadar, AFRL DIRSIG) is complex. For the demo,
SAR can be approximated by:

1. **Real archived SAR data** (Sentinel-1, ICEYE open archive) annotated with
   ground truth — use for the TUTOR dialogue only, not for the PUPIL
2. **SAR simulation tools** (RaySAR, openSAR) applied to the 3D scene model
3. **Ground truth oracle** — replace SAR with a ground truth label feed for
   the TUTOR input, and note this simplification in the paper

For the research contribution, the SAR modality only needs to appear in the
TUTOR dialogue. The PUPIL never sees SAR images; it only receives the optical
correlate rules that DD produces.

---

## 6. MCP Server Interface

The ground station exposes an MCP server that Claude Code (or any MCP client)
can call to issue natural language commands that result in swarm behaviour
changes:

```python
# Available MCP tools

navigate_to(
    location: str,              # "grid_zone_7", "extraction_point_A"
    avoid: list[str] = [],      # ["near_pillar_3", "zone_7_east"]
    mode: str = "normal",       # "normal" | "careful" | "grid_sweep"
) -> NavigationResult

update_hazard_map(
    zone_id: str,               # coordinates or named zone
    hazard_class: str,          # "survivor_under_panel", "unstable_debris"
    confidence: float,          # from DD pool validation
    rule_id: str,               # which rule triggered this
) -> None

broadcast_rule(
    rule: Rule,                 # accepted DD rule
    tiers: list[str],           # ["scout", "commander"] or ["scout"]
) -> BroadcastResult           # includes per-drone acknowledgement

reprocess_archive(
    rule_id: str,
    lookback_seconds: int = 5400,
    tier: str = "scout",
) -> list[ReclassifiedFrame]

get_swarm_state() -> SwarmState  # positions, coverage map, current rules

get_camera_frame(
    drone_id: str,              # "S17", "C1"
    timestamp: str = "latest",
) -> ImageFrame                 # returned as base64 for DD input
```

**Example Claude Code session**:

```
User: "Run DD on S17's frame from 22 minutes ago against the SAR confirmation
       at zone 7. If a rule is accepted, broadcast to all scouts and reprocess
       the last 90 minutes."

Claude calls:
  1. get_camera_frame("S17", timestamp="-22m")
  2. run_dd_session(frame, confirmation="SAR strong return at zone 7 coords")
  3. [if rule accepted] broadcast_rule(rule, tiers=["scout"])
  4. [if rule accepted] reprocess_archive(rule.id, lookback_seconds=5400)
  5. get_swarm_state()  → reports 3 new survivor candidates from archive
```

---

## 7. Evaluation Framework

### 7.1 Primary metrics

| Metric | Measurement | Target |
|---|---|---|
| Time to fleet-wide rule deployment | Wall-clock from rule acceptance to last drone acknowledgement | < 60 seconds |
| Retroactive recall | Survivors found in archive / total survivors present in archive | > 0.80 |
| Rule precision on pool | TP / (TP + FP) on held-out validation pool | ≥ 0.90 |
| Cross-tier rule consistency | Fraction of scout classifications matching commander on same frame | > 0.85 |
| Conventional pipeline latency | Time to equivalent update via retrain/validate/deploy | Baseline (hours) |

### 7.2 Ablation conditions

| Condition | What it removes | Purpose |
|---|---|---|
| No DD, no retraining | Baseline AI only | Shows structural blind spot |
| No DD, human review queue | Human analysts review flagged frames | Shows confident-miss problem |
| DD without grounding check | Rules deployed without tier observability filter | Shows unobservable criteria harm precision |
| DD without retroactive | Rule applies forward only | Quantifies retroactive value |
| DD single-tier (no heterogeneous) | All scouts use commander-tier rule verbatim | Shows value of tier-specific variants |

### 7.3 Confusable pair

The primary confusable pair for this use case:

**person_under_panel** vs **empty_panel**

Both look like: a corrugated or flat metal panel lying on rubble. Ground truth
requires SAR confirmation or physical extraction.

Secondary pairs (for expanded experiments):
- person_under_emergency_blanket vs empty_emergency_blanket
- person_in_void_space vs debris_void (dark gap in rubble)
- person_partially_buried vs rubble_mound (irregular elevation)

---

## 8. Data Strategy

### 8.1 Simulation-first

The first experiments generate all frames from the AirSim simulation. DD rules
are distilled from simulated frames with known ground truth. This is internally
consistent: the TUTOR authors rules about simulated optical features, the
grounding check validates them against simulated frames, the pool validation
uses simulated labeled pairs. The sim-to-real gap is acknowledged as future work.

### 8.2 Real data augmentation

Several public datasets contain relevant imagery for augmenting or validating
the simulation results:

| Dataset | Content | Relevance | Access |
|---|---|---|---|
| AIDER (Aerial Image Dataset for Emergency Response) | Aerial RGB of collapsed buildings, flooding, fire | Scene realism for debris fields | GitHub |
| HERIDAL | Aerial imagery for person detection in wilderness SAR | Person-in-scene examples | IEEE DataPort |
| UAV123 | 123 aerial sequences, various targets | Drone perspective optical examples | Public |
| Sentinel-1 SAR archive | Real SAR imagery, various resolutions | TUTOR dialogue grounding | Copernicus Open Access Hub |
| FloodNet | Post-hurricane UAV imagery | Disaster scene texture | GitHub |

For the DD loop specifically, only the TUTOR needs SAR imagery (Sentinel-1).
The PUPIL and pool validation can use simulated or AIDER/HERIDAL optical frames.

### 8.3 Minimum viable dataset for a DD session

The DD loop requires far fewer examples than supervised training:

- 1 confirmed failure frame (the PUPIL's misclassified image)
- 1 ground truth confirmation (SAR or manual)
- 20–40 labeled pool frames (10–20 positive, 10–20 negative)

All of these can be sourced from simulation or from the public datasets above,
making the first experiment achievable without collecting new data.

---

## 9. Repository Layout

```
usecases/robotics/drone-swarm/
├── README.md                    ← User-facing scenario and motivation
├── DESIGN.md                    ← This file
├── knowledge_base/
│   ├── .gitkeep
│   └── person_under_panel_vs_empty_panel.json   ← after first session
├── python/
│   ├── domain_config.py         ← DomainConfig + tier observability contexts
│   ├── agents.py                ← Wraps core DD agents with drone swarm config
│   ├── fleet.py                 ← Swarm state, tier management, rule broadcast
│   ├── archive.py               ← Frame buffer + retroactive reprocessing
│   ├── mcp_server.py            ← MCP tool definitions for Claude Code control
│   ├── simulation/
│   │   ├── airsim_bridge.py     ← AirSim camera frame capture
│   │   ├── gazebo_bridge.py     ← Gazebo + PX4 SITL interface
│   │   └── scene_builder.py     ← Disaster scene construction
│   └── run_dd_session.py        ← Standalone DD session runner
└── assets/
    ├── panel_failure_frame.jpg  ← Example failure case for documentation
    └── sar_confirmation.png     ← SAR return showing survivor
```

---

## 10. Implementation Sequence

Recommended build order to reach a demonstrable result with minimum effort:

**Phase 1 (standalone DD, no simulator)**
- Implement `domain_config.py` with cross-modal TUTOR prompt
- Implement multi-tier grounding check in `agents.py`
- Collect 20–40 labeled pool frames from AIDER/HERIDAL or simulation
- Run one complete DD session end-to-end
- Validate rule precision on held-out pool

**Phase 2 (fleet simulation, no SAR)**
- Integrate Gazebo + PX4 SITL with 10-drone reduced swarm
- Implement `fleet.py` with rule broadcast and retroactive reprocessing
- Use ground truth oracle in place of SAR for TUTOR input
- Demonstrate: fleet misses novel case → DD session → fleet patched → archive reprocessed

**Phase 3 (full AirSim integration)**
- Replace Gazebo camera feed with AirSim RTX rendering
- Validate that Qwen3-VL-8B can observe DD rule criteria in rendered frames
- Scale swarm to 40 drones
- Add MCP server for Claude Code control loop

**Phase 4 (real data validation)**
- Substitute real Sentinel-1 SAR imagery for TUTOR dialogue
- Test rules on HERIDAL aerial person detection frames
- Quantify sim-to-real gap; adjust rule vocabulary as needed

---

## 11. Connection to Broader KF Architecture

This use case extends DD in three directions not previously explored:

| Extension | Prior DD work | This use case |
|---|---|---|
| Sensor modality | Single modality throughout | Cross-modal: TUTOR in SAR, PUPIL in optical |
| Fleet scale | Single PUPIL model | 40 heterogeneous PUPIL instances |
| Temporal scope | Forward-only rule application | Retroactive archive reprocessing |

The rule pool is a KF artifact. The semantic hazard map is a KF artifact. The
fleet broadcast protocol is a KF distribution mechanism. This use case is
therefore not just a DD application — it is a demonstration of KF as a
runtime knowledge infrastructure for multi-agent embodied systems.

The MCP server interface means the entire system is controllable through natural
language from a Claude Code session, establishing a clean separation between
the cognitive layer (Claude + KF) and the physical layer (drone fleet + physics
simulation).
