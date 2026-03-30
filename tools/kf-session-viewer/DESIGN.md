# KF Session Viewer Design

## Purpose

The KF Session Viewer is a reusable browser-based visualization shell for
Knowledge Fabric sessions. It is designed to support many use cases with very
different artifacts while preserving a common session model.

The immediate target is ARC-AGI-3, but the tool should also be suitable for
other domains such as expert-guided learning, document critique, and future KF
workflows.

## Design Principles

1. Session-centric
   The viewer operates on a generic session log rather than on a benchmark-
   specific runtime.

2. Adapter-based rendering
   Domain-specific artifact rendering should be isolated behind renderer
   adapters.

3. State and goal awareness
   The shared shell should treat evolving state snapshots and goals as
   first-class session elements rather than burying them inside generic
   metadata.

4. Entity-aware visualization
   The shell should surface changing Knowledge Fabric entities such as rules,
   tools, interpreted states, and goals in a renderer-independent panel.

5. Recency cues
   Because many session fields change continuously, the UI should highlight
   recently changed information and let that emphasis fade over time so users
   can quickly spot what just changed.

6. Tooltip-first details
   Core widgets should expose concise default displays with richer detail
   available on hover.

7. No required backend
   The viewer should work with a static JSON session bundle loaded from a file
   picker. A tiny local server is provided only for convenience.

8. Stable shell, swappable domains
   The playback controls, metadata layout, entity panels, and commentary
   sections should remain generic even when the artifact renderer changes.

9. Visual analysis first
   The UI should make state changes obvious. Diff overlays, clear notes, and
   per-step metadata matter more than decorative UI.

## High-Level Architecture

The tool is split into:

- generic viewer shell
  - session loading
  - timeline navigation
  - autoplay
  - side panels
  - renderer selection
  - state and entity rendering
  - recent-change highlighting

- renderer adapters
  - one adapter per artifact type or domain
  - current adapter: `arc-grid`

- exporters
  - domain-specific scripts that convert native logs into the generic session
    schema consumed by the viewer

## Data Model

The viewer consumes a `session` JSON object with:

- top-level session metadata
- a renderer hint
- a session-level entity bundle
- a step list
- per-step artifacts, action, commentary, transition summary, state, goals, and
  entity bundles

The formal schema is described in `SESSION_SCHEMA.md`.

## Why a separate shared folder

This tool is intentionally not embedded inside a single use case. ARC-AGI-3 is
the first adapter, but the tool is expected to become shared infrastructure
across multiple Knowledge Fabric use cases. Placing it in `tools/` keeps that
trajectory explicit.

## ARC-AGI-3 rendering choices

For ARC-AGI-3, the primary artifact is a 2D integer grid. The viewer:

- maps integer values to colors
- renders the current frame at a configurable zoom
- optionally overlays changed cells
- groups diffs into connected components to avoid misleading single global boxes
- displays state summaries such as `NOT_FINISHED` and level progress
- exposes explicit goal progress and interpreted state entities

## Non-goals for v1

- editing sessions in-browser
- multi-user collaboration
- live WebSocket streaming
- deep analytics dashboards
- use-case-specific workflow controls

These can be added later if needed.
