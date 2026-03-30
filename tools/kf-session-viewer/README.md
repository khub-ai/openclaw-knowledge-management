# KF Session Viewer

`kf-session-viewer` is a shared browser-based visualization tool for Knowledge
Fabric sessions. It is intended to work across many use cases, even when the
domain artifacts differ substantially.

The viewer is session-centric rather than benchmark-centric. It visualizes:

- step-by-step progression
- the current primary artifact
- actions taken
- commentary / notes
- evolving state snapshots
- Knowledge Fabric entities such as rules, tools, states, and goals
- transition overlays
- per-step metadata

The first supported adapter is `arc-grid`, which renders ARC-AGI-3 grid frames.

## Why this tool exists

ARC-AGI-3 required a browser-based playback and analysis GUI, but the same core
interaction model is likely useful across other Knowledge Fabric workflows:

- ARC-AGI interactive play
- expert-knowledge-transfer use cases
- future learning, critique, or review sessions

This tool therefore starts from ARC-AGI-3 but is designed as reusable KF
infrastructure rather than a one-off ARC replay page.

## Folder Layout

- `client/`
  Browser UI shell and renderer adapters.
- `samples/`
  Session JSON bundles used for testing or demos.
- `DESIGN.md`
  Design overview and architectural choices.
- `SESSION_SCHEMA.md`
  Generic session schema that use cases should export.
- `server.mjs`
  Minimal static server with sample listing.

## Run

From this folder:

```bash
npm start
```

Then open:

```text
http://localhost:4179
```

If you already have a session JSON file, you can also load it directly through
the browser file picker in the UI.

## ARC-AGI-3 sample

A sample ARC-AGI-3 session can be generated from a playlog with:

```bash
python C:\_backup\github\khub-knowledge-fabric\usecases\arc-agi\python\export_arc3_session.py ^
  --playlog-dir C:\_backup\github\khub-knowledge-fabric\tests\arc-agi-3\playlogs\20260329-084020 ^
  --output C:\_backup\github\khub-knowledge-fabric\tools\kf-session-viewer\samples\ls20-20260329-084020.session.json
```

## Current v1 scope

- browser-based timeline playback
- autoplay / stepping
- generic side panel for notes, metadata, state, and KF entities
- change-highlighting that fades over time for recently updated fields
- mouse-over detail bubbles via tooltips
- renderer adapter architecture
- ARC grid renderer with diff overlay
- local sample loading and file loading

## Next likely additions

- thumbnail timeline
- previous/current split view
- entity history over time
- artifact selector for multi-artifact steps
- more renderers for non-ARC use cases
