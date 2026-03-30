# KF Session Schema

This viewer consumes a generic session JSON structure that can be exported from
different Knowledge Fabric workflows.

## Top-level shape

```json
{
  "schema_version": "0.2",
  "session_type": "arc-agi-3",
  "session_id": "ls20-20260329-084020",
  "title": "LS20 exploratory play session",
  "renderer": "arc-grid",
  "metadata": {
    "game_id": "ls20-9607627b"
  },
  "goals": [],
  "entities": {
    "rules": [],
    "tools": [],
    "states": [],
    "goals": []
  },
  "steps": []
}
```

## Required top-level fields

- `schema_version`
  Version of the exported viewer schema.
- `session_type`
  Domain or use-case identifier.
- `session_id`
  Stable session identifier.
- `title`
  Human-readable display title.
- `renderer`
  Default renderer adapter id.
- `steps`
  Ordered list of step objects.

## Recommended top-level fields

- `metadata`
  Session-level metadata.
- `goals`
  Stable session-level goals or objectives.
- `entities`
  Session-level Knowledge Fabric entities grouped by category.

## Step shape

```json
{
  "index": 1,
  "timestamp": "2026-03-29T08:40:21Z",
  "action": {
    "type": "ACTION1",
    "label": "ACTION1"
  },
  "commentary": {
    "note": "Start with ACTION1 to establish a first post-reset state..."
  },
  "artifacts": {
    "primary": {
      "kind": "grid-2d",
      "data": [[1, 2, 3]]
    }
  },
  "transition": {
    "from_step": 0,
    "diff": {
      "kind": "grid-cell-diff",
      "cells": [[34, 40, 3, 12]],
      "components": [
        { "x_min": 34, "x_max": 38, "y_min": 40, "y_max": 44 }
      ],
      "change_types": {
        "3->12": 10
      }
    }
  },
  "state": {
    "status": "NOT_FINISHED"
  },
  "goals": [],
  "entities": {
    "rules": [],
    "tools": [],
    "states": [],
    "goals": []
  },
  "metadata": {}
}
```

## Recommended step fields

- `index`
  Step number.
- `timestamp`
  Optional UTC timestamp or equivalent ordering string.
- `action`
  Action object with machine and human labels.
- `commentary.note`
  Human-readable note for why the step happened or what it means.
- `artifacts.primary`
  The main artifact to render for the step.
- `transition.diff`
  Optional transition summary from the previous step.
- `state`
  High-level state summary relevant to the use case.
- `goals`
  Goal status at this step. This may include active, blocked, completed, or
  newly discovered goals.
- `entities`
  Knowledge Fabric entities in canonical categories:
  - `rules`
  - `tools`
  - `states`
  - `goals`
- `metadata`
  Domain-specific details that do not belong in the generic state block.

## Entity item shape

Entity arrays may contain either strings or richer objects. Recommended object
shape:

```json
{
  "id": "goal-complete-levels",
  "label": "Complete all required levels",
  "status": "active",
  "description": "Primary objective for this ARC-AGI-3 game",
  "progress": {
    "value": 0,
    "total": 7
  },
  "tags": ["objective"]
}
```

Useful optional fields include:

- `id`
- `label`
- `status`
- `description`
- `progress`
- `tags`

## Viewer expectations

The viewer shell uses:

- the top-level `renderer` as a default adapter
- `artifacts.primary.kind` to confirm compatibility
- `commentary.note` for side commentary
- `transition.diff` for overlay rendering when available
- `state` for evolving state display
- `goals` for explicit progress / objective visualization
- `entities` for renderer-independent Knowledge Fabric entity panels
