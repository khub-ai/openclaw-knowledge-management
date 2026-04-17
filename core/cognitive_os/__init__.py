"""core.cognitive_os — Cognitive OS namespace.

All COS work lives under this package, clearly separated from non-COS
KF framework code (``core.knowledge``, ``core.pipeline``,
``core.benchmark``, ``core.dialogic_distillation``).

Current contents:

* ``core.cognitive_os.engine``  — domain-agnostic symbolic reasoning
                                   substrate shared by sequential-
                                   reasoning tasks and robotics.
                                   See ``core/cognitive_os/engine/DESIGN.md``.

This top-level package intentionally exports nothing at module level —
import from the specific sub-package you need (currently
``core.cognitive_os.engine``).
"""
