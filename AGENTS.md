# Project conventions

- Read README.md and docs/REPRODUCIBILITY.md before changes.
- RGB is single-class segmentation: `sweepable_litter`. Intact regular cartons, robots, equipment, people, floor seams, stains and reflections are background.
- Predictions are pre-annotations. Only reviewed labels may become training truth.
- Keep RGB semantics separate from depth geometry. Never infer registration or depth units from image dimensions.
- Preserve existing datasets, model weights and outputs; use distinct experiment names.
- Use explicit paths for portable entry points. Historical experiment scripts may retain documented environment dependencies.
- Store release staging, archives, checksums and verification sidecars under `E:/gpt/_release` on the original Windows workstation.
- Use checks proportional to the change: review links and commands for documentation, and run relevant module tests for code changes. Do not default to full regression, smoke tests, training, or large-file hashing; record material changes to validation scope.
