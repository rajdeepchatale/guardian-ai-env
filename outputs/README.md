# GuardianAI Training Outputs

This directory contains training artifacts from GRPO fine-tuning runs.

## Training Run: Qwen3-1.7B + GRPO (Final)

- **Date:** April 2026
- **GPU:** NVIDIA T4 (Kaggle)
- **Steps:** 30
- **Duration:** ~4.5 hours

### Results Summary

| Metric | Before | After | Change |
|---|---|---|---|
| Reward (mean) | 0.45 | 0.60 | +33% |
| Loss | 0.12 | 0.06 | -50% |
| Entropy | 0.15 | 0.13 | -13% |

### Files

- `training_log.jsonl` — Per-step reward and loss values
- See `assets/trackio_dashboard.png` for visual training curves
- Live dashboard: [Trackio Space](https://huggingface.co/spaces/rajdeepchatale/guardian-ai-grpo-Qwen3)
