# Stage 2 Experiment Tracker

Quick reference for tracking which AI system created which experiment variant.

## Current Experiments

| Experiment ID | AI System | Model | Date | Status | Key Approach |
|---------------|-----------|-------|------|--------|--------------|
| `flux_talk_register_tokens` | Cursor | Composer | 2026-02-07 | Ready | FluxTalk register tokens + nested dropout |

## Quick Lookup

### flux_talk_register_tokens
- **Location**: `variants/flux_talk_register_tokens/`
- **Created by**: Cursor (Composer)
- **Key Features**: 
  - 4 register tokens (3 overlapping, 1 complementary)
  - Nested dropout with linear schedule
  - Per-modality token processing
  - Contrastive alignment on overlapping tokens

## Adding New Experiments

When adding a new experiment from a different AI system:

1. Add entry to table above
2. Create variant directory with metadata
3. Include header comments in all Python files
4. Create README.md with experiment details
5. Update this file

## Metadata Format

Each experiment should include:

```python
"""
Created by: [AI System Name]
Model: [Specific Model Name]  
Date: [YYYY-MM-DD]
Experiment ID: [unique_id]
Version: [version]
"""
```

## Notes

- All experiments use the same Stage 1 checkpoint
- Evaluation metrics should be comparable
- Document any hyperparameter differences
- Track which AI system created each variant for analysis
