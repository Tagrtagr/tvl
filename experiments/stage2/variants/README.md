# Stage 2 Experiment Variants

This directory contains different experimental approaches for Stage 2 cross-modality alignment.

## Tracking Experiments

Each variant includes metadata identifying:
- **Created by**: Which AI system/model created the implementation
- **Date**: When the variant was created
- **Experiment ID**: Unique identifier for this variant
- **Version**: Implementation version number

This helps track which experiments came from which AI system (Claude, OpenAI, Cursor, etc.)

## Current Variants

### 1. `flux_talk_register_tokens`
- **Created by**: Cursor (Composer)
- **Model**: Composer (Cursor's AI coding assistant)
- **Date**: February 7, 2026
- **Approach**: FluxTalk-style register tokens with nested dropout
- **Key Features**: 
  - 4 register tokens (3 overlapping + 1 complementary)
  - Cross-attention transformer layers
  - Nested dropout for hierarchical learning
  - Contrastive alignment on overlapping tokens

## Adding New Variants

To add a new experimental variant:

1. Create a new directory: `variants/your_variant_name/`
2. Copy or create your implementation files
3. **Add metadata header** to each Python file:
   ```python
   """
   Created by: [AI System Name]
   Model: [Specific model name]
   Date: [Date]
   Experiment ID: [unique_id]
   Version: [version]
   """
   ```
4. Create a `README.md` with:
   - Experiment metadata section
   - Approach description
   - Design decisions
   - Usage instructions
5. Update this README with your variant details

## Template for New Variants

When creating a new variant, use this template:

```markdown
# [Variant Name] - Stage 2 Variant

## Experiment Metadata

**Created by**: [AI System]  
**Model**: [Specific Model Name]  
**Date**: [Date]  
**Experiment ID**: `[unique_id]`  
**Implementation Version**: 1.0

## Approach

[Describe your approach]

## Design Decisions

[Key design choices and rationale]

## Usage

[How to run]
```

## Running Experiments

Each variant has its own training script. Navigate to the variant directory and run:

```bash
cd variants/your_variant_name
python main_train_stage2.py [args...]
```

## Comparison

When comparing variants, ensure:
- Same Stage 1 checkpoint
- Same dataset splits
- Same evaluation metrics
- Document hyperparameter differences
- Note which AI system created each variant

## Experiment Log

| Variant | Created By | Date | Status | Notes |
|---------|------------|------|--------|-------|
| flux_talk_register_tokens | Cursor (Composer) | 2026-02-07 | Ready | Base FluxTalk implementation |
