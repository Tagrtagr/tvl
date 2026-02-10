# Stage 2: Cross-Modality Alignment

Goal:
- Align overlapping information across modalities
- Preserve complementary information using register tokens

Scope:
- Frozen Stage-1 encoders
- Learnable register tokens
- Contrastive alignment (initial)
- No downstream task

Owner: Taarush

## Approaches

| Directory | Method | Status |
|-----------|--------|--------|
| `claude_flextok/` | FlexTok register tokens + nested dropout + contrastive/flow matching (Claude Code) | Implementation complete |
