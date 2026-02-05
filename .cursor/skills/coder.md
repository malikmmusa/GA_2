# Skill: Coder (The Engineer)
**Trigger:** When asked to "implement", "write", or "fix" code.

## PyTorch Rigor
- **Type Hinting:** All functions must have Python type hints and Tensor shape comments.
  - Example: `def forward(self, x: torch.Tensor) -> torch.Tensor: # [B, 3, 224, 224]`
- **Device Management:** Never assume CPU or CUDA. Use `device = x.device`.
- **Silent Failures:** - Assert tensor shapes at critical junctions (skip connections, concatenation).
  - Use `einops` for complex reshapes to avoid `.view()` errors.

## Implementation Rules
1. Read the specific task in `TODO.md`.
2. Implement *only* that task.
3. Do not assume data loaders exist unless you see them.