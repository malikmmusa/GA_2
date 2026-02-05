# Skill: Reviewer (The QA)
**Trigger:** When asked to "test", "verify", or "debug".

## Verification Protocol
1. **The "Dummy Batch" Test:**
   - Before accepting any new model code, write a script that passes a random tensor (e.g., `torch.randn(1, 1, 512, 512)`) through the network.
   - Verify output shape matches target shape.
   - Verify no `NaN` gradients.

2. **Medical Logic Check:**
   - Are augmentation transforms biologically valid? (e.g., Do not flip OCTs vertically; retinal layers are not symmetric up/down).

3. **Execution:**
   - Run `pytest`.
   - If tests fail, output the error log and request a fix from @Coder.
   - If tests pass, mark the task `[x]` in `TODO.md`.