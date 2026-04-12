# AGENTS.md

## Cursor Cloud specific instructions

### Overview

Atrophy Advisor is a two-service app: Python/FastAPI backend (port 8000) + React/Vite frontend (port 3000). No database. The frontend proxies `/api` requests to the backend via Vite config.

### Running Services

- **Backend**: `cd /workspace && source venv/bin/activate && PYTHONPATH="$(pwd)" python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000`
- **Frontend**: `cd /workspace/src/frontend && npx vite --port 3000 --host 0.0.0.0`

### Testing

- **Backend tests**: `cd /workspace && source venv/bin/activate && PYTHONPATH="$(pwd)" pytest tests/ -v`
- **Frontend build check**: `cd /workspace/src/frontend && npx tsc --noEmit && npx vite build`
- Standard commands are documented in `README.md` under "Development".

### Known Caveats

- **ESLint config missing**: The repo does not include an `.eslintrc.*` file, so `npm run lint` fails. The `tsc --noEmit` + `vite build` pipeline works as an alternative check.
- **SAM-2 pip install**: `SAM-2>=1.0` in `requirements.txt` cannot be resolved from PyPI. It must be installed from GitHub first: `pip install "git+https://github.com/facebookresearch/sam2.git"`. The update script handles this.
- **Model weights not in repo**: `weights/best_disc_model.pth` and `weights/sam2.1_hiera_tiny.pt` are gitignored. Disc detection and SAM-based GA refinement endpoints will return errors without them, but the rest of the API and all tests work fine.
- **`pandas` needed for tests**: `test_disc_vs_ground_truth.py` imports `pandas`, which is not listed in `requirements.txt`. The update script installs it.

### Production Deployment (Railway)

- **Disc model weights**: The disc detector auto-downloads weights on first startup from Hugging Face env vars. Without them, disc detection falls back to a geometric heuristic (center of en-face, 30-70% height) which is completely wrong for real scans.
 - Set `DISC_MODEL_URL_V2=https://huggingface.co/malikmmusa/Atrophy_Advisor/resolve/main/best_disc_model_v2.pth` — v2 is preferred and loaded automatically when present.
 - Set `DISC_MODEL_URL=https://huggingface.co/malikmmusa/Atrophy_Advisor/resolve/main/best_disc_model.pth` — v1 fallback if v2 is unavailable.
 - **Auto-update**: On each startup the service compares local file size against the remote (HEAD request). If the remote file changed, it re-downloads automatically — no need to clear the volume.
 - Set `DISC_MODEL_FORCE_REDOWNLOAD=1` to force re-download regardless of size match (useful after uploading a same-size checkpoint). Remove the var after one successful deploy.
 - Add a Railway volume mounted at `/app/weights` so the files persist across restarts.
 - Verify the model loaded by checking `GET /api/disc-detector/status`.
