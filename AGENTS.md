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
