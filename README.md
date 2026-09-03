# LunarAlign

Multi-modal, sun-angle & scale invariant image correspondence using
Chandrayaan-2 optical images (OHRC, TMC-2, IIRS).

## Getting Started for New Contributors

Welcome! Here is the quick setup workflow to get up and running:

1. **Python Virtual Environment**:
   - Create: `python -m venv .venv`
   - Activate (PowerShell): `.\.venv\Scripts\Activate.ps1` (or `source .venv/bin/activate` on bash)
   - Install dependencies: `pip install -r requirements.txt`
2. **Frontend Setup**:
   - `cd frontend && npm install`
3. **Run Locally**:
   - Backend: `python -m uvicorn backend.main:app --reload --port 8000` (API on http://localhost:8000)
   - Frontend: `npm run dev` inside `frontend/` (UI on http://localhost:5173)
4. **Architecture & Specifications**:
   - Full specifications: [`SIH26166_PROJECT_SPEC.md`](SIH26166_PROJECT_SPEC.md)
   - Deep-dive documentation: [`docs/`](docs/) (architecture, algorithms, evaluation)

### Prerequisites
- Python 3.10+ (tested with 3.11)
- Node.js 18+ (tested with 20.20.0)
- npm 10+

### Backend

```powershell
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# On Linux/macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the API server
python -m uvicorn backend.main:app --reload --port 8000
```

Verify: open http://localhost:8000/health

> **Note:** If `.venv` is not activated, prefix commands with
> `.\.venv\Scripts\python.exe -m` (e.g., `.\.venv\Scripts\python.exe -m uvicorn ...`).

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Verify: open http://localhost:5173

### Tests

```powershell
# With .venv activated:
python -m pytest tests/ -v

# Without activation:
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

## Project Structure

See `PROJECT_SPEC.md` §8 for the full directory layout.

## Documentation

- [Architecture](docs/architecture.md)
- [Algorithms](docs/algorithms.md)
- [Evaluation](docs/evaluation.md)
- [Project Specification](SIH26166_PROJECT_SPEC.md)
