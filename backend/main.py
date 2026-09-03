"""
SIH26166 — FastAPI Backend Entry Point

Minimal scaffold for the multi-modal lunar image registration system.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.upload import router as upload_router
from backend.api.register import router as register_router

app = FastAPI(
    title="SIH26166 — Lunar Image Registration API",
    description="Multi-modal, sun-angle & scale invariant image correspondence "
                "using Chandrayaan-2 optical images (OHRC, TMC-2, IIRS).",
    version="0.1.0",
)

# Allow the React dev server (default port 5173) to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routers ---
app.include_router(upload_router)
app.include_router(register_router)


@app.get("/health")
async def health_check():
    """Health endpoint — confirms the API is running."""
    return {
        "status": "ok",
        "service": "sih26166-backend",
        "version": "0.1.0",
    }
