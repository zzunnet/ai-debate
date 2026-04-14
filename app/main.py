import os
import subprocess
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.database import init_db
from app.routers import sessions
from app.config import MISSING_KEYS


def _get_version() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        date = subprocess.check_output(
            ["git", "log", "-1", "--format=%ci"], stderr=subprocess.DEVNULL
        ).decode().strip()[:16]  # "YYYY-MM-DD HH:MM"
    except Exception:
        commit = os.getenv("RENDER_GIT_COMMIT", "unknown")[:7]
        date = ""
    return {"commit": commit, "date": date}

VERSION = _get_version()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize DB
    await init_db()
    
    if MISSING_KEYS:
        print(f"⚠️  WARNING: Missing API keys: {', '.join(MISSING_KEYS)}")
        print("Please set them in .env file.")
        
    yield
    # Shutdown: Clean up resources if any

app = FastAPI(
    title="AI Debate Tool",
    description="Claude vs Gemini Real-time Debate",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(sessions.router)

# Static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return FileResponse("static/index.html")

# Health check
@app.get("/health")
async def health():
    return {"status": "ok", "missing_keys": MISSING_KEYS}


@app.get("/version")
async def version():
    return VERSION
