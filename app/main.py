import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.database import init_db
from app.routers import sessions
from app.config import MISSING_KEYS

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
