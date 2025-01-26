from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import agents
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(
    title="AI Agent Marketplace",
    version="1.0.0",
    root_path="/api/v1",  # Add root path prefix for proxy
    openapi_url="/api/v1/openapi.json",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc"
)

# Configure CORS for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ai-agent-marketplace-9z6l1gh1.devinapps.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Include routers
app.include_router(agents.router)

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {
        "name": "AI Agent Marketplace",
        "version": "1.0.0",
        "description": "A marketplace for self-hosted AI agents"
    }
