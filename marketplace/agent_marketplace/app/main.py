from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .routes import agents
import logging
import uuid

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Agent Marketplace",
    version="1.0.0",
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Changed to False since we're using token auth
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Convert HTTPException to our standard error response format"""
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail if isinstance(exc.detail, dict) else {
            "status": "error",
            "error": {
                "code": "REQUEST_ERROR",
                "message": str(exc.detail),
                "details": {
                    "error_type": "HTTPException",
                    "status_code": exc.status_code
                }
            },
            "trace_id": str(uuid.uuid4())
        }
    )



# Include routers
app.include_router(agents.router, prefix="/api/v1/agents", tags=["agents"])
logger.info(f"Mounted agents router at /api/v1/agents with routes: {[route.path for route in agents.router.routes]}")

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
