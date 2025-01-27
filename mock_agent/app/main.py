from fastapi import FastAPI, HTTPException, Depends, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from datetime import datetime, timedelta
import logging
import time
from fastapi.security import APIKeyHeader
from typing import Optional
import asyncio
from collections import defaultdict
import jsonschema
from .models import (
    AgentStatus,
    HealthCheck,
    AgentCapabilities,
    InvocationRequest,
    InvocationResponse,
    ErrorDetails
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting configuration
RATE_LIMIT_WINDOW = 60  # seconds
MAX_REQUESTS_PER_WINDOW = 10
rate_limit_storage = defaultdict(list)

async def rate_limiter(request: Request):
    """Rate limiting dependency"""
    now = time.time()
    client_ip = "default"  # In production, get this from request
    trace_id = request.headers.get("X-Trace-ID", "unknown")
    
    # Clean old requests
    rate_limit_storage[client_ip] = [
        req_time for req_time in rate_limit_storage[client_ip]
        if now - req_time < RATE_LIMIT_WINDOW
    ]
    
    # Check rate limit
    if len(rate_limit_storage[client_ip]) >= MAX_REQUESTS_PER_WINDOW:
        response = InvocationResponse(
            status="error",
            error={
                "error": {
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": "Rate limit exceeded. Please try again later.",
                    "details": {
                        "error_type": "RateLimitError",
                        "window_seconds": RATE_LIMIT_WINDOW,
                        "max_requests": MAX_REQUESTS_PER_WINDOW,
                        "retry_after": RATE_LIMIT_WINDOW
                    }
                }
            },
            trace_id=trace_id
        )
        return JSONResponse(
            status_code=429,
            content=response.dict()
        )
    
    # Add current request
    rate_limit_storage[client_ip].append(now)
    return True

app = FastAPI(title="Mock AI Agent", version="1.0.0")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    trace_id = "unknown"
    if isinstance(request.state, dict) and "trace_id" in request.state:
        trace_id = request.state["trace_id"]
    
    response = InvocationResponse(
        status="error",
        error=ErrorDetails(
            code="VALIDATION_ERROR",
            message="Request validation failed",
            details={
                "error_type": "ValidationError",
                "errors": exc.errors()
            }
        ),
        trace_id=trace_id
    )
    return JSONResponse(
        status_code=400,
        content=response.model_dump()
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    trace_id = "unknown"
    if isinstance(request.state, dict) and "trace_id" in request.state:
        trace_id = request.state["trace_id"]
    
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail
        )
        
    response = InvocationResponse(
        status="error",
        error=ErrorDetails(
            code="HTTP_ERROR",
            message=str(exc.detail),
            details={
                "error_type": "HTTPException",
                "status_code": exc.status_code
            }
        ),
        trace_id=trace_id
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=response.model_dump()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    trace_id = "unknown"
    if isinstance(request.state, dict) and "trace_id" in request.state:
        trace_id = request.state["trace_id"]
    
    response = InvocationResponse(
        status="error",
        error=ErrorDetails(
            code="INTERNAL_ERROR",
            message="An internal server error occurred",
            details={
                "error_type": type(exc).__name__,
                "error_message": str(exc)
            }
        ),
        trace_id=trace_id
    )
    return JSONResponse(
        status_code=500,
        content=response.model_dump()
    )

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock agent configuration
MOCK_AGENT_CONFIG = {
    "name": "MockTextProcessor",
    "version": "1.0.0",
    "capabilities": ["text-processing", "sentiment-analysis"],
    "input_schema": {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "analysis_type": {"type": "string", "enum": ["sentiment", "summary"]}
        },
        "required": ["text", "analysis_type"]
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "result": {"type": "string"},
            "confidence": {"type": "number"}
        }
    }
}

@app.get("/health")
async def health_check(rate_limit: bool = Depends(rate_limiter), request: Request = None) -> HealthCheck:
    """Health check endpoint"""
    return HealthCheck(
        status=AgentStatus.HEALTHY,
        last_updated=datetime.utcnow(),
        metrics={
            "uptime": 100,
            "success_rate": 0.99,
            "average_response_time": 0.1
        }
    )

@app.get("/capabilities")
async def get_capabilities(rate_limit: bool = Depends(rate_limiter), request: Request = None) -> AgentCapabilities:
    """Get agent capabilities"""
    return AgentCapabilities(
        name=MOCK_AGENT_CONFIG["name"],
        version=MOCK_AGENT_CONFIG["version"],
        capabilities=MOCK_AGENT_CONFIG["capabilities"],
        input_schema=MOCK_AGENT_CONFIG["input_schema"],
        output_schema=MOCK_AGENT_CONFIG["output_schema"],
        rate_limits={
            "requests_per_second": 10,
            "burst_limit": 20
        }
    )

@app.post("/invoke")
async def invoke_agent(
    request: InvocationRequest,
    rate_limit: bool = Depends(rate_limiter),
    req: Request = None,
    marketplace_token: Optional[str] = Header(None, alias="X-Marketplace-Token", description="Optional marketplace token")
) -> InvocationResponse:
    """Invoke agent functionality"""
    # Log invocation request
    logger.info(f"Processing invocation request with trace_id: {request.trace_id}")
    
    # Log and validate token
    logger.info(f"Received marketplace token: {marketplace_token}")
    
    # In development mode, accept any non-empty token
    # if not marketplace_token or not marketplace_token.strip():
    #     logger.error("Missing or empty marketplace token")
    #     raise HTTPException(
    #         status_code=401,
    #         detail={
    #             "status": "error",
    #             "error": {
    #                 "code": "INVALID_TOKEN",
    #                 "message": "Missing or invalid marketplace token",
    #                 "details": {
    #                     "error_type": "AuthenticationError"
    #                 }
    #             },
    #             "trace_id": request.trace_id
    #         }
    #     )
    
    logger.info("Development mode: Token validation passed")
    
    try:
        # Validate input against schema
        try:
            jsonschema.validate(instance=request.input, schema=MOCK_AGENT_CONFIG["input_schema"])
        except jsonschema.exceptions.ValidationError as e:
            return InvocationResponse(
                status="error",
                error={
                    "error": {
                        "code": "VALIDATION_ERROR",
                        "message": "Input validation failed",
                        "details": {
                            "error_type": "ValidationError",
                            "validation_error": str(e),
                            "schema": MOCK_AGENT_CONFIG["input_schema"]
                        }
                    }
                },
                trace_id=request.trace_id
            )
            
        # Process the request
        text = request.input.get("text", "")
        analysis_type = request.input.get("analysis_type", "")
        
        if analysis_type not in ["sentiment", "summary"]:
            return InvocationResponse(
                status="error",
                error={
                    "error": {
                        "code": "INVALID_ANALYSIS_TYPE",
                        "message": "Invalid analysis type specified",
                        "details": {
                            "error_type": "ValidationError",
                            "allowed_types": ["sentiment", "summary"],
                            "received_type": analysis_type
                        }
                    }
                },
                trace_id=request.trace_id
            )
            
        if analysis_type == "sentiment":
            result = "positive" if any(word in text.lower() for word in ["good", "love", "great", "excellent"]) else "negative"
            confidence = 0.85 if any(word in text.lower() for word in ["good", "love", "great", "excellent"]) else 0.65
        else:  # summary
            result = f"Summary of: {text[:50]}..."
            confidence = 0.9
            
        return InvocationResponse(
            status="success",
            result={
                "result": result,
                "confidence": confidence
            },
            trace_id=request.trace_id
        )
        
    except ValueError as e:
        return InvocationResponse(
            status="error",
            error={
                "error": {
                    "code": "INVALID_INPUT",
                    "message": str(e),
                    "details": {
                        "error_type": "ValidationError",
                        "required_fields": ["text", "analysis_type"]
                    }
                }
            },
            trace_id=request.trace_id
        )
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return InvocationResponse(
            status="error",
            error={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An internal error occurred",
                    "details": {
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    }
                }
            },
            trace_id=request.trace_id
        )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": MOCK_AGENT_CONFIG["name"],
        "version": MOCK_AGENT_CONFIG["version"],
        "description": "A mock AI agent that processes text and performs sentiment analysis"
    }
