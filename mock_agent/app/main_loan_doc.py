from fastapi import FastAPI, HTTPException, Depends, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from datetime import datetime
import logging
import time
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

app = FastAPI(title="Loan Document Processor Agent", version="1.0.0")

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
    "name": "LoanDocProcessor",
    "version": "1.0.0",
    "description": "Processes and verifies loan documentation and initial applications",
    "capabilities": ["document-verification", "application-review", "borrower-info-collection", "initial-processing"],
    "input_schema": {
        "type": "object",
        "properties": {
            "document_type": {
                "type": "string",
                "enum": ["application", "income", "employment", "assets", "credit"]
            },
            "document_content": {"type": "string"},
            "borrower_id": {"type": "string"},
            "loan_type": {
                "type": "string",
                "enum": ["conventional", "fha", "va", "jumbo"]
            }
        },
        "required": ["document_type", "document_content", "borrower_id"]
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "verification_status": {
                "type": "string",
                "enum": ["verified", "incomplete", "rejected"]
            },
            "missing_items": {
                "type": "array",
                "items": {"type": "string"}
            },
            "notes": {"type": "string"},
            "confidence_score": {"type": "number"}
        }
    }
}

@app.get("/health")
async def health_check(rate_limit: bool = Depends(rate_limiter)) -> HealthCheck:
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
async def get_capabilities(rate_limit: bool = Depends(rate_limiter)) -> AgentCapabilities:
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
    marketplace_token: Optional[str] = Header(None, alias="X-Marketplace-Token")
) -> InvocationResponse:
    """Invoke agent functionality"""
    logger.info(f"Processing invocation request with trace_id: {request.trace_id}")
    
    # Validate token
    if not marketplace_token or not marketplace_token.strip():
        raise HTTPException(
            status_code=401,
            detail={
                "status": "error",
                "error": {
                    "code": "INVALID_TOKEN",
                    "message": "Missing or invalid marketplace token",
                    "details": {
                        "error_type": "AuthenticationError"
                    }
                },
                "trace_id": request.trace_id
            }
        )
    
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
        document_type = request.input["document_type"]
        document_content = request.input["document_content"]
        borrower_id = request.input["borrower_id"]
        loan_type = request.input.get("loan_type", "conventional")
        
        # Mock document processing logic
        content_length = len(document_content)
        has_required_info = all(
            keyword in document_content.lower() 
            for keyword in ["name", "address", "income"]
        )
        
        if content_length < 50:  # Too short
            status = "incomplete"
            missing = ["detailed information", "supporting documentation"]
            notes = "Document content is insufficient"
            confidence = 0.3
        elif not has_required_info:
            status = "incomplete"
            missing = ["basic borrower information"]
            notes = "Missing required borrower information"
            confidence = 0.5
        else:
            status = "verified"
            missing = []
            notes = "All required information present"
            confidence = 0.9
            
        return InvocationResponse(
            status="success",
            result={
                "verification_status": status,
                "missing_items": missing,
                "notes": notes,
                "confidence_score": confidence
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
        "description": MOCK_AGENT_CONFIG["description"]
    }
