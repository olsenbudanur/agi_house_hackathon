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

app = FastAPI(title="Closing Coordinator Agent", version="1.0.0")

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
    "name": "ClosingCoordinator",
    "version": "1.0.0",
    "description": "Coordinates closing process and document preparation",
    "capabilities": ["closing-doc-prep", "closing-scheduling", "clear-to-close", "final-approval"],
    "input_schema": {
        "type": "object",
        "properties": {
            "loan_id": {"type": "string"},
            "closing_date": {"type": "string", "format": "date"},
            "closing_location": {"type": "string"},
            "participants": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {
                            "type": "string",
                            "enum": ["borrower", "seller", "title_agent", "attorney", "loan_officer"]
                        },
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                        "phone": {"type": "string"}
                    },
                    "required": ["role", "name", "email"]
                }
            },
            "document_package": {
                "type": "object",
                "properties": {
                    "closing_disclosure": {"type": "boolean"},
                    "note": {"type": "boolean"},
                    "deed": {"type": "boolean"},
                    "title_insurance": {"type": "boolean"}
                }
            }
        },
        "required": ["loan_id", "closing_date", "participants"]
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["scheduled", "documents_pending", "clear_to_close", "completed"]
            },
            "scheduled_time": {"type": "string"},
            "missing_documents": {
                "type": "array",
                "items": {"type": "string"}
            },
            "action_items": {
                "type": "array",
                "items": {"type": "string"}
            },
            "closing_confirmation": {
                "type": "object",
                "properties": {
                    "confirmation_id": {"type": "string"},
                    "location_details": {"type": "string"},
                    "participant_confirmations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string"},
                                "confirmed": {"type": "boolean"}
                            }
                        }
                    }
                }
            }
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
        loan_id = request.input["loan_id"]
        closing_date = request.input["closing_date"]
        participants = request.input["participants"]
        doc_package = request.input.get("document_package", {})
        
        # Mock closing coordination logic
        missing_docs = []
        action_items = []
        
        # Check required documents
        if not doc_package.get("closing_disclosure"):
            missing_docs.append("Closing Disclosure")
            action_items.append("Generate and review Closing Disclosure")
        if not doc_package.get("note"):
            missing_docs.append("Promissory Note")
            action_items.append("Prepare Promissory Note")
        if not doc_package.get("deed"):
            missing_docs.append("Deed")
            action_items.append("Prepare Deed")
        if not doc_package.get("title_insurance"):
            missing_docs.append("Title Insurance")
            action_items.append("Order Title Insurance")
            
        # Check participant confirmations
        participant_confirmations = []
        for participant in participants:
            # Mock confirmation logic based on email domain
            confirmed = "confirmed" in participant["email"].lower()
            participant_confirmations.append({
                "role": participant["role"],
                "confirmed": confirmed
            })
            if not confirmed:
                action_items.append(f"Follow up with {participant['role']} for confirmation")
                
        # Determine closing status
        if missing_docs:
            status = "documents_pending"
        elif not all(conf["confirmed"] for conf in participant_confirmations):
            status = "scheduled"
        elif action_items:
            status = "scheduled"
        else:
            status = "clear_to_close"
            
        return InvocationResponse(
            status="success",
            result={
                "status": status,
                "scheduled_time": f"{closing_date}T14:00:00Z",  # Default to 2 PM
                "missing_documents": missing_docs,
                "action_items": action_items,
                "closing_confirmation": {
                    "confirmation_id": f"CLOSE-{loan_id[:8]}",
                    "location_details": request.input.get("closing_location", "TBD"),
                    "participant_confirmations": participant_confirmations
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
        "description": MOCK_AGENT_CONFIG["description"]
    }
