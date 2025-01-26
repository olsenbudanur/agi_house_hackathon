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

app = FastAPI(title="Funding Manager Agent", version="1.0.0")

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
    "name": "FundingManager",
    "version": "1.0.0",
    "description": "Manages loan funding and settlement processes",
    "capabilities": ["fund-disbursement", "settlement-coordination", "wire-transfer", "post-funding-review"],
    "input_schema": {
        "type": "object",
        "properties": {
            "loan_id": {"type": "string"},
            "funding_amount": {"type": "number"},
            "funding_date": {"type": "string", "format": "date"},
            "disbursement_details": {
                "type": "object",
                "properties": {
                    "recipient_name": {"type": "string"},
                    "bank_name": {"type": "string"},
                    "account_number": {"type": "string"},
                    "routing_number": {"type": "string"},
                    "wire_instructions": {"type": "string"}
                },
                "required": ["recipient_name", "bank_name", "account_number", "routing_number"]
            },
            "settlement_agent": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "company": {"type": "string"},
                    "email": {"type": "string"},
                    "phone": {"type": "string"}
                },
                "required": ["name", "company", "email"]
            }
        },
        "required": ["loan_id", "funding_amount", "funding_date", "disbursement_details"]
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "funding_status": {
                "type": "string",
                "enum": ["pending", "in_progress", "completed", "failed"]
            },
            "transaction_details": {
                "type": "object",
                "properties": {
                    "transaction_id": {"type": "string"},
                    "confirmation_number": {"type": "string"},
                    "timestamp": {"type": "string"},
                    "status_description": {"type": "string"}
                }
            },
            "disbursement_confirmation": {
                "type": "object",
                "properties": {
                    "amount_disbursed": {"type": "number"},
                    "recipient_confirmation": {"type": "string"},
                    "wire_reference": {"type": "string"}
                }
            },
            "settlement_status": {
                "type": "string",
                "enum": ["pending", "confirmed", "completed"]
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
        funding_amount = request.input["funding_amount"]
        funding_date = request.input["funding_date"]
        disbursement_details = request.input["disbursement_details"]
        settlement_agent = request.input.get("settlement_agent", {})
        
        # Mock funding process logic
        # Generate mock transaction ID and confirmation number
        import hashlib
        transaction_id = hashlib.md5(f"{loan_id}-{funding_date}".encode()).hexdigest()[:12]
        confirmation_number = f"FUND-{transaction_id[:6]}"
        
        # Mock wire reference
        wire_reference = f"WIRE-{transaction_id[-6:]}"
        
        # Simulate funding status based on input validation
        has_valid_wire = all(
            disbursement_details.get(field)
            for field in ["account_number", "routing_number", "wire_instructions"]
        )
        
        has_settlement_agent = all(
            settlement_agent.get(field)
            for field in ["name", "company", "email"]
        )
        
        if not has_valid_wire:
            funding_status = "failed"
            settlement_status = "pending"
            status_description = "Invalid wire transfer details"
        elif not has_settlement_agent:
            funding_status = "pending"
            settlement_status = "pending"
            status_description = "Awaiting settlement agent confirmation"
        else:
            funding_status = "completed"
            settlement_status = "completed"
            status_description = "Funding successfully completed"
            
        return InvocationResponse(
            status="success",
            result={
                "funding_status": funding_status,
                "transaction_details": {
                    "transaction_id": transaction_id,
                    "confirmation_number": confirmation_number,
                    "timestamp": datetime.utcnow().isoformat(),
                    "status_description": status_description
                },
                "disbursement_confirmation": {
                    "amount_disbursed": funding_amount,
                    "recipient_confirmation": f"Sent to {disbursement_details['recipient_name']}",
                    "wire_reference": wire_reference
                },
                "settlement_status": settlement_status
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
