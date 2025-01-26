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

app = FastAPI(title="Underwriting Analyzer Agent", version="1.0.0")

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
    "name": "UnderwritingAnalyzer",
    "version": "1.0.0",
    "description": "Analyzes loan applications and performs underwriting assessments",
    "capabilities": ["credit-verification", "loan-quality-check", "risk-assessment", "conditional-approval"],
    "input_schema": {
        "type": "object",
        "properties": {
            "borrower_id": {"type": "string"},
            "loan_amount": {"type": "number"},
            "loan_type": {
                "type": "string",
                "enum": ["conventional", "fha", "va", "jumbo"]
            },
            "credit_score": {"type": "integer"},
            "debt_to_income": {"type": "number"},
            "employment_history": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "employer": {"type": "string"},
                        "years": {"type": "number"}
                    }
                }
            },
            "assets": {
                "type": "object",
                "properties": {
                    "liquid": {"type": "number"},
                    "retirement": {"type": "number"},
                    "other": {"type": "number"}
                }
            }
        },
        "required": ["borrower_id", "loan_amount", "loan_type", "credit_score", "debt_to_income"]
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["approved", "conditionally_approved", "denied"]
            },
            "conditions": {
                "type": "array",
                "items": {"type": "string"}
            },
            "risk_factors": {
                "type": "array",
                "items": {"type": "string"}
            },
            "max_loan_amount": {"type": "number"},
            "notes": {"type": "string"}
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
        credit_score = request.input["credit_score"]
        debt_to_income = request.input["debt_to_income"]
        loan_amount = request.input["loan_amount"]
        loan_type = request.input["loan_type"]
        
        # Mock underwriting logic
        conditions = []
        risk_factors = []
        
        # Credit score analysis
        if credit_score < 620:
            decision = "denied"
            risk_factors.append("Credit score below minimum requirement")
        elif credit_score < 680:
            conditions.append("Additional credit history documentation required")
            risk_factors.append("Marginal credit score")
        
        # DTI analysis
        if debt_to_income > 43:
            decision = "denied"
            risk_factors.append("DTI ratio exceeds maximum allowance")
        elif debt_to_income > 36:
            conditions.append("Compensating factors required for DTI")
            risk_factors.append("High DTI ratio")
            
        # Loan amount analysis
        max_loan_amount = credit_score * 1000  # Simplified calculation
        if loan_amount > max_loan_amount:
            conditions.append(f"Loan amount exceeds calculated maximum of ${max_loan_amount}")
            risk_factors.append("High loan amount relative to qualifications")
            
        # Final decision logic
        if not risk_factors:
            decision = "approved"
            notes = "Application meets all standard requirements"
        elif len(risk_factors) == 1 and conditions:
            decision = "conditionally_approved"
            notes = "Application can proceed with conditions"
        elif not "denied" in locals():
            decision = "conditionally_approved"
            notes = "Multiple risk factors identified - strong compensating factors required"
            
        return InvocationResponse(
            status="success",
            result={
                "decision": decision,
                "conditions": conditions,
                "risk_factors": risk_factors,
                "max_loan_amount": max_loan_amount,
                "notes": notes
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
