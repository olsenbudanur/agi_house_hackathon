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

app = FastAPI(title="Quality Controller Agent", version="1.0.0")

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
    "name": "QualityController",
    "version": "1.0.0",
    "description": "Performs quality control reviews and compliance checks",
    "capabilities": ["post-closing-review", "compliance-check", "qc-audit", "investor-requirements"],
    "input_schema": {
        "type": "object",
        "properties": {
            "loan_id": {"type": "string"},
            "review_type": {
                "type": "string",
                "enum": ["pre_funding", "post_closing", "investor_audit"]
            },
            "loan_data": {
                "type": "object",
                "properties": {
                    "loan_amount": {"type": "number"},
                    "loan_type": {"type": "string"},
                    "interest_rate": {"type": "number"},
                    "term": {"type": "integer"},
                    "property_value": {"type": "number"}
                }
            },
            "document_checklist": {
                "type": "object",
                "properties": {
                    "application": {"type": "boolean"},
                    "income_verification": {"type": "boolean"},
                    "asset_verification": {"type": "boolean"},
                    "appraisal": {"type": "boolean"},
                    "title_report": {"type": "boolean"},
                    "closing_disclosure": {"type": "boolean"}
                }
            }
        },
        "required": ["loan_id", "review_type", "loan_data"]
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "review_status": {
                "type": "string",
                "enum": ["passed", "failed", "pending_corrections"]
            },
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "severity": {
                            "type": "string",
                            "enum": ["critical", "major", "minor"]
                        },
                        "category": {"type": "string"},
                        "description": {"type": "string"},
                        "remediation": {"type": "string"}
                    }
                }
            },
            "compliance_status": {
                "type": "object",
                "properties": {
                    "regulatory_compliance": {"type": "boolean"},
                    "investor_guidelines": {"type": "boolean"},
                    "internal_policies": {"type": "boolean"}
                }
            },
            "review_summary": {"type": "string"}
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
        review_type = request.input["review_type"]
        loan_data = request.input["loan_data"]
        doc_checklist = request.input.get("document_checklist", {})
        
        # Mock QC review logic
        findings = []
        
        # Document completeness check
        for doc_type, present in doc_checklist.items():
            if not present:
                findings.append({
                    "severity": "major",
                    "category": "Documentation",
                    "description": f"Missing {doc_type.replace('_', ' ')}",
                    "remediation": f"Obtain and upload {doc_type.replace('_', ' ')}"
                })
                
        # Loan data validation
        if loan_data["loan_amount"] > loan_data["property_value"]:
            findings.append({
                "severity": "critical",
                "category": "Loan-to-Value",
                "description": "Loan amount exceeds property value",
                "remediation": "Review and correct loan amount or property value"
            })
            
        if loan_data["interest_rate"] < 2.0 or loan_data["interest_rate"] > 18.0:
            findings.append({
                "severity": "major",
                "category": "Interest Rate",
                "description": "Interest rate outside normal range",
                "remediation": "Verify interest rate and provide justification"
            })
            
        # Determine review status
        if any(finding["severity"] == "critical" for finding in findings):
            review_status = "failed"
        elif any(finding["severity"] == "major" for finding in findings):
            review_status = "pending_corrections"
        else:
            review_status = "passed"
            
        # Mock compliance check
        compliance_status = {
            "regulatory_compliance": len(findings) == 0,
            "investor_guidelines": len(findings) <= 1,
            "internal_policies": review_status != "failed"
        }
            
        return InvocationResponse(
            status="success",
            result={
                "review_status": review_status,
                "findings": findings,
                "compliance_status": compliance_status,
                "review_summary": f"QC review completed for loan {loan_id}. Status: {review_status}"
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
