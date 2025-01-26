from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from datetime import datetime, timedelta
import logging
import time
from fastapi.security import APIKeyHeader
from typing import Optional, Dict, Any, List

# Authentication
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    """Verify API key"""
    # TODO: Implement proper API key verification
    if not api_key or api_key != "test-key":  # Replace with proper key verification
        raise HTTPException(
            status_code=401,
            detail={
                "status": "error",
                "error": {
                    "code": "UNAUTHORIZED",
                    "message": "Invalid API key",
                    "details": {
                        "error_type": "AuthenticationError"
                    }
                }
            }
        )
    return api_key
import asyncio
from collections import defaultdict
import jsonschema
import json
import aiohttp
import os
from models import (
    AgentStatus,
    HealthCheck,
    AgentCapabilities,
    InvocationRequest,
    InvocationResponse,
    ErrorDetails
)
from logic import (
    get_db_connection,
    embed,
    query as rag_query,
    chat as rag_chat
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting configuration
RATE_LIMIT_WINDOW = 60  # seconds
MAX_REQUESTS_PER_WINDOW = 10
rate_limit_storage = defaultdict(list)

# Environment variables (TODO: Move to secure configuration)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_HOST = os.getenv("DB_HOST")
DB_PASS = os.getenv("DB_PASS")
DB_USER = os.getenv("DB_USER")
DB_NAME = os.getenv("DB_NAME")

async def rate_limiter(request: Request):
    """Rate limiting dependency"""
    now = time.time()
    client_ip = request.client.host
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
            error=ErrorDetails(
                code="RATE_LIMIT_EXCEEDED",
                message="Rate limit exceeded. Please try again later.",
                details={
                    "error_type": "RateLimitError",
                    "window_seconds": RATE_LIMIT_WINDOW,
                    "max_requests": MAX_REQUESTS_PER_WINDOW,
                    "retry_after": RATE_LIMIT_WINDOW
                }
            ),
            trace_id=trace_id
        )
        return JSONResponse(
            status_code=429,
            content=response.dict()
        )
    
    # Add current request
    rate_limit_storage[client_ip].append(now)
    return True

app = FastAPI(title="Fannie RAG Agent", version="1.0.0")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    trace_id = request.headers.get("X-Trace-ID", "unknown")
    
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
    trace_id = request.headers.get("X-Trace-ID", "unknown")
    
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
    trace_id = request.headers.get("X-Trace-ID", "unknown")
    
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

# Agent configuration
@app.get("/health")
async def health_check(
    rate_limit: bool = Depends(rate_limiter),
    api_key: str = Depends(verify_api_key)
) -> HealthCheck:
    """Health check endpoint that matches JavaScript implementation pattern"""
    try:
        # Initialize metrics with default values
        metrics = {
            "uptime": 100.0,  # Mock value, should be calculated from start time
            "success_rate": 0.99,  # Mock value, should track actual success/failure
            "average_response_time": 0.2,  # Mock value, should be measured
            "total_queries": 0.0,  # Mock value, should be tracked
            "total_chats": 0.0,  # Mock value, should be tracked
            "database_connection_status": 0.0,
            "database_row_count": 0.0
        }

        # Test database connection using same pattern as JavaScript
        conn = None
        cursor = None
        try:
            # Simple connection test first, like JavaScript
            conn = await get_db_connection()
            cursor = conn.cursor()
            
            # Test basic connectivity first
            cursor.execute("SELECT 1")
            if cursor.fetchone():
                logger.info("Basic database connectivity test passed")
                
                # If basic test passes, check table
                cursor.execute("SELECT COUNT(*) as count FROM myvectortable")
                result = cursor.fetchone()
                metrics["database_row_count"] = float(result["count"]) if result and "count" in result else 0.0
                metrics["database_connection_status"] = 1.0
                logger.info("Database health check successful")
            
        except Exception as db_err:
            logger.error(f"Database health check failed: {str(db_err)}")
            # Keep default 0.0 status for database metrics
            
        finally:
            # Ensure connections are properly closed
            if cursor:
                try:
                    cursor.close()
                except Exception as e:
                    logger.error(f"Error closing cursor: {str(e)}")
            if conn:
                try:
                    conn.close()
                except Exception as e:
                    logger.error(f"Error closing connection: {str(e)}")

        # Determine status based on database connectivity
        status = AgentStatus.HEALTHY if metrics["database_connection_status"] == 1.0 else AgentStatus.DEGRADED
        
        return HealthCheck(
            status=status,
            last_updated=datetime.utcnow(),
            metrics=metrics
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthCheck(
            status=AgentStatus.UNHEALTHY,
            last_updated=datetime.utcnow(),
            metrics={
                "uptime": 0.0,
                "success_rate": 0.0,
                "average_response_time": 0.0,
                "total_queries": 0.0,
                "total_chats": 0.0,
                "database_connection_status": 0.0,
                "database_row_count": 0.0
            }
        )

@app.get("/capabilities")
async def get_capabilities(
    rate_limit: bool = Depends(rate_limiter),
    api_key: str = Depends(verify_api_key)
) -> AgentCapabilities:
    """Get agent capabilities"""
    return AgentCapabilities(
        name=RAG_AGENT_CONFIG["name"],
        version=RAG_AGENT_CONFIG["version"],
        capabilities=RAG_AGENT_CONFIG["capabilities"],
        input_schema=RAG_AGENT_CONFIG["input_schema"],
        output_schema=RAG_AGENT_CONFIG["output_schema"],
        rate_limits={
            "requests_per_second": MAX_REQUESTS_PER_WINDOW / RATE_LIMIT_WINDOW,
            "burst_limit": MAX_REQUESTS_PER_WINDOW
        }
    )

@app.post("/invoke")
async def invoke_agent(
    request: InvocationRequest,
    rate_limit: bool = Depends(rate_limiter),
    api_key: str = Depends(verify_api_key)
) -> InvocationResponse:
    """Invoke agent functionality"""
    logger.info(f"Processing invocation request with trace_id: {request.trace_id}")
    
    try:
        # Input validation is handled by Pydantic model
        input_dict = request.input
        if not isinstance(input_dict, dict):
            return InvocationResponse(
                status="error",
                error=ErrorDetails(
                    code="VALIDATION_ERROR",
                    message="Input must be a dictionary",
                    details={
                        "error_type": "ValidationError",
                        "received_type": str(type(input_dict))
                    }
                ),
                trace_id=request.trace_id
            )
        
        mode = request.input.get("mode")
        
        if mode == "chat":
            messages = request.input.get("messages", [])
            if not messages:
                raise ValueError("Messages array is required for chat mode")
                
            # Call the existing chat function from logic.py
            try:
                response = await rag_chat(messages)
                return InvocationResponse(
                    status="success",
                    result={
                        "answer": response,
                        "context": [],  # Context is embedded in the response for chat mode
                        "confidence": 0.9  # TODO: Implement real confidence scoring
                    },
                    trace_id=request.trace_id
                )
            except Exception as e:
                logger.error(f"Chat error: {str(e)}")
                return InvocationResponse(
                    status="error",
                    error=ErrorDetails(
                        code="CHAT_ERROR",
                        message="Failed to process chat request",
                        details={
                            "error_type": type(e).__name__,
                            "error_message": str(e)
                        }
                    ),
                    trace_id=request.trace_id
                )
                
        elif mode == "query":
            query_text = request.input.get("query")
            if not query_text:
                raise ValueError("Query text is required for query mode")
                
            # Call the existing query function from logic.py
            try:
                results = await rag_query(query_text)
                # Format raw query results into agent interface response
                if not results or len(results) == 0:
                    return InvocationResponse(
                        status="success",
                        result={
                            "answer": "No relevant information found",
                            "context": [],
                            "confidence": 0.0
                        },
                        trace_id=request.trace_id
                    )
                
                # Use the highest scoring result as the answer and remaining as context
                sorted_results = sorted(results, key=lambda x: float(x["score"]), reverse=True)
                return InvocationResponse(
                    status="success",
                    result={
                        "answer": sorted_results[0]["text"],
                        "context": [r["text"] for r in sorted_results[1:]] if len(sorted_results) > 1 else [],
                        "confidence": float(sorted_results[0]["score"])
                    },
                    trace_id=request.trace_id
                )
            except Exception as e:
                logger.error(f"Query error: {str(e)}")
                return InvocationResponse(
                    status="error",
                    error=ErrorDetails(
                        code="QUERY_ERROR",
                        message="Failed to process query request",
                        details={
                            "error_type": type(e).__name__,
                            "error_message": str(e)
                        }
                    ),
                    trace_id=request.trace_id
                )
                
        else:
            return InvocationResponse(
                status="error",
                error=ErrorDetails(
                    code="INVALID_MODE",
                    message=f"Invalid mode: {mode}",
                    details={
                        "error_type": "ValidationError",
                        "allowed_modes": ["chat", "query"],
                        "received_mode": mode
                    }
                ),
                trace_id=request.trace_id
            )
            
    except ValueError as e:
        return InvocationResponse(
            status="error",
            error=ErrorDetails(
                code="INVALID_INPUT",
                message=str(e),
                details={
                    "error_type": "ValidationError"
                }
            ),
            trace_id=request.trace_id
        )
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return InvocationResponse(
            status="error",
            error=ErrorDetails(
                code="INTERNAL_ERROR",
                message="An internal error occurred",
                details={
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            ),
            trace_id=request.trace_id
        )

@app.post("/marketplace/register")
async def register_agent(
    rate_limit: bool = Depends(rate_limiter),
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """Register agent with the marketplace"""
    try:
        registration_info = {
            "agent_id": "fannie-rag-agent",  # Unique identifier for the agent
            "name": RAG_AGENT_CONFIG["name"],
            "version": RAG_AGENT_CONFIG["version"],
            "capabilities": RAG_AGENT_CONFIG["capabilities"],
            "endpoints": {
                "health": "/health",
                "capabilities": "/capabilities",
                "invoke": "/invoke"
            },
            "provider": {
                "name": "Fannie RAG Team",
                "contact": "support@example.com"
            },
            "status": "active",
            "registration_time": datetime.utcnow().isoformat(),
            "last_heartbeat": datetime.utcnow().isoformat()
        }
        
        return {
            "status": "success",
            "message": "Agent registered successfully",
            "registration": registration_info
        }
        
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "error": {
                    "code": "REGISTRATION_FAILED",
                    "message": "Failed to register agent",
                    "details": {
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    }
                }
            }
        )

RAG_AGENT_CONFIG = {
    "name": "FannieRAGAgent",
    "version": "1.0.0",
    "capabilities": ["chat", "query"],
    "input_schema": {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": ["chat", "query"]},
            "messages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string", "enum": ["system", "user", "assistant"]},
                        "content": {"type": "string"}
                    },
                    "required": ["role", "content"]
                }
            },
            "query": {"type": "string"}
        },
        "required": ["mode"],
        "allOf": [
            {
                "if": {
                    "properties": {"mode": {"const": "chat"}},
                },
                "then": {
                    "required": ["messages"]
                }
            },
            {
                "if": {
                    "properties": {"mode": {"const": "query"}},
                },
                "then": {
                    "required": ["query"]
                }
            }
        ]
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "result": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "context": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number"}
                },
                "required": ["answer"]
            }
        }
    }
}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3001)
