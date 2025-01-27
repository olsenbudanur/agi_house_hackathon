from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from json.decoder import JSONDecodeError
from typing import Dict, Any, Optional
import os
import logging
import datetime
from pydantic import BaseModel, Field, HttpUrl
import time
from collections import defaultdict
from typing import Dict, List, Optional, Any, Literal
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from PIL import Image
from io import BytesIO
from app.validation import validate_matrix_data, format_validation_response
from app.matrix_types import (
    MatrixResponse, ErrorResponse, ProcessingMethod, MatrixData,
    AgentStatus, HealthCheck, AgentCapabilities, InvocationRequest,
    InvocationResponse, ErrorDetails
)
from app.ocr_processor import process_matrix_with_ocr
import time
import httpx


class InvocationRequest(BaseModel):
    input: Dict[str, Any]
    callback_url: Optional[HttpUrl] = None
    timeout: Optional[int] = Field(default=30, ge=1, le=300)
    trace_id: str

class InvocationResponse(BaseModel):
    status: Literal["success", "error", "pending"]
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Dict[str, Any]]] = None
    trace_id: str

# Load environment variables
load_dotenv(find_dotenv())

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global start time for uptime calculation
START_TIME = time.time()

# Global success/failure counters for metrics
request_counts = {
    "success": 0,
    "total": 0,
    "response_times": []
}

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
        return JSONResponse(
            status_code=429,
            content={
                "status": "error",
                "error": {
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": "Rate limit exceeded. Please try again later.",
                    "details": {
                        "error_type": "RateLimitError",
                        "window_seconds": RATE_LIMIT_WINDOW,
                        "max_requests": MAX_REQUESTS_PER_WINDOW,
                        "retry_after": RATE_LIMIT_WINDOW
                    }
                },
                "trace_id": trace_id
            }
        )
    
    # Add current request
    rate_limit_storage[client_ip].append(now)
    return True

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())  # This will search parent directories for .env file

# Verify OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found in environment variables")

app = FastAPI()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Disable CORS. Do not remove this for full-stack development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    trace_id = request.headers.get("X-Trace-ID", "unknown")
    # Clean up validation errors to remove body-level errors
    errors = [err for err in exc.errors() if not (
        isinstance(err.get('loc'), tuple) and 
        len(err['loc']) > 0 and 
        err['loc'][0] == 'body'
    )]
    
    if not errors:
        # If all errors were body-level, this is likely a JSON parsing error
        errors = [{
            "type": "json_parse_error",
            "msg": "Invalid JSON format in request body"
        }]
    
    return JSONResponse(
        status_code=400,
        content={
            "status": "error",
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": {
                    "error_type": "ValidationError",
                    "errors": errors
                }
            },
            "trace_id": trace_id
        }
    )

@app.get("/")
async def root(rate_limit: bool = Depends(rate_limiter)):
    return {
        "message": "Insurance Matrix Processor API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/healthz",
            "process": "/api/process-matrix"
        }
    }

import os
import sys
import logging
import datetime
from typing import List
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Agent configuration
AGENT_CONFIG = {
    "name": "MatrixProcessorAgent",
    "version": "1.0.0",
    "capabilities": [
        "matrix-parsing",
        "guideline-extraction",
        "pdf-to-image-conversion",
        "ocr-processing"
    ],
    "input_schema": {
        "type": "object",
        "properties": {
            "file_url": {
                "type": "string",
                "format": "uri",
                "description": "URL of the matrix image or PDF file to process"
            }
        },
        "required": ["file_url"]
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "data": {
                "type": "object",
                "properties": {
                    "ltv_requirements": {"type": "object"},
                    "credit_requirements": {"type": "object"},
                    "income_requirements": {"type": "object"},
                    "property_requirements": {"type": "object"},
                    "additional_requirements": {"type": "object"}
                }
            },
            "validation": {
                "type": "object",
                "properties": {
                    "errors": {"type": "array"},
                    "warnings": {"type": "array"}
                }
            },
            "processing_info": {
                "type": "object",
                "properties": {
                    "confidence_scores": {"type": "object"}
                }
            }
        }
    }
}

@app.get("/health")
async def health_check(rate_limit: bool = Depends(rate_limiter)) -> HealthCheck:
    """Health check endpoint for the marketplace."""
    # Calculate metrics
    uptime = int(time.time() - START_TIME)
    success_rate = request_counts["success"] / request_counts["total"] if request_counts["total"] > 0 else 1.0
    avg_response_time = (
        sum(request_counts["response_times"]) / len(request_counts["response_times"])
        if request_counts["response_times"]
        else 0.0
    )

    try:
        # Test OpenAI API
        if not client.api_key:
            return HealthCheck(
                status=AgentStatus.UNHEALTHY,
                last_updated=datetime.datetime.utcnow(),
                metrics={
                    "uptime": uptime,
                    "success_rate": success_rate,
                    "average_response_time": avg_response_time
                }
            )
        
        # Test OpenAI connectivity
        client.models.list()
        
        # Test Tesseract
        import subprocess
        tesseract_cmd = None
        for path in ['/usr/local/bin/tesseract', '/usr/bin/tesseract', 'tesseract']:
            if path and (os.path.exists(path) or subprocess.run(['which', path], capture_output=True).returncode == 0):
                tesseract_cmd = path
                break
        
        if not tesseract_cmd:
            return HealthCheck(
                status=AgentStatus.DEGRADED,
                last_updated=datetime.datetime.utcnow(),
                metrics={
                    "uptime": uptime,
                    "success_rate": success_rate,
                    "average_response_time": avg_response_time
                }
            )
        
        # All checks passed
        return HealthCheck(
            status=AgentStatus.HEALTHY,
            last_updated=datetime.datetime.utcnow(),
            metrics={
                "uptime": uptime,
                "success_rate": success_rate,
                "average_response_time": avg_response_time
            }
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthCheck(
            status=AgentStatus.UNHEALTHY,
            last_updated=datetime.datetime.utcnow(),
            metrics={
                "uptime": uptime,
                "success_rate": success_rate,
                "average_response_time": avg_response_time
            }
        )

@app.get("/capabilities")
async def get_capabilities(rate_limit: bool = Depends(rate_limiter)) -> AgentCapabilities:
    """Get agent capabilities endpoint for the marketplace."""
    # Define agent configuration if not already defined
    agent_config = {
        "name": "MatrixProcessorAgent",
        "version": "1.0.0",
        "capabilities": [
            "matrix-parsing",
            "pdf-to-image-conversion"
        ],
        "input_schema": {
            "type": "object",
            "properties": {
                "file_url": {"type": "string", "format": "uri"}
            },
            "required": ["file_url"]
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "data": {"type": "object"},
                "validation": {"type": "object"}
            }
        }
    }
    
    return AgentCapabilities(
        name=agent_config["name"],
        version=agent_config["version"],
        capabilities=agent_config["capabilities"],
        input_schema=agent_config["input_schema"],
        output_schema=agent_config["output_schema"],
        rate_limits={
            "requests_per_second": 10,  # Fixed integer value as per spec
            "burst_limit": 20  # Fixed integer value as per spec
        }
    )

@app.post("/invoke")
async def invoke_agent(
    request: InvocationRequest,
    rate_limit: bool = Depends(rate_limiter),
    marketplace_token: Optional[str] = Header(None, alias="X-Marketplace-Token")
) -> InvocationResponse:
    """Invoke agent functionality endpoint for the marketplace."""
    logger.info(f"Processing invocation request with trace_id: {request.trace_id}")
    
    # Validate marketplace token
    if not marketplace_token or not marketplace_token.strip():
        return InvocationResponse(
            status="error",
            error={
                "error": {
                    "code": "INVALID_TOKEN",
                    "message": "Missing or invalid marketplace token",
                    "details": {
                        "error_type": "AuthenticationError"
                    }
                }
            },
            trace_id=request.trace_id
        )
    
    try:
        # Extract and validate file_url
        file_url = request.input.get("file_url")
        if not file_url:
            return InvocationResponse(
                status="error",
                error={
                    "error": {
                        "code": "INVALID_INPUT",
                        "message": "file_url is required in input",
                        "details": {
                            "error_type": "ValidationError",
                            "required_fields": ["file_url"]
                        }
                    }
                },
                trace_id=request.trace_id
            )
        
        try:
            # Download file from URL
            async with httpx.AsyncClient() as client:
                response = await client.get(file_url, timeout=request.timeout or 30)
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                contents = response.content
                
                # Process the matrix using our existing OCR logic
                matrix_data, validation_errors, validation_warnings = await process_matrix_with_ocr(contents, content_type)
                
                # Return successful response
                return InvocationResponse(
                    status="success",
                    result={
                        "data": matrix_data,
                        "validation": {
                            "errors": validation_errors,
                            "warnings": validation_warnings
                        },
                        "processing_info": {
                            "content_type": content_type,
                            "file_size": len(contents)
                        }
                    },
                    trace_id=request.trace_id
                )
                
        except httpx.TimeoutException:
            return InvocationResponse(
                status="error",
                error={
                    "error": {
                        "code": "TIMEOUT",
                        "message": "Request timed out while downloading file",
                        "details": {
                            "error_type": "TimeoutError",
                            "timeout_seconds": request.timeout or 30
                        }
                    }
                },
                trace_id=request.trace_id
            )
        except httpx.HTTPError as e:
            return InvocationResponse(
                status="error",
                error={
                    "error": {
                        "code": "DOWNLOAD_ERROR",
                        "message": f"Failed to download file: {str(e)}",
                        "details": {
                            "error_type": "HTTPError",
                            "status_code": e.response.status_code if hasattr(e, 'response') else None
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

@app.get("/healthz")
async def healthz(rate_limit: bool = Depends(rate_limiter)):
    try:
        # Test OpenAI client initialization
        if not client.api_key:
            return {"status": "error", "message": "OpenAI API key not configured"}
        
        # Basic model list request to verify API access
        client.models.list()
        
        # Check Tesseract installation
        import subprocess
        tesseract_status = {"installed": False, "version": None, "path": None}
        
        try:
            # Enhanced Tesseract verification with multiple paths
            logger.info("Starting Tesseract verification...")
            
            # Check environment variables
            env_vars = {
                'TESSDATA_PREFIX': os.getenv('TESSDATA_PREFIX'),
                'TESSERACT_CMD': os.getenv('TESSERACT_CMD'),
                'PATH': os.getenv('PATH'),
                'LC_ALL': os.getenv('LC_ALL')
            }
            logger.info(f"Environment variables: {env_vars}")
            
            # Try multiple Tesseract paths
            tesseract_paths = [
                os.getenv('TESSERACT_CMD'),  # First try env var
                '/usr/local/bin/tesseract',  # Then try local bin
                '/usr/bin/tesseract',        # Then try system bin
                'tesseract'                  # Finally try PATH
            ]
            
            tesseract_cmd = None
            for path in tesseract_paths:
                if path and (os.path.exists(path) or subprocess.run(['which', path], capture_output=True).returncode == 0):
                    tesseract_cmd = path
                    logger.info(f"Found Tesseract at: {path}")
                    break
            
            if not tesseract_cmd:
                logger.error("Tesseract not found in any standard location")
                return {"installed": False, "version": None, "path": None, "error": "Tesseract not found in any standard location"}
            
            try:
                # Check version
                version_output = subprocess.run(
                    [tesseract_cmd, '--version'],
                    capture_output=True,
                    text=True,
                    check=True
                )
                logger.info(f"Tesseract version: {version_output.stdout}")
                
                # Check languages
                langs_output = subprocess.run(
                    [tesseract_cmd, '--list-langs'],
                    capture_output=True,
                    text=True,
                    check=True
                )
                logger.info(f"Available languages: {langs_output.stdout}")
                
                # Check tessdata in multiple locations
                tessdata_paths = [
                    '/usr/local/share/tessdata',
                    '/usr/share/tesseract-ocr/tessdata',
                    os.getenv('TESSDATA_PREFIX', '')
                ]
                
                for tessdata_path in tessdata_paths:
                    if os.path.exists(tessdata_path):
                        tessdata_files = os.listdir(tessdata_path)
                        logger.info(f"Found tessdata files in {tessdata_path}: {tessdata_files}")
                        if 'eng.traineddata' in tessdata_files:
                            logger.info(f"Found eng.traineddata in {tessdata_path}")
                            return {"installed": True, "version": version_output.stdout, "path": tesseract_cmd}
                
                return {"installed": False, "version": None, "path": None, "error": "No valid tessdata directory found with eng.traineddata"}
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Tesseract verification failed: {str(e)}")
                return {"installed": False, "version": None, "path": None, "error": str(e)}
            
            if not tesseract_cmd or not os.path.isfile(tesseract_cmd):
                raise FileNotFoundError(f"Tesseract not found. Checked paths: {tesseract_cmd}")
            
            # Check tesseract version using found path
            result = subprocess.run([tesseract_cmd, '--version'], 
                                 capture_output=True, text=True, check=True)
            tesseract_status["installed"] = True
            tesseract_status["version"] = result.stdout.strip()
            tesseract_status["path"] = tesseract_cmd
            
            # Check available languages
            langs_result = subprocess.run([tesseract_cmd, '--list-langs'],
                                        capture_output=True, text=True, check=True)
            tesseract_status["available_languages"] = langs_result.stdout.strip().split('\n')[1:]
            
            # Check tessdata paths
            tessdata_prefix = os.getenv('TESSDATA_PREFIX', '')
            tessdata_paths = tessdata_prefix.split(':') if tessdata_prefix else []
            if not tessdata_paths:
                # Try common tessdata locations
                tessdata_paths = [
                    '/usr/share/tesseract-ocr/4.00/tessdata',
                    '/usr/share/tesseract-ocr/tessdata',
                    '/usr/local/share/tessdata'
                ]
            
            tesseract_status["tessdata_paths"] = []
            for path in tessdata_paths:
                if path and os.path.exists(path):
                    tesseract_status["tessdata_paths"].append({
                        "path": path,
                        "exists": True,
                        "has_eng": os.path.exists(os.path.join(path, 'eng.traineddata')),
                        "files": os.listdir(path) if os.path.exists(path) else []
                    })
            
            # Log detailed information
            logger.info(f"Tesseract Status: {tesseract_status}")
                    
        except Exception as te:
            logger.error(f"Tesseract verification failed: {str(te)}")
            tesseract_status["error"] = str(te)
            tesseract_status["installed"] = False
        
        return {
            "status": "ok",
            "openai_status": "connected",
            "api_version": "v1",
            "tesseract_status": tesseract_status,
            "environment": {
                "PATH": os.environ.get('PATH', ''),
                "TESSDATA_PREFIX": os.environ.get('TESSDATA_PREFIX', ''),
                "TESSERACT_CMD": os.environ.get('TESSERACT_CMD', ''),
                "PWD": os.getcwd(),
                "PYTHONPATH": os.environ.get('PYTHONPATH', '')
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/api/process-matrix", response_model=MatrixResponse)
async def process_matrix(
    file: UploadFile = File(...),
    rate_limit: bool = Depends(rate_limiter)
):
    """
    Process an uploaded matrix image using OCR and GPT-4 Vision analysis.
    
    Args:
        file (UploadFile): The uploaded image file containing the guideline matrix
        
    Returns:
        MatrixResponse: Structured data extracted from the matrix with validation results
    """
    start_time = time.time()
    request_counts["total"] += 1
    try:
        # Read the uploaded file
        contents = await file.read()
        
        logger.info(f"Processing file: {file.filename} (type: {file.content_type}, size: {len(contents)} bytes)")
        
        # Convert content type to lowercase for consistent comparison
        content_type = file.content_type.lower()
        
        # Allow both images and PDFs
        allowed_types = {'image/png', 'image/jpeg', 'image/jpg', 'image/webp', 'application/pdf'}
        
        if file.content_type not in allowed_types:
            logger.error(f"Rejected file type: {file.content_type}")
            return JSONResponse(
                status_code=400,
                content=ErrorResponse(
                    error="Unsupported File Type",
                    details={
                        "message": f"File type {file.content_type} is not supported",
                        "received_type": file.content_type,
                        "allowed_types": list(allowed_types),
                        "suggestion": "Please upload an image file (PNG or JPEG) containing the guideline matrix."
                    }
                ).dict()
            )
            
        # Validate file size (max 10MB)
        if len(contents) > 10 * 1024 * 1024:
            logger.warning(f"File too large: {len(contents)} bytes")
            return JSONResponse(
                status_code=400,
                content=ErrorResponse(
                    error="File Too Large",
                    details={
                        "message": "The uploaded file exceeds the maximum size limit of 10MB",
                        "suggestion": "Please compress the image or upload a smaller file"
                    }
                ).dict()
            )

        try:
            # Handle PDF conversion if necessary
            if content_type == 'application/pdf':
                logger.info("Converting PDF to image")
                import pdf2image
                images = pdf2image.convert_from_bytes(contents)
                # For MVP, just process the first page
                img = images[0]
                # Convert PIL Image to bytes
                img_byte_arr = BytesIO()
                img.save(img_byte_arr, format='PNG')
                contents = img_byte_arr.getvalue()
                content_type = 'image/png'
                logger.info("Successfully converted PDF to PNG image")
            
            # Process the matrix using OCR and GPT-4 Vision
            matrix_data, validation_errors, validation_warnings = await process_matrix_with_ocr(contents, content_type)
            
            # Format the response
            response_data = {
                "data": matrix_data,
                "validation": {
                    "errors": validation_errors,
                    "warnings": validation_warnings
                },
                "processing_info": {
                    "file_name": file.filename,
                    "file_type": file.content_type,
                    "file_size": len(contents),
                    "timestamp": datetime.datetime.now().isoformat()
                }
            }
            
            logger.info(f"Matrix processing completed successfully for {file.filename}")
            request_counts["success"] += 1
            request_counts["response_times"].append(time.time() - start_time)
            return JSONResponse(content=response_data)
            
        except Exception as processing_error:
            logger.error(f"Matrix processing failed: {str(processing_error)}")
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error="Processing Error",
                    details={
                        "message": "Failed to process the matrix image",
                        "error": str(processing_error),
                        "suggestion": "Please ensure the image is clear and contains a valid guideline matrix"
                    }
                ).dict()
            )
    except Exception as e:
        logger.error(f"Unexpected error processing matrix: {str(e)}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Server Error",
                details={
                    "message": "An unexpected error occurred while processing the request",
                    "error": str(e)
                }
            ).dict()
        )
