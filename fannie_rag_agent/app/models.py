from pydantic import BaseModel, Field, HttpUrl
from typing import Dict, List, Optional, Any, Literal
from enum import Enum
from datetime import datetime

class AgentStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class HealthCheck(BaseModel):
    status: AgentStatus
    last_updated: datetime
    metrics: Dict[str, float]

class AgentCapabilities(BaseModel):
    name: str
    version: str
    capabilities: List[str]
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    rate_limits: Dict[str, int]

class InvocationRequest(BaseModel):
    input: Dict[str, Any]
    callback_url: Optional[HttpUrl] = None
    timeout: Optional[int] = Field(default=30, ge=1, le=300)
    trace_id: str

class ErrorDetails(BaseModel):
    code: str
    message: str
    details: Dict[str, Any]

class InvocationResponse(BaseModel):
    status: Literal["success", "error", "pending"]
    result: Optional[Dict[str, Any]] = None
    error: Optional[ErrorDetails] = None
    trace_id: str
