from pydantic import BaseModel, Field, HttpUrl
from typing import List, Dict, Optional, Any, Literal
from enum import Enum
from datetime import datetime

class AgentStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class ProviderDetails(BaseModel):
    organization: str
    contact_email: str
    website: str

class AgentRegistration(BaseModel):
    agent_name: str
    description: str
    version: str
    capabilities: List[str]
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    health_check_endpoint: HttpUrl
    invocation_endpoint: HttpUrl
    provider_details: ProviderDetails

class AgentResponse(BaseModel):
    agent_id: str
    registration_timestamp: datetime
    marketplace_token: str

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
    description: str
    provider_details: ProviderDetails
    health_check_endpoint: HttpUrl
    invocation_endpoint: HttpUrl
    agent_id: Optional[str] = None
    registration_timestamp: Optional[datetime] = None
    embedding: Optional[List[float]] = None  # Store the pre-computed embedding

class InvocationRequest(BaseModel):
    input: Dict[str, Any]
    callback_url: Optional[HttpUrl] = None
    timeout: Optional[int] = Field(default=30, ge=1, le=300)
    trace_id: str

class InvocationResponse(BaseModel):
    status: Literal["success", "error", "pending"]
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Dict[str, Any]]] = Field(
        None,
        example={
            "code": "INVOCATION_ERROR",
            "message": "Error message",
            "details": {"error_type": "TypeError"}
        }
    )
    trace_id: str
