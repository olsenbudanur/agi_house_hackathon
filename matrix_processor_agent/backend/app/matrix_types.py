from typing import Dict, List, Optional, Union, Any, Literal
from pydantic import BaseModel, Field, HttpUrl
from enum import Enum
from datetime import datetime

class AgentStatus(str, Enum):
    """Enum for agent health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class HealthCheck(BaseModel):
    """Model for agent health check response."""
    status: AgentStatus
    last_updated: datetime
    metrics: Dict[str, float]

class AgentCapabilities(BaseModel):
    """Model for agent capabilities response."""
    name: str
    version: str
    capabilities: List[str]
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    rate_limits: Dict[str, int]

class InvocationRequest(BaseModel):
    """Model for agent invocation request."""
    input: Dict[str, Any]
    callback_url: Optional[HttpUrl] = None
    timeout: Optional[int] = Field(default=30, ge=1, le=300)
    trace_id: str

class ErrorDetails(BaseModel):
    """Model for error details in responses."""
    code: str
    message: str
    details: Dict[str, Any]

class InvocationResponse(BaseModel):
    """Model for agent invocation response."""
    status: Literal["success", "error", "pending"]
    result: Optional[Dict[str, Any]] = None
    error: Optional[ErrorDetails] = None
    trace_id: str

class HeadingData(BaseModel):
    """Model for hierarchical heading data using 'heading ^ subheading' notation."""
    heading: str
    subheading: Optional[str] = None
    
    def __str__(self) -> str:
        if self.subheading:
            return f"{self.heading} ^ {self.subheading}"
        return self.heading

class SpanningData(BaseModel):
    """Model for data that can span multiple rows with (1), (2), etc. notation."""
    value: str
    span_index: Optional[int] = None
    span_total: Optional[int] = None
    heading: Optional[HeadingData] = None  # Associated heading data if any

class LoanRequirements(BaseModel):
    max_ltv: str = Field(description="Maximum Loan-to-Value ratio as percentage")
    min_fico: int = Field(description="Minimum FICO score required")
    max_loan: SpanningData = Field(description="Maximum loan amount with potential row spanning")
    heading: Optional[HeadingData] = Field(None, description="Hierarchical heading data")

class PropertyTypeRequirements(BaseModel):
    purchase: LoanRequirements
    rate_and_term: LoanRequirements
    cash_out: LoanRequirements

class LTVRequirements(BaseModel):
    primary_residence: PropertyTypeRequirements
    second_home: PropertyTypeRequirements
    investment: PropertyTypeRequirements

class CreditEvents(BaseModel):
    """Model for credit event requirements."""
    bankruptcy: Optional[str] = None
    foreclosure: Optional[str] = None
    short_sale: Optional[str] = None

class CreditRequirements(BaseModel):
    """Model for credit-related requirements."""
    minimum_fico: Optional[int] = None
    maximum_dti: Optional[float] = None
    credit_events: Dict[str, Optional[str]] = Field(default_factory=dict)

class ReserveRequirements(BaseModel):
    """Model for reserve requirements."""
    months_pitia: Optional[int] = None
    description: Optional[str] = None
    conditions: List[str] = Field(default_factory=list)

class RequirementsData(BaseModel):
    """Model for all requirements sections."""
    max_dti: Optional[float] = None
    credit_requirements: CreditRequirements = Field(default_factory=CreditRequirements)
    reserve_requirements: Optional[str] = None
    geographic_restrictions: List[str] = Field(default_factory=list)

class SelfEmployed(BaseModel):
    requirements: list[str]
    restrictions: list[str]

class IncomeDocumentation(BaseModel):
    required_documents: list[str]
    self_employed: SelfEmployed

class PropertyRequirements(BaseModel):
    eligible_types: list[str]
    ineligible_types: Optional[list[str]] = Field(default_factory=list)
    geographic_restrictions: Optional[list[str]] = Field(default_factory=list)
    restrictions: Optional[list[str]] = Field(default_factory=list)

class AdditionalRequirements(BaseModel):
    reserves: str
    appraisal: str
    mortgage_insurance: str

class MatrixData(BaseModel):
    """Model for complete matrix data."""
    program_name: str
    effective_date: str
    processing_methods: List[str]
    confidence_scores: Dict[str, str]
    ltv_requirements: LTVRequirements
    requirements: RequirementsData = Field(default_factory=RequirementsData)
    property_requirements: PropertyRequirements
    processing_metadata: Dict[str, Union[str, int, float, bool]]
    gpt4_structured_analysis: Optional[Dict[str, Any]] = None
    income_documentation: Optional[IncomeDocumentation] = None
    additional_requirements: Optional[AdditionalRequirements] = None

    class Config:
        extra = "allow"  # Allow additional fields for flexibility

class ValidationResult(BaseModel):
    is_valid: bool
    errors: list[str]
    warnings: list[str] = Field(default_factory=list)

class ProcessingMethod(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Union[str, int, float, bool]] = Field(default_factory=dict)

class MatrixResponse(BaseModel):
    data: MatrixData
    validation: ValidationResult
    processing_method: ProcessingMethod

class ErrorResponse(BaseModel):
    error: str
    details: Optional[Dict] = None
