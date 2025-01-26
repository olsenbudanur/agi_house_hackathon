from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field

class LoanRequirements(BaseModel):
    max_ltv: str = Field(description="Maximum Loan-to-Value ratio as percentage")
    min_fico: int = Field(description="Minimum FICO score required")
    max_loan: float = Field(description="Maximum loan amount")

class PropertyTypeRequirements(BaseModel):
    purchase: LoanRequirements
    rate_and_term: LoanRequirements
    cash_out: LoanRequirements

class LTVRequirements(BaseModel):
    primary_residence: PropertyTypeRequirements
    second_home: PropertyTypeRequirements
    investment: PropertyTypeRequirements

class CreditEvents(BaseModel):
    bankruptcy: str
    foreclosure: str
    short_sale: str

class CreditRequirements(BaseModel):
    minimum_fico: int
    maximum_dti: float
    credit_events: CreditEvents

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
    program_name: str
    effective_date: str
    processing_methods: List[str]
    confidence_scores: Dict[str, str]
    ltv_requirements: LTVRequirements
    credit_requirements: CreditRequirements
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
