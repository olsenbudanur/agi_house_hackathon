from typing import Dict, List, Union, Optional

from typing import Dict, List, Tuple, Union
from .matrix_types import MatrixData, ValidationResult

def validate_ltv(ltv: str) -> bool:
    """Validate LTV percentage."""
    try:
        ltv_value = float(str(ltv).strip('%'))
        return 0 <= ltv_value <= 100
    except ValueError:
        return False

def validate_fico(score: int) -> bool:
    """Validate FICO score."""
    return 300 <= score <= 850

def validate_loan_amount(amount: float) -> bool:
    """Validate loan amount."""
    return 10000 <= amount <= 100000000

def validate_dti(dti: float) -> bool:
    """Validate Debt-to-Income ratio."""
    return 0 <= dti <= 100

def validate_matrix_data(data: Dict) -> Tuple[List[str], List[str]]:
    """
    Validates the extracted matrix data against known rules and patterns.
    Returns a tuple of (errors, warnings).
    """
    errors = []
    warnings = []
    confidence_scores = {
        "ltv_values": 0.0,
        "fico_scores": 0.0,
        "loan_amounts": 0.0,
        "overall": 0.0,
        "matrix_structure": 0.0,
        "gpt4_analysis": 0.0
    }
    total_checks = 0
    passed_checks = 0
    
    try:
        # Convert dict to MatrixData model for initial validation
        matrix_data = MatrixData(**data)
        
        # Validate LTV requirements
        for property_type in ['primary_residence', 'second_home', 'investment']:
            property_reqs = getattr(matrix_data.ltv_requirements, property_type)
            for loan_type in ['purchase', 'rate_and_term', 'cash_out']:
                loan_reqs = getattr(property_reqs, loan_type)
                total_checks += 3  # One each for LTV, FICO, and loan amount
                
                if validate_ltv(loan_reqs.max_ltv):
                    passed_checks += 1
                    confidence_scores["ltv_values"] += 0.111  # 1/9 for each LTV (9 total combinations)
                else:
                    errors.append(
                        f"Invalid LTV value in {property_type}/{loan_type}: {loan_reqs.max_ltv}"
                    )
                
                if validate_fico(loan_reqs.min_fico):
                    passed_checks += 1
                    confidence_scores["fico_scores"] += 0.111
                else:
                    errors.append(
                        f"Invalid FICO score in {property_type}/{loan_type}: {loan_reqs.min_fico}"
                    )
                
                if validate_loan_amount(loan_reqs.max_loan):
                    passed_checks += 1
                    confidence_scores["loan_amounts"] += 0.111
                else:
                    errors.append(
                        f"Invalid loan amount in {property_type}/{loan_type}: {loan_reqs.max_loan}"
                    )
        
        # Validate credit requirements
        total_checks += 2  # FICO and DTI checks
        
        if validate_fico(matrix_data.credit_requirements.minimum_fico):
            passed_checks += 1
            confidence_scores["fico_scores"] += 0.5  # Additional weight for global FICO requirement
        else:
            errors.append(
                f"Invalid minimum FICO score: {matrix_data.credit_requirements.minimum_fico}"
            )
            
        if validate_dti(matrix_data.credit_requirements.maximum_dti):
            passed_checks += 1
        else:
            errors.append(
                f"Invalid maximum DTI: {matrix_data.credit_requirements.maximum_dti}"
            )
        
        # Validate relationships between values
        total_checks += 3  # Additional relationship checks
        
        # Check if primary residence LTVs are higher than second home
        primary_ltvs = [float(getattr(matrix_data.ltv_requirements.primary_residence, lt).max_ltv.strip('%')) 
                       for lt in ['purchase', 'rate_and_term', 'cash_out']]
        second_ltvs = [float(getattr(matrix_data.ltv_requirements.second_home, lt).max_ltv.strip('%')) 
                      for lt in ['purchase', 'rate_and_term', 'cash_out']]
        
        if all(p >= s for p, s in zip(primary_ltvs, second_ltvs)):
            passed_checks += 1
            confidence_scores["ltv_values"] += 0.2
        else:
            warnings.append("Unexpected LTV relationship: primary residence vs second home")
            
        # Check if second home LTVs are higher than investment
        investment_ltvs = [float(getattr(matrix_data.ltv_requirements.investment, lt).max_ltv.strip('%')) 
                          for lt in ['purchase', 'rate_and_term', 'cash_out']]
        
        if all(s >= i for s, i in zip(second_ltvs, investment_ltvs)):
            passed_checks += 1
            confidence_scores["ltv_values"] += 0.2
        else:
            warnings.append("Unexpected LTV relationship: second home vs investment")
            
        # Add warnings for potentially problematic values
        if matrix_data.credit_requirements.minimum_fico < 620:
            warnings.append("Unusually low minimum FICO score detected")
            
        # Check property requirements if available
        if matrix_data.property_requirements and len(matrix_data.property_requirements.eligible_types) == 0:
            warnings.append("No eligible property types specified")
            
        # Update confidence scores based on GPT-4 analysis
        if matrix_data.gpt4_structured_analysis:
            confidence_scores["gpt4_analysis"] = 1.0
            confidence_scores["matrix_structure"] += 0.5
            
        # Improve matrix structure confidence if we have consistent data
        if all(hasattr(matrix_data.ltv_requirements, pt) for pt in ['primary_residence', 'second_home', 'investment']):
            confidence_scores["matrix_structure"] += 0.5
            
        # Normalize confidence scores
        for key in confidence_scores:
            if key != "overall":
                confidence_scores[key] = min(1.0, confidence_scores[key])
            
    except Exception as e:
        errors.append(f"Validation error: {str(e)}")
    
    # Calculate confidence scores
    if total_checks > 0:
        confidence_scores["overall"] = passed_checks / total_checks
        
    # Add confidence information to warnings if scores are low
    for metric, score in confidence_scores.items():
        if score < 0.7:  # Less than 70% confidence
            warnings.append(f"Low confidence in {metric} detection: {score:.1%}")
    
    return errors, warnings

def format_validation_response(data: Dict, errors: List[str], warnings: List[str]) -> Dict:
    """
    Formats the validation response with the data, validation messages, and confidence scores.
    """
    # Calculate confidence scores based on validation results
    confidence = {
        "ltv_values": 1.0 if not any("LTV" in err for err in errors) else 0.5,
        "fico_scores": 1.0 if not any("FICO" in err for err in errors) else 0.5,
        "loan_amounts": 1.0 if not any("loan amount" in err for err in errors) else 0.5
    }
    
    # Overall confidence is the average of individual scores
    confidence["overall"] = sum(confidence.values()) / len(confidence)
    
    return {
        "data": data,
        "validation": {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "confidence_scores": confidence
        }
    }
