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

def validate_loan_amount(amount: Dict) -> bool:
    """Validate loan amount with support for spanning data."""
    try:
        # Extract numeric value from string (remove $ and ,)
        value_str = amount["value"].replace("$", "").replace(",", "")
        numeric_value = float(value_str)
        
        # Basic range validation
        if not (10000 <= numeric_value <= 100000000):
            return False
            
        # Validate spanning data if present
        if amount["span_index"] is not None:
            if amount["span_index"] < 1 or amount["span_total"] < amount["span_index"]:
                return False
                
        return True
    except (ValueError, KeyError, TypeError):
        return False

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
        "gpt4_analysis": 0.0,
        "geographic_restrictions": 0.0,
        "reserve_requirements": 0.0
    }
    total_checks = 0
    passed_checks = 0
    
    try:
        # Convert dict to MatrixData model for validation
        matrix_data = MatrixData(**data)
        
        # Validate geographic restrictions
        if matrix_data.property_requirements and matrix_data.property_requirements.geographic_restrictions:
            total_checks += 1
            geo_restrictions = matrix_data.property_requirements.geographic_restrictions
            
            # Check for state-specific LTV adjustments
            for restriction in geo_restrictions:
                if any(state in restriction.upper() for state in ["NJ", "CT", "IL"]):
                    passed_checks += 1
                    confidence_scores["geographic_restrictions"] += 0.5
                    
                    # Look for LTV reduction indicators
                    if "reduction" in restriction.lower():
                        confidence_scores["geographic_restrictions"] += 0.5
                    else:
                        warnings.append(f"Geographic restriction found for {restriction} but no clear LTV reduction specified")
        
        # Validate reserve requirements
        if hasattr(matrix_data, "additional_requirements") and matrix_data.additional_requirements:
            total_checks += 1
            reserves = matrix_data.additional_requirements.reserves
            
            if reserves:
                passed_checks += 1
                confidence_scores["reserve_requirements"] += 0.5
                
                # Check for minimum reserve requirements
                if isinstance(reserves, str):
                    reserve_text = reserves.lower()
                    if any(str(i) in reserve_text for i in range(6, 13)):  # Check for 6-12 months
                        confidence_scores["reserve_requirements"] += 0.5
                    else:
                        warnings.append("Reserve requirements may be below minimum expected months")
                        
                # Additional reserve requirement checks
                if hasattr(matrix_data, "ltv_requirements"):
                    loan_amount = None
                    for prop in ["primary_residence", "second_home", "investment"]:
                        if hasattr(matrix_data.ltv_requirements, prop):
                            prop_reqs = getattr(matrix_data.ltv_requirements, prop)
                            if hasattr(prop_reqs, "purchase"):
                                if prop_reqs.purchase.max_loan:
                                    loan_amount = prop_reqs.purchase.max_loan.value
                                    break
                    
                    if loan_amount:
                        try:
                            amount = float(loan_amount.replace("$", "").replace(",", ""))
                            if amount > 1000000 and "12" not in str(reserves):
                                warnings.append("Loan amount exceeds $1M but 12-month reserves not explicitly specified")</old_str>
<new_str>def validate_matrix_data(data: Dict) -> Tuple[List[str], List[str]]:
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
        "gpt4_analysis": 0.0,
        "geographic_restrictions": 0.0,
        "reserve_requirements": 0.0
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
        if matrix_data.property_requirements:
            total_checks += 2  # Geographic and property type checks
            
            # Validate geographic restrictions
            if matrix_data.property_requirements.geographic_restrictions:
                passed_checks += 1
                for restriction in matrix_data.property_requirements.geographic_restrictions:
                    restriction_lower = restriction.lower()
                    
                    # Check for state-specific LTV adjustments with detailed validation
                    if "nj" in restriction_lower:
                        if any(x in restriction_lower for x in ["10%", "10 %", "reduction", "70%", "70 %"]):
                            confidence_scores["geographic_restrictions"] += 0.3
                            if "70%" in restriction_lower or "70 %" in restriction_lower:
                                confidence_scores["geographic_restrictions"] += 0.2
                        else:
                            warnings.append("NJ restriction found but unclear LTV reduction (expecting 10% reduction or max 70% LTV)")
                            
                    if "ct" in restriction_lower or "il" in restriction_lower:
                        if any(x in restriction_lower for x in ["5%", "5 %", "reduction", "75%", "75 %"]):
                            confidence_scores["geographic_restrictions"] += 0.2
                            if "75%" in restriction_lower or "75 %" in restriction_lower:
                                confidence_scores["geographic_restrictions"] += 0.1
                        else:
                            state = "CT" if "ct" in restriction_lower else "IL"
                            warnings.append(f"{state} restriction found but unclear LTV reduction (expecting 5% reduction or max 75% LTV)")
            else:
                warnings.append("No geographic restrictions specified - verify if this is intentional")
            
            # Check eligible property types
            if len(matrix_data.property_requirements.eligible_types) == 0:
                warnings.append("No eligible property types specified")
            else:
                passed_checks += 1
                
                # Validate common property types are included
                common_types = {"sfr", "pud", "condo", "townhouse"}
                found_types = {t.lower() for t in matrix_data.property_requirements.eligible_types}
                if not any(t in found_types for t in common_types):
                    warnings.append("No common property types (SFR, PUD, Condo, Townhouse) found in eligible types")
        
        # Validate reserve requirements with enhanced checks
        if hasattr(matrix_data, "additional_requirements") and matrix_data.additional_requirements:
            total_checks += 3  # Basic reserves, loan amount specific checks, and PITIA validation
            reserves = matrix_data.additional_requirements.reserves
            
            if reserves:
                passed_checks += 1
                reserve_text = str(reserves).lower()
                
                # Check for PITIA specification
                if "pitia" in reserve_text or "p.i.t.i.a" in reserve_text:
                    confidence_scores["reserve_requirements"] += 0.2
                else:
                    warnings.append("Reserve requirements should specify PITIA (Principal, Interest, Taxes, Insurance, Association Dues)")
                
                # Enhanced check for minimum reserve requirements
                months_found = [str(i) for i in range(6, 13) if str(i) in reserve_text or 
                              any(word in reserve_text for word in [f"{i} month", f"{i}-month"])]
                
                if months_found:
                    confidence_scores["reserve_requirements"] += 0.3
                    months = int(months_found[0])
                    if months < 12:
                        warnings.append(f"Reserve requirement of {months} months may be insufficient for some loan amounts")
                else:
                    warnings.append("Reserve requirements may be below minimum expected months (6-12 months PITIA typically required)")
                
                # Additional reserve checks based on loan amount with enhanced validation
                try:
                    max_loan_amount = 0
                    for prop_type in ["primary_residence", "second_home", "investment"]:
                        if hasattr(matrix_data.ltv_requirements, prop_type):
                            prop_reqs = getattr(matrix_data.ltv_requirements, prop_type)
                            for loan_type in ["purchase", "rate_and_term", "cash_out"]:
                                if hasattr(prop_reqs, loan_type):
                                    loan_req = getattr(prop_reqs, loan_type)
                                    if loan_req.max_loan and loan_req.max_loan.value:
                                        amount_str = loan_req.max_loan.value.replace("$", "").replace(",", "")
                                        amount = float(amount_str)
                                        max_loan_amount = max(max_loan_amount, amount)
                    
                    if max_loan_amount > 0:
                        passed_checks += 1
                        if max_loan_amount > 1000000:
                            if "12" in reserve_text or "twelve" in reserve_text:
                                confidence_scores["reserve_requirements"] += 0.2
                            else:
                                warnings.append(f"High loan amount (${max_loan_amount:,.2f}) requires 12 months PITIA reserves")
                        elif max_loan_amount <= 1000000:
                            if "6" in reserve_text or "six" in reserve_text:
                                confidence_scores["reserve_requirements"] += 0.2
                            
                except (ValueError, AttributeError) as e:
                    warnings.append(f"Could not validate reserves against loan amounts: {str(e)}")
            else:
                errors.append("No reserve requirements specified - this is required for all loans")
                
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
