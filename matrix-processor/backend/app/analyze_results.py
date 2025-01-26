import json
import sys
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_matrix_response(response_json: Dict[str, Any]) -> None:
    """Analyze the matrix processing response and print detailed information."""
    logger.info("Analyzing matrix processing response...")
    
    # Check validation status
    validation = response_json.get("validation", {})
    logger.info(f"Validation Status: {'Valid' if validation.get('is_valid') else 'Invalid'}")
    if validation.get("errors"):
        logger.error("Validation Errors:")
        for error in validation["errors"]:
            logger.error(f"  - {error}")
    if validation.get("warnings"):
        logger.warning("Validation Warnings:")
        for warning in validation["warnings"]:
            logger.warning(f"  - {warning}")
    
    # Analyze LTV requirements
    ltv_data = response_json.get("data", {}).get("ltv_requirements", {})
    logger.info("\nLTV Requirements Analysis:")
    for property_type in ["primary_residence", "second_home", "investment"]:
        logger.info(f"\n{property_type.replace('_', ' ').title()}:")
        property_data = ltv_data.get(property_type, {})
        for loan_type in ["purchase", "rate_and_term", "cash_out"]:
            loan_data = property_data.get(loan_type, {})
            logger.info(f"  {loan_type.replace('_', ' ').title()}:")
            logger.info(f"    - Max LTV: {loan_data.get('max_ltv')}")
            logger.info(f"    - Min FICO: {loan_data.get('min_fico')}")
            logger.info(f"    - Max Loan: ${loan_data.get('max_loan'):,}")

    # Analyze credit requirements
    credit_data = response_json.get("data", {}).get("credit_requirements", {})
    logger.info("\nCredit Requirements:")
    logger.info(f"  Minimum FICO: {credit_data.get('minimum_fico')}")
    logger.info(f"  Maximum DTI: {credit_data.get('maximum_dti')}%")
    
    # Processing method information
    method = response_json.get("processing_method", {})
    logger.info(f"\nProcessing Method: {method.get('name')} - {method.get('description')}")

if __name__ == "__main__":
    # Read response from curl command
    response_text = sys.stdin.read()
    try:
        response_data = json.loads(response_text)
        analyze_matrix_response(response_data)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        sys.exit(1)
