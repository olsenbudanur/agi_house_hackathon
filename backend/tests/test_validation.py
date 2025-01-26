import pytest
from app.validation import validate_matrix_data
from app.matrix_types import MatrixData, LoanRequirements, PropertyTypeRequirements, LTVRequirements
from app.matrix_types import CreditRequirements, PropertyRequirements, AdditionalRequirements, SpanningData

def test_geographic_restrictions_validation():
    """Test validation of geographic restrictions."""
    data = {
        "program_name": "Test Program",
        "effective_date": "2024-01-01",
        "processing_methods": ["OCR", "GPT-4"],
        "confidence_scores": {},
        "ltv_requirements": {
            "primary_residence": {
                "purchase": {
                    "max_ltv": "85%",
                    "min_fico": 740,
                    "max_loan": {"value": "$1,000,000", "span_index": None, "span_total": None}
                }
            }
        },
        "credit_requirements": {
            "minimum_fico": 680,
            "maximum_dti": 45,
            "credit_events": {}
        },
        "property_requirements": {
            "eligible_types": ["SFR", "Condo"],
            "geographic_restrictions": [
                "NJ - 10% reduction, Max 70% LTV",
                "CT, IL - 5% reduction, Max 75% LTV"
            ]
        }
    }
    
    errors, warnings = validate_matrix_data(data)
    assert not any("geographic" in err.lower() for err in errors)
    assert any("geographic_restrictions" in score for score in data.get("confidence_scores", {}))

def test_reserve_requirements_validation():
    """Test validation of reserve requirements."""
    data = {
        "program_name": "Test Program",
        "effective_date": "2024-01-01",
        "processing_methods": ["OCR", "GPT-4"],
        "confidence_scores": {},
        "ltv_requirements": {
            "primary_residence": {
                "purchase": {
                    "max_ltv": "85%",
                    "min_fico": 740,
                    "max_loan": {"value": "$1,500,000", "span_index": None, "span_total": None}
                }
            }
        },
        "credit_requirements": {
            "minimum_fico": 680,
            "maximum_dti": 45,
            "credit_events": {}
        },
        "additional_requirements": {
            "reserves": "12 months PITIA"
        }
    }
    
    errors, warnings = validate_matrix_data(data)
    assert not any("reserve" in err.lower() for err in errors)
    assert any("reserve_requirements" in score for score in data.get("confidence_scores", {}))

def test_missing_reserve_requirements():
    """Test validation when reserve requirements are missing."""
    data = {
        "program_name": "Test Program",
        "effective_date": "2024-01-01",
        "processing_methods": ["OCR", "GPT-4"],
        "confidence_scores": {},
        "ltv_requirements": {
            "primary_residence": {
                "purchase": {
                    "max_ltv": "85%",
                    "min_fico": 740,
                    "max_loan": {"value": "$1,500,000", "span_index": None, "span_total": None}
                }
            }
        },
        "credit_requirements": {
            "minimum_fico": 680,
            "maximum_dti": 45,
            "credit_events": {}
        },
        "additional_requirements": {}
    }
    
    errors, warnings = validate_matrix_data(data)
    assert any("reserve" in err.lower() for err in errors)

def test_high_loan_amount_reserve_check():
    """Test validation of reserve requirements for high loan amounts."""
    data = {
        "program_name": "Test Program",
        "effective_date": "2024-01-01",
        "processing_methods": ["OCR", "GPT-4"],
        "confidence_scores": {},
        "ltv_requirements": {
            "primary_residence": {
                "purchase": {
                    "max_ltv": "85%",
                    "min_fico": 740,
                    "max_loan": {"value": "$2,000,000", "span_index": None, "span_total": None}
                }
            }
        },
        "credit_requirements": {
            "minimum_fico": 680,
            "maximum_dti": 45,
            "credit_events": {}
        },
        "additional_requirements": {
            "reserves": "6 months PITIA"  # Should trigger warning for high loan amount
        }
    }
    
    errors, warnings = validate_matrix_data(data)
    assert any("12 months" in warning.lower() for warning in warnings)
