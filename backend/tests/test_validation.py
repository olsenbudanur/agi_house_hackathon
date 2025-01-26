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
    
    errors, warnings, confidence_scores = validate_matrix_data(data)
    assert not any("geographic" in err.lower() for err in errors)
    assert "geographic_restrictions" in confidence_scores

def test_reserve_requirements_validation():
    """Test validation of reserve requirements and ARM details."""
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
            "reserves": "12 months PITIA",
            "arm_margins": "2.25%",
            "arm_caps": "5/2/5",
            "qualifying_payment": "Greater of Note Rate or Fully Indexed Rate",
            "qualifying_rate": "Note Rate + 2%"
        }
    }
    
    errors, warnings, confidence_scores = validate_matrix_data(data)
    
    # Check reserve requirements
    assert not any("reserve" in err.lower() for err in errors)
    assert "reserve_requirements" in confidence_scores
    
    # Check ARM requirements
    assert "arm_margins" in data["additional_requirements"]
    assert "arm_caps" in data["additional_requirements"]
    assert "qualifying_payment" in data["additional_requirements"]
    assert "qualifying_rate" in data["additional_requirements"]
    
    # Validate ARM margin format
    margin = data["additional_requirements"]["arm_margins"]
    assert "%" in margin
    assert float(margin.strip("%")) > 0
    
    # Validate ARM caps format (e.g., "5/2/5")
    caps = data["additional_requirements"]["arm_caps"]
    cap_values = [int(x) for x in caps.split("/")]
    assert len(cap_values) == 3
    assert all(x > 0 for x in cap_values)
    
    # Validate qualifying payment rules
    payment_rule = data["additional_requirements"]["qualifying_payment"].lower()
    assert any(term in payment_rule for term in ["note rate", "index"])
    
    # Test with missing ARM data
    data_no_arm = data.copy()
    data_no_arm["additional_requirements"] = {"reserves": "12 months PITIA"}
    errors_no_arm, warnings_no_arm, _ = validate_matrix_data(data_no_arm)
    assert any("ARM" in warning for warning in warnings_no_arm)

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
    
    errors, warnings, confidence_scores = validate_matrix_data(data)
    assert any("reserve" in err.lower() for err in errors)
    assert confidence_scores["reserve_requirements"] < 0.5  # Low confidence due to missing data

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
    
    errors, warnings, confidence_scores = validate_matrix_data(data)
    assert any("12 months" in warning.lower() for warning in warnings)
    assert confidence_scores["reserve_requirements"] < 1.0  # Not perfect due to warning
