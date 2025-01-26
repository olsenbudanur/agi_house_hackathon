import pytest
from app.ocr_processor import parse_requirements_section

def test_parse_max_dti():
    """Test parsing of Max DTI requirements."""
    text = """
    Requirements ^ Max DTI
    Maximum DTI of 45%
    DTI cannot exceed 43% for investment properties
    """
    result = parse_requirements_section(text)
    assert result["max_dti"] == 45.0

def test_parse_credit_requirements():
    """Test parsing of credit requirements."""
    text = """
    Requirements ^ Credit Requirements
    Minimum FICO score of 720
    Maximum DTI of 43%
    Bankruptcy: None in last 7 years
    Foreclosure: None in last 7 years
    Short Sale: None in last 4 years
    """
    result = parse_requirements_section(text)
    assert result["credit_requirements"]["minimum_fico"] == 720
    assert result["credit_requirements"]["maximum_dti"] == 43.0
    assert "bankruptcy" in result["credit_requirements"]["credit_events"]
    assert "foreclosure" in result["credit_requirements"]["credit_events"]
    assert "short_sale" in result["credit_requirements"]["credit_events"]

def test_parse_reserve_requirements():
    """Test parsing of reserve requirements."""
    text = """
    Requirements ^ Reserve Requirements
    12 months PITIA required
    6 months PITIA for loans under $1MM
    """
    result = parse_requirements_section(text)
    assert result["reserve_requirements"] is not None
    assert "12 months" in result["reserve_requirements"]

def test_parse_geographic_restrictions():
    """Test parsing of geographic restrictions."""
    text = """
    Requirements ^ Geographic Restrictions
    NJ - 10% reduction, Max 70% LTV
    CT, IL - 5% reduction, Max 75% LTV
    """
    result = parse_requirements_section(text)
    assert len(result["geographic_restrictions"]) == 2
    assert any("NJ" in r for r in result["geographic_restrictions"])
    assert any("CT, IL" in r for r in result["geographic_restrictions"])

def test_multiple_sections():
    """Test parsing multiple requirement sections together."""
    text = """
    Requirements ^ Max DTI
    Maximum DTI of 45%
    
    Requirements ^ Credit Requirements
    Minimum FICO 720
    
    Requirements ^ Reserve Requirements
    12 months PITIA
    
    Requirements ^ Geographic Restrictions
    NJ - Max 70% LTV
    """
    result = parse_requirements_section(text)
    assert result["max_dti"] == 45.0
    assert result["credit_requirements"]["minimum_fico"] == 720
    assert result["reserve_requirements"] is not None
    assert len(result["geographic_restrictions"]) == 1
