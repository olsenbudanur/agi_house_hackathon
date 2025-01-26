import pytest
from app.ocr_processor import parse_ltv_section
from app.matrix_types import HeadingData

def test_parse_ltv_with_hierarchical_headings():
    """Test parsing LTV section with hierarchical heading structure."""
    sample_text = """
    Primary Residence ^ Purchase
    Max LTV 85%
    Min FICO 740
    Max Loan Amount $1,000,000
    
    Second Home ^ Rate and Term
    Max LTV 75%
    Min FICO 740
    Max Loan Amount $750,000
    
    Investment Property
    Max LTV 70%
    Min FICO 760
    Max Loan Amount $500,000
    """
    
    result = parse_ltv_section(sample_text)
    
    # Check hierarchical heading structure
    assert "Primary Residence ^ Purchase" in result
    assert result["Primary Residence ^ Purchase"]["heading"].heading == "Primary Residence"
    assert result["Primary Residence ^ Purchase"]["heading"].subheading == "Purchase"
    assert result["Primary Residence ^ Purchase"]["max_ltv"] == "85%"
    
    # Check single heading structure
    assert "Investment Property" in result
    assert result["Investment Property"]["heading"].heading == "Investment Property"
    assert result["Investment Property"]["heading"].subheading is None
    assert result["Investment Property"]["max_ltv"] == "70%"

def test_parse_ltv_with_spanning_data_and_headings():
    """Test parsing LTV section with both spanning data and headings."""
    sample_text = """
    Primary Residence ^ Purchase
    Max LTV 85%
    Min FICO 740
    Max Loan Amount $1,000,000 (1)
    Max Loan Amount $750,000 (2)
    """
    
    result = parse_ltv_section(sample_text)
    
    section = result["Primary Residence ^ Purchase"]
    assert section["heading"].heading == "Primary Residence"
    assert section["heading"].subheading == "Purchase"
    assert section["max_loan"]["value"] == "$1,000,000"
    assert section["max_loan"]["span_index"] == 1
    assert section["max_loan"]["span_total"] == 2
