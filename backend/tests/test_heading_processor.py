import pytest
from app.heading_processor import (
    parse_heading,
    combine_headings,
    detect_heading_pattern,
    extract_heading_components
)

def test_parse_heading_with_subheading():
    """Test parsing text with heading and subheading."""
    heading, subheading = parse_heading("Primary Residence ^ Rate/Term")
    assert heading == "Primary Residence"
    assert subheading == "Rate/Term"

def test_parse_heading_without_subheading():
    """Test parsing text with only heading."""
    heading, subheading = parse_heading("Program Requirements")
    assert heading == "Program Requirements"
    assert subheading is None

def test_combine_headings():
    """Test combining heading and subheading."""
    result = combine_headings("Primary Residence", "Purchase")
    assert result == "Primary Residence ^ Purchase"
    
    result = combine_headings("Credit Requirements")
    assert result == "Credit Requirements"

def test_detect_heading_pattern():
    """Test detection of common heading patterns."""
    # Test valid heading patterns
    is_heading, _ = detect_heading_pattern("Primary Residence")
    assert is_heading
    
    is_heading, _ = detect_heading_pattern("Second Home")
    assert is_heading
    
    is_heading, _ = detect_heading_pattern("Investment Property")
    assert is_heading
    
    is_heading, _ = detect_heading_pattern("Purchase")
    assert is_heading
    
    is_heading, _ = detect_heading_pattern("Rate and Term")
    assert is_heading
    
    is_heading, _ = detect_heading_pattern("Program Requirements")
    assert is_heading
    
    # Test invalid heading patterns
    is_heading, _ = detect_heading_pattern("Random Text")
    assert not is_heading
    
    is_heading, _ = detect_heading_pattern("123456")
    assert not is_heading
    
    # Test requirement category detection
    is_heading, category = detect_heading_pattern("Requirements ^ Max DTI")
    assert is_heading
    assert category == "max_dti_requirements"
    
    is_heading, category = detect_heading_pattern("Credit Requirements")
    assert is_heading
    assert category == "credit_requirements"

def test_extract_heading_components():
    """Test extraction of heading components."""
    # Test with valid heading and subheading
    result = extract_heading_components("Primary Residence ^ Rate/Term")
    assert result == ("Primary Residence", "Rate/Term")
    
    # Test with valid heading only
    result = extract_heading_components("Credit Requirements")
    assert result == ("Credit Requirements", None)
    
    # Test with non-heading text
    result = extract_heading_components("Random Text")
    assert result is None
