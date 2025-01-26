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
    assert detect_heading_pattern("Primary Residence")
    assert detect_heading_pattern("Second Home")
    assert detect_heading_pattern("Investment Property")
    assert detect_heading_pattern("Purchase")
    assert detect_heading_pattern("Rate and Term")
    assert detect_heading_pattern("Program Requirements")
    assert not detect_heading_pattern("Random Text")
    assert not detect_heading_pattern("123456")

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
