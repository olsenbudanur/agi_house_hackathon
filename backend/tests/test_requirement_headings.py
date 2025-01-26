import pytest
from app.heading_processor import detect_heading_pattern, parse_heading

def test_detect_max_dti_requirement():
    """Test detection of Max DTI requirement headings."""
    test_cases = [
        "Requirements ^ Max DTI",
        "Max DTI Requirements",
        "DTI Requirements",
        "Requirements ^ DTI Limits"
    ]
    for text in test_cases:
        is_heading, category = detect_heading_pattern(text)
        assert is_heading
        assert category == 'max_dti_requirements'

def test_detect_credit_requirements():
    """Test detection of Credit requirement headings."""
    test_cases = [
        "Requirements ^ Credit Requirements",
        "Credit Requirements",
        "Requirements ^ Credit Guidelines"
    ]
    for text in test_cases:
        is_heading, category = detect_heading_pattern(text)
        assert is_heading
        assert category == 'credit_requirements'

def test_detect_reserve_requirements():
    """Test detection of Reserve requirement headings."""
    test_cases = [
        "Requirements ^ Reserve Requirements",
        "Reserve Requirements",
        "Requirements ^ Reserves"
    ]
    for text in test_cases:
        is_heading, category = detect_heading_pattern(text)
        assert is_heading
        assert category == 'reserve_requirements'

def test_detect_geographic_restrictions():
    """Test detection of Geographic restriction headings."""
    test_cases = [
        "Requirements ^ Geographic Restrictions",
        "Geographic Restrictions",
        "Geographic Requirements"
    ]
    for text in test_cases:
        is_heading, category = detect_heading_pattern(text)
        assert is_heading
        assert category == 'geographic_restrictions'

def test_heading_with_subheading():
    """Test parsing of headings with subheadings."""
    text = "Requirements ^ Max DTI"
    is_heading, category = detect_heading_pattern(text)
    assert is_heading
    assert category == 'max_dti_requirements'
    
    heading, subheading = parse_heading(text)
    assert heading == "Requirements"
    assert subheading == "Max DTI"

def test_non_requirement_heading():
    """Test that non-requirement headings are still detected but without category."""
    text = "Primary Residence"
    is_heading, category = detect_heading_pattern(text)
    assert is_heading
    assert category is None

def test_non_heading():
    """Test that non-headings return false with no category."""
    text = "Some random text"
    is_heading, category = detect_heading_pattern(text)
    assert not is_heading
    assert category is None
