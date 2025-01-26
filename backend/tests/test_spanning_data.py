import pytest
from app.ocr_processor import detect_spanning_data

def test_detect_spanning_data_basic():
    """Test basic spanning data detection."""
    lines = [
        "Min Loan Amount (1)",
        "$250,000 (1)",
        "Min Loan Amount (2)",
        "$250,000 (2)"
    ]
    pattern = r"Min Loan Amount|[\$]?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?"
    result = detect_spanning_data(lines, 0, pattern)
    assert len(result) == 2
    assert result[0][1] == 1  # First index
    assert result[1][1] == 2  # Second index
    assert result[0][2] == 2  # Total spans
    assert result[1][2] == 2  # Total spans

def test_detect_spanning_data_implicit_first():
    """Test spanning data with implicit first index."""
    lines = [
        "Residual Income",
        "$3,000 (1)",
        "Residual Income (2)",
        "$3,000 (2)"
    ]
    pattern = r"Residual Income|[\$]?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?"
    result = detect_spanning_data(lines, 0, pattern)
    assert len(result) == 2
    assert "Residual Income (1)" in result[0][0]
    assert result[0][1] == 1
    assert result[1][1] == 2

def test_detect_spanning_data_single():
    """Test non-spanning single line data."""
    lines = [
        "Max Loan Amount",
        "$2,500,000"
    ]
    pattern = r"Max Loan Amount|[\$]?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?"
    result = detect_spanning_data(lines, 0, pattern)
    assert len(result) == 1
    assert result[0][1] == 0  # No span index
    assert result[0][2] == 1  # Total span of 1

def test_detect_spanning_data_sequence_repair():
    """Test sequence repair for inconsistent indices."""
    lines = [
        "No Housing History (1)",
        "Max 75% LTV (3)",  # Inconsistent index
        "Min 700 FICO (2)"
    ]
    pattern = r"No Housing History|Max \d+%|Min \d+"
    result = detect_spanning_data(lines, 0, pattern)
    assert len(result) == 3
    indices = [r[1] for r in result]
    assert indices == [1, 2, 3]  # Should be repaired to sequential
