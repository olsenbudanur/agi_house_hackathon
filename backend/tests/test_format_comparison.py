import pytest
from fastapi.testclient import TestClient
from app.main import app
import json
import csv
from pathlib import Path
import os

client = TestClient(app)

def load_reference_data():
    """Load the reference CSV data for comparison."""
    ref_path = Path("/home/ubuntu/attachments/caba05df-5980-4a5a-87b5-6772f9e762dd/FullDocMatrixReference.csv")
    reference_data = {}
    with open(ref_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row.get('heading', '')}^{row.get('subheading', '')}" if row.get('subheading') else row.get('heading', '')
            reference_data[key] = row
    return reference_data

def test_pdf_png_format_comparison():
    """Compare processing results between PDF and PNG formats."""
    test_files = {
        "pdf": "/home/ubuntu/attachments/5823ee63-0834-4968-ade6-2c8273d6f697/Full-Doc-Matrix-thumb.pdf",
        "png": "/home/ubuntu/attachments/3c591b88-d4eb-4d4f-a6ae-721a7308f306/Full+Doc+Matrix+Thumbnail.png"
    }
    
    results = {}
    for fmt, filepath in test_files.items():
        with open(filepath, "rb") as f:
            content_type = "application/pdf" if fmt == "pdf" else "image/png"
            files = {"file": (f"test.{fmt}", f, content_type)}
            response = client.post("/process-matrix", files=files)
            assert response.status_code == 200, f"Failed to process {fmt}"
            results[fmt] = response.json()
    
    # Compare key metrics from notes.txt
    key_metrics = {
        "Program name detection": lambda x: x.get("program_name", ""),
        "Basic LTV percentages": lambda x: x.get("ltv_requirements", {}).get("primary_residence", {}).get("purchase", {}).get("max_ltv", ""),
        "FICO score requirements": lambda x: x.get("credit_requirements", {}).get("minimum_fico", 0),
        "Loan amount tiers": lambda x: x.get("ltv_requirements", {}).get("primary_residence", {}).get("purchase", {}).get("max_loan", {}).get("value", ""),
        "Property type requirements": lambda x: sorted(x.get("property_requirements", {}).get("eligible_types", [])),
        "Transaction type mapping": lambda x: sorted(list(x.get("ltv_requirements", {}).get("primary_residence", {}).keys()))
    }
    
    # Write comparison results
    results_dir = Path("/home/ubuntu/matrix-processor/backend/test_results")
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / "format_comparison.md", "w") as f:
        f.write("# PDF vs PNG Format Comparison Results\n\n")
        
        # Compare metrics
        f.write("## Key Metrics Comparison\n\n")
        for metric, extractor in key_metrics.items():
            pdf_value = extractor(results["pdf"].get("data", {}))
            png_value = extractor(results["png"].get("data", {}))
            match = pdf_value == png_value
            f.write(f"### {metric}\n")
            f.write(f"- PDF: {pdf_value}\n")
            f.write(f"- PNG: {png_value}\n")
            f.write(f"- Match: {'✓' if match else '✗'}\n\n")
        
        # Compare confidence scores
        f.write("## Confidence Scores\n\n")
        for fmt in results:
            f.write(f"### {fmt.upper()}\n")
            scores = results[fmt].get("data", {}).get("confidence_scores", {})
            for key, score in scores.items():
                f.write(f"- {key}: {score}\n")
            f.write("\n")
        
        # Compare validation results
        f.write("## Validation Results\n\n")
        for fmt in results:
            f.write(f"### {fmt.upper()}\n")
            errors = results[fmt].get("validation_errors", [])
            warnings = results[fmt].get("validation_warnings", [])
            f.write(f"- Errors: {len(errors)}\n")
            for err in errors:
                f.write(f"  - {err}\n")
            f.write(f"- Warnings: {len(warnings)}\n")
            for warn in warnings:
                f.write(f"  - {warn}\n")
            f.write("\n")

def test_reference_data_accuracy():
    """Compare processed data against reference CSV format."""
    reference = load_reference_data()
    
    # Test both formats against reference
    test_files = {
        "pdf": "/home/ubuntu/attachments/5823ee63-0834-4968-ade6-2c8273d6f697/Full-Doc-Matrix-thumb.pdf",
        "png": "/home/ubuntu/attachments/3c591b88-d4eb-4d4f-a6ae-721a7308f306/Full+Doc+Matrix+Thumbnail.png"
    }
    
    results_dir = Path("/home/ubuntu/matrix-processor/backend/test_results")
    with open(results_dir / "reference_comparison.md", "w") as f:
        f.write("# Reference Data Comparison Results\n\n")
        
        for fmt, filepath in test_files.items():
            f.write(f"## {fmt.upper()} Format Results\n\n")
            
            with open(filepath, "rb") as file:
                content_type = "application/pdf" if fmt == "pdf" else "image/png"
                files = {"file": (f"test.{fmt}", file, content_type)}
                response = client.post("/process-matrix", files=files)
                assert response.status_code == 200
                
                data = response.json().get("data", {})
                
                # Check key data points from reference
                f.write("### Geographic Restrictions\n")
                geo_restrictions = data.get("property_requirements", {}).get("geographic_restrictions", [])
                f.write("Found:\n")
                for restriction in geo_restrictions:
                    f.write(f"- {restriction}\n")
                f.write("\nExpected:\n- NJ (10% reduction)\n- CT/IL (5% reduction)\n\n")
                
                f.write("### Reserve Requirements\n")
                reserves = data.get("additional_requirements", {}).get("reserves", "")
                f.write(f"Found: {reserves}\n")
                f.write("Expected: Should include month counts\n\n")
                
                f.write("### Row-Spanning Data Check\n")
                loan_amounts = []
                for prop_type in data.get("ltv_requirements", {}):
                    for trans_type in data["ltv_requirements"][prop_type]:
                        amount = data["ltv_requirements"][prop_type][trans_type].get("max_loan", {})
                        if isinstance(amount, dict) and "span_index" in amount:
                            loan_amounts.append(amount)
                f.write("Found spanning data:\n")
                for amount in loan_amounts:
                    f.write(f"- {amount}\n")
                f.write("\n")
                
                f.write("### Heading Structure Check\n")
                headings = []
                for prop_type in data.get("ltv_requirements", {}):
                    for trans_type in data["ltv_requirements"][prop_type]:
                        heading = data["ltv_requirements"][prop_type][trans_type].get("heading", {})
                        if heading:
                            headings.append(str(heading))
                f.write("Found headings:\n")
                for heading in headings:
                    f.write(f"- {heading}\n")
                f.write("\n")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
