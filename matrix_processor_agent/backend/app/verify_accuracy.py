import json
import logging
import os
import re
import sys
from pathlib import Path
from PIL import Image
import pytesseract
from typing import Dict, List, Tuple, Optional, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_actual_values(image_path: str) -> Dict[str, List[Tuple[str, str]]]:
    """Extract key values from the matrix image for verification."""
    try:
        # Load the image
        image = Image.open(image_path)
        
        # Extract text using OCR with specific configurations
        # Try different PSM modes for better table structure recognition
        psm_modes = ['6', '4', '11']  # 6=uniform block, 4=single column, 11=sparse text
        texts = []
        for psm in psm_modes:
            custom_config = f'--oem 3 --psm {psm}'
            text = pytesseract.image_to_string(image, config=custom_config)
            texts.append(text)
            logger.debug(f"OCR Output (PSM {psm}):\n{text}")
        
        # Combine all OCR results
        combined_text = '\n'.join(texts)
        
        # Extract and return key values
        values = {
            "ltv_values": [],
            "fico_scores": [],
            "loan_amounts": []
        }
        
        # Enhanced pattern matching with multiple variations
        ltv_patterns = [
            r'(\d{2,3})\s*%',  # Basic percentage
            r'(?:LTV|ltv)[:\s]*(\d{2,3})\s*%',  # LTV prefix
            r'(?:Max|Maximum)[:\s]*(?:LTV|ltv)[:\s]*(\d{2,3})\s*%',  # Max LTV prefix
            r'(\d{2,3})\s*(?:%|percent)',  # Percentage variations
        ]
        
        fico_patterns = [
            r'(?:FICO|Score)[:\s]*(\d{3})',  # Basic FICO
            r'(\d{3})\s*(?:FICO|Score)',  # FICO suffix
            r'(?:Min|Minimum)[:\s]*(?:FICO|Score)[:\s]*(\d{3})',  # Min FICO prefix
            r'(?:FICO|Score)[:\s]*(?:>=|>|=)\s*(\d{3})',  # FICO with comparison
            r'(?:FICO|Score)[:\s]*(\d{3})\+?',  # FICO with optional plus
            r'(?:^|\s|>)(\d{3})(?:\s|$)',  # Standalone 3-digit number
            r'(?:FICO|Score)?[:\s]*(\d{3})[:\s]*(?:Min|Minimum|Required)?',  # More variations
            r'(?:^|\|)\s*(\d{3})\s*(?:\||$)',  # Numbers in table cells
            r'(?:FICO|Score)?[:\s]*>=?\s*(\d{3})',  # Greater than or equal format
        ]
        
        # Additional preprocessing for FICO detection
        def clean_text_for_fico(text: str) -> str:
            # Remove common OCR errors
            text = re.sub(r'[lI]', '1', text)  # Replace l or I with 1
            text = re.sub(r'[oO]', '0', text)  # Replace o or O with 0
            text = re.sub(r'[sS]', '5', text)  # Replace s or S with 5
            return text
            
        # Clean the text for FICO detection
        combined_text = clean_text_for_fico(combined_text)
        
        loan_amount_patterns = [
            # Standard formats
            r'[\$]?\s*(?:1|one)[,.]?000[,.]?000',
            r'[\$]?\s*1[,.]000[,.]001\s*-\s*[\$]?\s*1[,.]500[,.]000',
            r'[\$]?\s*1[,.]500[,.]001\s*-\s*[\$]?\s*2[,.]000[,.]000',
            r'[\$]?\s*2[,.]000[,.]001\s*-\s*[\$]?\s*2[,.]500[,.]000',
            # Additional variations
            r'(?:<=|<|>|>=)?\s*[\$]?\s*\d{1,3}(?:[,.]\d{3})*(?:\s*[kKmMbB])?',
            r'[\$]?\s*\d{1,3}(?:[,.]\d{3})*\s*(?:-|to)\s*[\$]?\s*\d{1,3}(?:[,.]\d{3})*',
        ]
        
        # Process text in chunks for better context
        lines = combined_text.split('\n')
        for i, line in enumerate(lines):
            # Create context window (current line + previous and next)
            context = ' '.join(lines[max(0, i-1):min(len(lines), i+2)])
            
            # Extract LTV values using multiple patterns
            for ltv_pattern in ltv_patterns:
                ltv_matches = re.finditer(ltv_pattern, line, re.IGNORECASE)
                for match in ltv_matches:
                    ltv = match.group(1)
                    try:
                        ltv_value = int(ltv)
                        if 0 < ltv_value <= 100:  # Validate LTV range
                            values["ltv_values"].append(("Found", f"LTV: {ltv}%"))
                    except ValueError:
                        continue
            
            # Extract FICO scores with context using multiple patterns
            for fico_pattern in fico_patterns:
                fico_matches = re.finditer(fico_pattern, context, re.IGNORECASE)
                for match in fico_matches:
                    fico = match.group(1)
                    try:
                        fico_value = int(fico)
                        if 300 <= fico_value <= 850:  # Validate FICO range
                            values["fico_scores"].append(("Found", f"FICO: {fico}"))
                    except ValueError:
                        continue
            
            # Extract loan amounts with improved patterns
            for pattern in loan_amount_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    values["loan_amounts"].append(("Found", line.strip()))
        
        return values
        
    except Exception as e:
        logger.error(f"Error extracting values: {e}")
        return {}

def verify_extracted_data(actual_values: Dict, processed_data: Dict) -> None:
    """Compare extracted values with processed data."""
    logger.info("\n=== Verification Results ===")
    
    # Get the data from the processed response
    data = processed_data.get("data", {})
    if not data:
        logger.error("No data field found in processed response")
        return
        
    # Log raw extracted values for debugging
    logger.info("\nRaw Extracted Values:")
    for key, values in actual_values.items():
        logger.info(f"{key}:")
        for value in values:
            logger.info(f"  - {value}")
    
    # Verify LTV values with improved matching
    ltv_data = data.get("ltv_requirements", {})
    logger.info("\nLTV Requirements Verification:")
    
    # First, extract all LTV values from the raw text for better matching
    raw_ltv_values = set()
    for value in actual_values.get("ltv_values", []):
        matches = re.findall(r'(\d{2,3})%', value[1])
        raw_ltv_values.update(matches)
    
    logger.info(f"Found raw LTV values: {raw_ltv_values}")
    
    # Track verification statistics
    total_ltvs = 0
    verified_ltvs = 0
    
    for property_type, loans in ltv_data.items():
        logger.info(f"\n{property_type.replace('_', ' ').title()}:")
        for loan_type, details in loans.items():
            ltv = details.get("max_ltv", "").replace("%", "").strip()
            if ltv:
                total_ltvs += 1
                logger.info(f"  {loan_type.replace('_', ' ').title()}: {ltv}%")
                # Check if this LTV appears in raw values
                if ltv in raw_ltv_values:
                    logger.info(f"  ✓ Verified: {ltv}% found in matrix")
                    verified_ltvs += 1
                else:
                    logger.warning(f"  ⚠ Unverified LTV: {ltv}%")
    
    logger.info(f"\nLTV Verification Rate: {verified_ltvs}/{total_ltvs} ({(verified_ltvs/total_ltvs)*100:.1f}% verified)")
    
    # Verify FICO scores with improved matching
    logger.info("\nFICO Score Verification:")
    
    # Extract all FICO scores from raw text
    raw_fico_scores = set()
    for value in actual_values.get("fico_scores", []):
        matches = re.findall(r'(\d{3})', value[1])
        raw_fico_scores.update(int(score) for score in matches if 300 <= int(score) <= 850)
    
    logger.info(f"Found raw FICO scores: {raw_fico_scores}")
    
    # Track FICO verification statistics
    total_fico = 0
    verified_fico = 0
    
    fico_scores = set()
    for property_data in ltv_data.values():
        for loan_data in property_data.values():
            if loan_data.get("min_fico"):
                fico_scores.add(loan_data["min_fico"])
    
    for fico in sorted(fico_scores):
        total_fico += 1
        if fico in raw_fico_scores:
            logger.info(f"✓ Verified FICO {fico}")
            verified_fico += 1
        else:
            logger.warning(f"⚠ Unverified FICO score: {fico}")
            
    logger.info(f"\nFICO Verification Rate: {verified_fico}/{total_fico} ({(verified_fico/total_fico)*100:.1f}% verified)")
    
    # Verify loan amounts with improved matching
    logger.info("\nLoan Amount Verification:")
    
    # Define standard loan amount values for better matching
    standard_amounts = {
        1000000: ["1000000", "1,000,000", "1MM", "$1MM", "1,000"],
        1500000: ["1500000", "1,500,000", "1.5MM", "$1.5MM", "1,500"],
        2000000: ["2000000", "2,000,000", "2MM", "$2MM", "2,000"],
        2500000: ["2500000", "2,500,000", "2.5MM", "$2.5MM", "2,500"]
    }
    
    # Extract all loan amounts from raw text
    raw_amounts = []
    for value in actual_values.get("loan_amounts", []):
        clean_value = value[1].replace(",", "").replace("$", "").strip().upper()
        raw_amounts.append(clean_value)
    
    logger.info(f"Found raw loan amounts: {raw_amounts}")
    
    # Track loan amount verification statistics
    total_amounts = 0
    verified_amounts = 0
    
    loan_amounts = set()
    for property_data in ltv_data.values():
        for loan_data in property_data.values():
            if loan_data.get("max_loan"):
                loan_amounts.add(loan_data["max_loan"])
    
    for amount in sorted(loan_amounts):
        total_amounts += 1
        amount_variations = standard_amounts.get(amount, [str(amount), f"{amount:,}"])
        found = False
        for raw_amount in raw_amounts:
            if any(var in raw_amount for var in amount_variations):
                logger.info(f"✓ Verified Amount ${amount:,}")
                verified_amounts += 1
                found = True
                break
        if not found:
            logger.warning(f"⚠ Unverified loan amount: ${amount:,}")
            
    logger.info(f"\nLoan Amount Verification Rate: {verified_amounts}/{total_amounts} ({(verified_amounts/total_amounts)*100:.1f}% verified)")
    
    # Overall verification summary
    total_items = total_ltvs + total_fico + total_amounts
    total_verified = verified_ltvs + verified_fico + verified_amounts
    logger.info(f"\n=== Overall Verification Summary ===")
    logger.info(f"Total Items: {total_items}")
    logger.info(f"Verified Items: {total_verified}")
    logger.info(f"Overall Verification Rate: {(total_verified/total_items)*100:.1f}%")

if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if os.getenv("DEBUG") else logging.INFO,
        format='%(levelname)s:%(name)s:%(message)s'
    )
    
    # Paths
    image_path = Path("/home/ubuntu/attachments/4d1b1ab6-1714-4314-bbd1-d094645f9d75/Full+Doc+Matrix+Thumbnail.png")
    
    if len(sys.argv) != 2:
        logger.error("Usage: python -m app.verify_accuracy <response_json_file>")
        sys.exit(1)
        
    json_file = sys.argv[1]
    
    try:
        # Extract actual values from image
        logger.info("Extracting values from matrix image...")
        actual_values = extract_actual_values(str(image_path))
        
        # Read and parse the JSON response file
        logger.info(f"Reading processed data from {json_file}...")
        with open(json_file, 'r') as f:
            content = f.read().strip()
            # Remove any trailing output that might have been added
            if '\n' in content:
                content = content.split('\n')[0]
            
            processed_data = json.loads(content)
            if "data" not in processed_data:
                logger.error("Invalid response format: 'data' field missing")
                logger.debug(f"Received content: {content[:200]}...")
                sys.exit(1)
            
            # Verify the data
            verify_extracted_data(actual_values, processed_data)
            
    except FileNotFoundError:
        logger.error(f"Response file not found: {json_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON response: {e}")
        logger.debug(f"Content causing error: {content[:200]}...")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)
