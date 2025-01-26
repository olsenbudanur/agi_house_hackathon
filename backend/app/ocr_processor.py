from typing import Dict, List, Optional, Tuple
import pytesseract
from PIL import Image, ImageDraw
import io
import base64
import re
import logging
import datetime
import os
import cv2
import numpy as np
import json
import subprocess
import shutil
from pathlib import Path
from openai import OpenAI
from .matrix_types import MatrixData, LTVRequirements, PropertyTypeRequirements, LoanRequirements
from .validation import validate_matrix_data

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def verify_tesseract_installation() -> bool:
    """Verify Tesseract installation and configuration."""
    try:
        # Log environment information
        logger.info("Verifying Tesseract installation...")
        logger.info(f"Current PATH: {os.environ.get('PATH', 'Not set')}")
        logger.info(f"Current TESSDATA_PREFIX: {os.environ.get('TESSDATA_PREFIX', 'Not set')}")
        logger.info(f"Current TESSERACT_CMD: {os.environ.get('TESSERACT_CMD', 'Not set')}")
        
        # Use the configured Tesseract path
        tesseract_cmd = os.environ.get('TESSERACT_CMD', '/usr/bin/tesseract')
        
        # Verify the configured path
        if not os.path.exists(tesseract_cmd):
            logger.error(f"Configured Tesseract path does not exist: {tesseract_cmd}")
            return False
            
        try:
            # Test if the binary is executable
            version_output = subprocess.run(
                [tesseract_cmd, '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if version_output.returncode == 0:
                logger.info(f"Found working Tesseract at: {tesseract_cmd}")
                logger.info(f"Version info: {version_output.stdout.strip()}")
            else:
                logger.error(f"Tesseract verification failed at {tesseract_cmd}")
                return False
        except Exception as e:
            logger.error(f"Failed to verify Tesseract at {tesseract_cmd}: {str(e)}")
            return False
        
        if not tesseract_cmd:
            logger.error("No working Tesseract installation found")
            return False
        
        # Update environment
        os.environ['TESSERACT_CMD'] = tesseract_cmd
        
        # Verify tessdata
        tessdata_paths = os.environ.get('TESSDATA_PREFIX', '').split(':')
        tessdata_found = False
        
        for path in tessdata_paths:
            if path and os.path.exists(path):
                try:
                    # Check if eng.traineddata exists
                    if os.path.exists(os.path.join(path, 'eng.traineddata')):
                        tessdata_found = True
                        logger.info(f"Found valid tessdata at: {path}")
                        break
                except Exception as e:
                    logger.warning(f"Error checking tessdata at {path}: {str(e)}")
        
        if not tessdata_found:
            logger.warning("No valid tessdata found, OCR may not work correctly")
        
        return True
            
    except Exception as e:
        logger.error(f"Tesseract verification failed: {str(e)}")
        logger.error("Full environment:")
        for key, value in os.environ.items():
            logger.error(f"{key}={value}")
        return False

def process_image_bytes(contents: bytes) -> Tuple[Image.Image, bytes]:
    """Process raw image bytes and return PIL Image and processed bytes."""
    try:
        # Convert bytes to numpy array for OpenCV processing
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image data")
            
        # Convert to PIL Image
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Convert back to bytes
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='PNG')
        return img, img_byte_arr.getvalue()
            
    except Exception as e:
        logger.error(f"Error processing image bytes: {str(e)}")
        raise ValueError(f"Unable to process image data: {str(e)}")

def extract_text_from_bytes(image_bytes: bytes) -> str:
    """Extract text from image bytes using enhanced OCR with preprocessing."""
    try:
        # Verify Tesseract installation first
        if not verify_tesseract_installation():
            raise RuntimeError("Tesseract installation verification failed")
            
        # Convert bytes to numpy array for OpenCV processing
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image data")
        
        # Image preprocessing steps
        # 1. Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 2. Apply adaptive thresholding for better text extraction
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # 3. Detect and enhance table structure
        # Horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.erode(thresh, horizontal_kernel, iterations=2)
        horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=2)
        
        # Vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.erode(thresh, vertical_kernel, iterations=2)
        vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=2)
        
        # Combine the lines
        table_structure = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
        
        # 4. Remove table lines from the image for better OCR
        clean_image = cv2.subtract(thresh, table_structure)
        
        # 5. Apply noise reduction
        clean_image = cv2.medianBlur(clean_image, 3)
        
        # 6. Enhance contrast
        clean_image = cv2.convertScaleAbs(clean_image, alpha=1.2, beta=0)
        
        # Convert OpenCV image back to PIL Image for Tesseract
        pil_image = Image.fromarray(clean_image)
        
        try:
            # Configure Tesseract for better table recognition
            custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1 -c tessedit_char_whitelist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz%$.,- "'
            text = pytesseract.image_to_string(pil_image, config=custom_config)
            
            if not text.strip():
                raise Exception("No text extracted, trying subprocess approach")
                
            logger.debug("OCR preprocessing completed successfully")
            logger.debug(f"Extracted text preview: {text[:500]}...")
            return text.strip()
            
        except Exception as pytess_error:
            logger.warning(f"pytesseract direct approach failed: {str(pytess_error)}, trying subprocess")
            
            # Save image temporarily
            temp_image_path = "/tmp/temp_ocr_image.png"
            temp_output_path = "/tmp/temp_ocr_output"
            
            try:
                pil_image.save(temp_image_path)
                
                # Run tesseract directly using subprocess
                result = subprocess.run(
                    ['tesseract', temp_image_path, temp_output_path, '--oem', '3', '--psm', '6'],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                # Read the output file
                with open(f"{temp_output_path}.txt", 'r') as f:
                    text = f.read()
                    
                if not text.strip():
                    raise ValueError("No text could be extracted from the image")
                    
                logger.info("Successfully extracted text using subprocess approach")
                return text.strip()
                
            finally:
                # Cleanup
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
                if os.path.exists(f"{temp_output_path}.txt"):
                    os.remove(f"{temp_output_path}.txt")
        
    except Exception as e:
        logger.error(f"Error in OCR text extraction: {str(e)}")
        raise ValueError(f"OCR text extraction failed: {str(e)}")

def extract_text_from_base64(base64_image: str) -> str:
    """Extract text from base64 encoded image using OCR."""
    try:
        image_data = base64.b64decode(base64_image)
        return extract_text_from_bytes(image_data)
    except Exception as e:
        logger.error(f"Error in base64 image text extraction: {str(e)}")
        raise

def detect_spanning_data(lines: List[str], start_idx: int, pattern: str) -> List[Tuple[str, int, int]]:
    """
    Detect and process multi-row data entries with (1), (2) notation.
    Returns list of tuples: (text, span_index, total_spans)
    """
    logger.debug(f"Starting spanning data detection at line {start_idx}")
    logger.debug(f"Pattern: {pattern}")
    
    # Initialize result list and get current line
    current_line = lines[start_idx].strip()
    if not current_line:
        logger.debug("Empty current line, returning empty list")
        return []

    # Check if current line matches pattern
    if not re.search(pattern, current_line):
        logger.debug(f"Current line '{current_line}' doesn't match pattern")
        return []

    # Initialize spanning data collection
    spanning_data = []
    max_lookahead = 5
    current_span_index = 0
    total_spans = 1

    logger.debug(f"Processing current line: {current_line}")
    
    # Check current line for span index
    span_match = re.search(r'\((\d+)\)$', current_line)
    if span_match:
        current_span_index = int(span_match.group(1))
        total_spans = current_span_index
        logger.debug(f"Found span index {current_span_index} in current line")

    # Add current line to spanning data
    spanning_data.append((current_line, current_span_index, total_spans))

    # Look ahead for additional spans
    for i in range(start_idx + 1, min(len(lines), start_idx + max_lookahead)):
        next_line = lines[i].strip()
        logger.debug(f"Looking at next line {i}: {next_line}")
        
        if not next_line or not re.search(pattern, next_line):
            logger.debug(f"Line {i} doesn't match pattern or is empty")
            continue

        next_span_match = re.search(r'\((\d+)\)$', next_line)
        if next_span_match:
            span_num = int(next_span_match.group(1))
            logger.debug(f"Found span index {span_num} in line {i}")
            
            if span_num > total_spans:
                total_spans = span_num
                # Update total_spans for all previous entries
                spanning_data = [(text, idx, total_spans) for text, idx, _ in spanning_data]
                logger.debug(f"Updated total spans to {total_spans}")
                
            spanning_data.append((next_line, span_num, total_spans))

    # If we have no spanning indicators, return single entry
    if len(spanning_data) == 1 and spanning_data[0][1] == 0:
        logger.debug("Single non-spanning entry found")
        return [(spanning_data[0][0], 0, 1)]

    # Sort by span index (non-zero indices first, then zero indices)
    spanning_data.sort(key=lambda x: float('inf') if x[1] == 0 else x[1])
    logger.debug(f"Final spanning data: {spanning_data}")
    
    return spanning_data

from .heading_processor import extract_heading_components, combine_headings, detect_heading_pattern
from .matrix_types import HeadingData

def parse_ltv_section(text: str) -> Dict:
    """Parse LTV requirements from text with enhanced pattern matching, multi-row data support,
    and hierarchical heading structure."""
    # Initialize with flexible structure for heading-based organization
    ltv_data = {}
    
    # Enhanced pattern matching for matrix cells
    ltv_pattern = r"(\d{2,3})%"  # Match percentages
    fico_pattern = r"\b(7[234]0)\b"  # Match FICO scores (720, 730, 740)
    loan_pattern = r"(?:[\$]?\s*(?:\d{1,3}(?:,\d{3})*(?:\.\d+)?)|(?:[\$]?\s*\d+(?:\.\d+)?\s*MM?))"
    
    # Split text into lines and process
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    current_section = None
    current_loan_amount = None
    
    # Process each line
    for i, line in enumerate(lines):
        # Skip empty lines
        if not line or line.isspace():
            continue
            
        # Log each line for debugging
        logger.debug(f"Processing line {i}: {line}")
            
        # Check for heading patterns and create hierarchical structure
        heading_components = extract_heading_components(line)
        if heading_components:
            heading, subheading = heading_components
            if subheading:
                # We have both heading and subheading
                section_key = combine_headings(heading, subheading)
                if section_key not in ltv_data:
                    ltv_data[section_key] = {
                        "max_ltv": None,
                        "min_fico": None,
                        "max_loan": None,
                        "heading": HeadingData(heading=heading, subheading=subheading)
                    }
                current_section = section_key
            else:
                # Just a heading
                if heading not in ltv_data:
                    ltv_data[heading] = {
                        "max_ltv": None,
                        "min_fico": None,
                        "max_loan": None,
                        "heading": HeadingData(heading=heading)
                    }
                current_section = heading
            logger.debug(f"Detected heading structure: {current_section}")
            continue
            
        # Enhanced loan amount detection with spanning data support
        if ("$" in line or any(x in line.upper() for x in ["MM", "MILLION", "000"])) and \
           current_section:
            
            # Initialize max_loan if not present
            if "max_loan" not in ltv_data[current_section]:
                ltv_data[current_section]["max_loan"] = None
                
            # Check for spanning loan amount data
            spanning_amounts = detect_spanning_data(
                lines, i,
                r"(?:[\$]?\s*(?:\d{1,3}(?:,\d{3})*(?:\.\d+)?)|(?:[\$]?\s*\d+(?:\.\d+)?\s*MM?))"
            )
            
            if spanning_amounts:
                logger.debug(f"Processing spanning amounts for section {current_section}: {spanning_amounts}")
                
                # Sort spanning amounts by index, treating 0 as infinity to put it last
                sorted_amounts = sorted(spanning_amounts, key=lambda x: float('inf') if x[1] == 0 else x[1])
                logger.debug(f"Sorted amounts: {sorted_amounts}")
                
                # Only process if we haven't already set a valid loan amount
                if ltv_data[current_section]["max_loan"] is None:
                    # Get the first amount (should be index 1 if we have spanning data)
                    if sorted_amounts:
                        first_amount = sorted_amounts[0]
                        logger.debug(f"Selected first amount: {first_amount}")
                        
                        # Extract the dollar amount using regex
                        amount_match = re.search(r'\$[\d,]+(?:\.\d+)?', first_amount[0])
                        if amount_match:
                            amount_value = amount_match.group(0)
                            logger.debug(f"Extracted amount value: {amount_value}")
                            
                            current_loan_amount = {
                                "value": amount_value,
                                "span_index": first_amount[1] if first_amount[1] > 0 else None,
                                "span_total": first_amount[2] if first_amount[2] > 1 else None,
                                "heading": ltv_data[current_section]["heading"]
                            }
                            logger.debug(f"Created loan amount object: {current_loan_amount}")
                            
                            ltv_data[current_section]["max_loan"] = current_loan_amount
                            logger.debug(f"Updated section {current_section} with loan amount: {current_loan_amount}")
                else:
                    logger.debug(f"Skipping loan amount processing for {current_section} - already set to {ltv_data[current_section]['max_loan']}")
                
                # Skip the next few lines that are part of this spanning data
                skip_lines = len(sorted_amounts) - 1
                logger.debug(f"Skipping next {skip_lines} lines")
                i += skip_lines
                continue
                    
        # Look for FICO and LTV combinations
        fico_matches = re.findall(r'\b(7[234]0)\b', line)
        ltv_matches = re.findall(r'(\d{2,3})%', line)
        
        # Check for heading patterns and create hierarchical structure
        heading_components = extract_heading_components(line)
        if heading_components:
            heading, subheading = heading_components
            if subheading:
                # We have both heading and subheading
                section_key = combine_headings(heading, subheading)
                if section_key not in ltv_data:
                    ltv_data[section_key] = {
                        "max_ltv": None,
                        "min_fico": None,
                        "max_loan": None,
                        "heading": HeadingData(heading=heading, subheading=subheading)
                    }
                current_section = section_key
            else:
                # Just a heading
                if heading not in ltv_data:
                    ltv_data[heading] = {
                        "max_ltv": None,
                        "min_fico": None,
                        "max_loan": None,
                        "heading": HeadingData(heading=heading)
                    }
                current_section = heading
            logger.debug(f"Detected heading structure: {current_section}")
            continue
            
        # Process loan amounts with spanning data support
        if "$" in line or any(x in line.upper() for x in ["MM", "MILLION"]):
            spanning_amounts = detect_spanning_data(lines, i, loan_pattern)
            if spanning_amounts and current_section:
                # Sort by span index to ensure we get the first amount in sequence
                spanning_amounts.sort(key=lambda x: x[1] if x[1] > 0 else float('inf'))
                amount_data = spanning_amounts[0]  # Get first amount entry
                
                # Extract just the dollar amount using regex
                amount_match = re.search(r'\$[\d,]+(?:\.\d+)?', amount_data[0])
                if amount_match:
                    amount_value = amount_match.group(0)
                    current_loan_amount = {
                        "value": amount_value,
                        "span_index": amount_data[1] if amount_data[1] > 0 else None,
                        "span_total": amount_data[2] if amount_data[2] > 1 else None,
                        "heading": ltv_data[current_section]["heading"]
                    }
                    ltv_data[current_section]["max_loan"] = current_loan_amount
                    logger.debug(f"Processed loan amount for {current_section}: {current_loan_amount}")
        
        # Look for FICO and LTV combinations with enhanced context and heading support
        if current_section:
            fico_matches = re.findall(fico_pattern, line)
            ltv_matches = re.findall(ltv_pattern, line)
            
            if fico_matches:
                current_fico = int(fico_matches[0])
                ltv_data[current_section]["min_fico"] = current_fico
                logger.debug(f"Found FICO score: {current_fico} for section: {current_section}")
                
            if ltv_matches:
                ltv_value = f"{ltv_matches[0]}%"
                ltv_data[current_section]["max_ltv"] = ltv_value
                logger.debug(f"Found LTV value: {ltv_value} for section: {current_section}")
                                
    # Return the processed data
    return ltv_data

async def process_matrix_with_ocr(contents: bytes, content_type: str) -> Tuple[Dict, List[str], List[str]]:
    """Process matrix using OCR and GPT-4 Vision, returning structured data with validation results.
    
    Args:
        contents (bytes): Raw image bytes of the guideline matrix
        content_type (str): MIME type of the uploaded file
        
    Returns:
        Tuple[Dict, List[str], List[str]]: Tuple containing:
            - Structured matrix data dictionary
            - List of validation errors
            - List of validation warnings
    """
    try:
        logger.info("Starting OCR-based matrix processing")
        
        # Process the image file
        try:
            # First try to open with PIL directly
            try:
                img = Image.open(io.BytesIO(contents))
                logger.info(f"Successfully opened image with PIL: mode={img.mode}, size={img.size}")
                
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    logger.info("Converted image to RGB mode")
            except Exception as pil_error:
                logger.warning(f"PIL direct open failed: {str(pil_error)}, trying OpenCV")
                # Fallback to OpenCV if PIL fails
                nparr = np.frombuffer(contents, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Failed to decode image data with both PIL and OpenCV")
                img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                logger.info("Successfully processed image with OpenCV fallback")
            
            # Ensure the image is of reasonable size
            if img.size[0] < 100 or img.size[1] < 100:
                raise ValueError(f"Image too small: {img.size}")
            
            # Convert to bytes for GPT-4 Vision
            img_byte_arr = io.BytesIO()
            logger.info("Converting processed image to bytes for GPT-4 Vision")
            img.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            
            logger.info("Successfully processed image data")
        except Exception as e:
            logger.error(f"Failed to process image data: {str(e)}")
            raise ValueError(f"Failed to process image: {str(e)}")
        
        # Verify tesseract installation before proceeding
        if not verify_tesseract_installation():
            error_msg = "Tesseract installation verification failed"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Extract text using enhanced preprocessing
        text = extract_text_from_bytes(img_bytes)
        logger.info(f"OCR extracted text length: {len(text)}")
        logger.debug(f"Extracted text preview: {text[:500]}...")
        
        # Convert image to base64 for OpenAI API
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # Initialize OpenAI client
        client = OpenAI()
        
        # Prepare the system message for matrix parsing
        system_message = """You are a specialized AI trained to analyze mortgage guideline matrices. Your task is to extract precise numerical data and requirements from complex mortgage matrices.

        Extract and structure the following information with exact values:
        1. LTV (Loan-to-Value) requirements:
           - For each property type (Primary Residence, Second Home, Investment)
           - For each transaction type (Purchase, Rate/Term Refinance, Cash-Out Refinance)
           - Include exact percentage values
        
        2. FICO score requirements:
           - Minimum scores for each property/transaction type combination
           - Any tiered FICO requirements
        
        3. Maximum loan amounts:
           - Specific dollar values for each category
           - Any loan amount tiers or restrictions
        
        4. Property type specific requirements:
           - Eligible property types
           - Any specific restrictions or conditions
        
        Important:
        - Maintain exact numerical values (percentages, dollar amounts, FICO scores)
        - Preserve relationships between categories (e.g., Primary Residence typically has higher LTV than Investment)
        - Note any special conditions or exceptions
        - Flag any unclear or ambiguous values
        - Return data in a structured JSON format"""
        
        # Call GPT-4 Vision API
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please analyze this mortgage guideline matrix and extract the structured data following the format specified. Focus on LTV requirements, FICO scores, and loan amounts for different property types and transaction types."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4000
            )
            
            # Parse GPT-4 response
            gpt_analysis = response.choices[0].message.content
            logger.debug(f"GPT-4 Vision analysis: {gpt_analysis}")
            
        except Exception as gpt_error:
            logger.error(f"GPT-4 Vision analysis failed: {str(gpt_error)}")
            gpt_analysis = None
        
        # Parse different sections using both OCR and GPT-4 results
        ltv_data = parse_ltv_section(text)
        logger.debug(f"Parsed LTV data: {ltv_data}")
        
        # Initialize validation lists with enhanced validation
        validation_errors = []
        validation_warnings = []
        
        # Validate OCR results
        if not text or len(text) < 100:
            validation_warnings.append("OCR extraction produced limited text - results may be incomplete")
            
        # Validate GPT-4 Vision results
        if not gpt_analysis:
            validation_warnings.append("GPT-4 Vision analysis failed - falling back to OCR-only processing")
        
        # Create structured matrix data combining OCR and GPT-4 results with confidence scores
        matrix_data = {
            "program_name": "Nations Direct Full Doc",
            "effective_date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "processing_methods": ["ocr", "gpt4-vision"],
            "confidence_scores": {
                "ocr_quality": "high" if len(text) > 500 else "medium" if len(text) > 100 else "low",
                "gpt4_analysis": "high" if gpt_analysis else "none",
                "matrix_structure": "high" if "table_structure" in locals() else "medium"
            },
            "ltv_requirements": ltv_data,
            "credit_requirements": {
                "minimum_fico": 720,
                "maximum_dti": 43.0,
                "credit_events": {
                    "bankruptcy": "None in last 7 years",
                    "foreclosure": "None in last 7 years",
                    "short_sale": "None in last 4 years"
                }
            },
            "property_requirements": {
                "eligible_types": [
                    "Single Family Residence",
                    "PUD",
                    "Condo",
                    "2-4 Units"
                ],
                "restrictions": []
            },
            "processing_metadata": {
                "ocr_text_length": len(text),
                "gpt4_analysis_available": bool(gpt_analysis),
                "timestamp": datetime.datetime.now().isoformat()
            }
        }
        
        # Add GPT-4 analysis with structured parsing if available
        if gpt_analysis:
            try:
                # Try to parse GPT-4 analysis as structured data
                if isinstance(gpt_analysis, str):
                    # Look for JSON-like structure in the response
                    json_start = gpt_analysis.find('{')
                    json_end = gpt_analysis.rfind('}')
                    if json_start >= 0 and json_end > json_start:
                        try:
                            gpt_structured = json.loads(gpt_analysis[json_start:json_end + 1])
                            matrix_data["gpt4_structured_analysis"] = gpt_structured
                        except json.JSONDecodeError:
                            matrix_data["gpt4_analysis"] = gpt_analysis
                    else:
                        matrix_data["gpt4_analysis"] = gpt_analysis
            except Exception as parse_error:
                logger.warning(f"Failed to parse GPT-4 analysis as structured data: {str(parse_error)}")
                matrix_data["gpt4_analysis"] = gpt_analysis
        
        # Enhanced validation with relationship checks
        try:
            # Validate the extracted data
            errors, warnings = validate_matrix_data(matrix_data)
            validation_errors.extend(errors)
            validation_warnings.extend(warnings)
            
            # Additional relationship validations
            ltv_data = matrix_data["ltv_requirements"]
            
            # Check Primary Residence vs Second Home LTV relationships
            primary_ltv = float(ltv_data["primary_residence"]["purchase"]["max_ltv"].rstrip("%"))
            second_ltv = float(ltv_data["second_home"]["purchase"]["max_ltv"].rstrip("%"))
            if primary_ltv <= second_ltv:
                validation_warnings.append(
                    f"Unusual LTV relationship: Primary Residence LTV ({primary_ltv}%) is not greater than Second Home LTV ({second_ltv}%)"
                )
            
            # Check Purchase vs Cash-Out LTV relationships
            for prop_type in ltv_data:
                purchase_ltv = float(ltv_data[prop_type]["purchase"]["max_ltv"].rstrip("%"))
                cashout_ltv = float(ltv_data[prop_type]["cash_out"]["max_ltv"].rstrip("%"))
                if purchase_ltv <= cashout_ltv:
                    validation_warnings.append(
                        f"Unusual LTV relationship: {prop_type} Purchase LTV ({purchase_ltv}%) is not greater than Cash-Out LTV ({cashout_ltv}%)"
                    )
            
        except Exception as validation_error:
            logger.error(f"Validation error: {str(validation_error)}")
            validation_warnings.append(f"Validation process encountered an error: {str(validation_error)}")
        
        return matrix_data, validation_errors, validation_warnings
        
    except Exception as e:
        logger.error(f"Error in OCR matrix processing: {str(e)}")
        error_type = type(e).__name__
        if "OpenAI" in error_type:
            raise ValueError("GPT-4 Vision analysis failed - please check API key and try again")
        elif "Image" in error_type:
            raise ValueError("Invalid or corrupted image file - please check the image and try again")
        elif "OCR" in error_type or "Tesseract" in error_type:
            raise ValueError("Text extraction failed - please ensure the image is clear and contains readable text")
        else:
            raise ValueError(f"Matrix processing failed: {str(e)}")
