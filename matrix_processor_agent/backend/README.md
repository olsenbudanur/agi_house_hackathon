# Insurance Matrix Processor Backend

FastAPI backend for processing insurance guideline matrices using OCR and LLM techniques.

## Features
- Image-based matrix processing
- Advanced OCR with preprocessing
- Structured data validation
- Comprehensive API endpoints
- Confidence scoring system

## Prerequisites

### System Requirements
```bash
# Install Tesseract OCR
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng
```

### Environment Variables
Create a `.env` file in the backend directory with:
```bash
OPENAI_API_KEY=your_openai_key_here
TESSDATA_PREFIX=/usr/share/tesseract-ocr/tessdata
TESSERACT_CMD=/usr/bin/tesseract
PATH="/usr/bin:${PATH}"
```

## Local Development Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd matrix-processor/backend
```

2. Install dependencies:
```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install --without dev
```

3. Start the development server:
```bash
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

4. Visit http://localhost:8000/docs to test the API endpoints

## Testing

Run the test suite:
```bash
poetry run python -m pytest
```

## API Endpoints

### POST /api/analyze
Upload and analyze a matrix image:
```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/matrix.png"
```

### GET /api/health
Check API health:
```bash
curl http://localhost:8000/api/health
```

## Validation System

The system implements comprehensive validation rules:

1. LTV Rules:
- Primary Residence LTV ≥ Second Home LTV
- Second Home LTV ≥ Investment Property LTV
- Purchase/Rate & Term LTV ≥ Cash-Out LTV

2. FICO Score Rules:
- Valid range (300-850)
- Consistent across transaction types
- Minimum requirements maintained

3. Loan Amount Rules:
- Decreasing tiers by property type
- Logical progression in amounts
- Valid range checks (10,000 to 100,000,000)

## OCR Processing Pipeline

1. Image Preprocessing:
   - Grayscale conversion
   - Adaptive thresholding
   - Table structure detection

2. Text Extraction:
   - Custom Tesseract configuration
   - Context-aware pattern matching
   - Confidence scoring

3. Validation:
   - Rule-based checks
   - Relationship validation
   - Error and warning generation

## Troubleshooting

1. Tesseract Issues:
   - Verify Tesseract installation: `tesseract --version`
   - Check TESSDATA_PREFIX path exists
   - Ensure proper permissions on Tesseract binary

2. API Connection:
   - Verify server is running on correct port
   - Check for firewall restrictions
   - Ensure correct Content-Type headers

## Notes

- The system is optimized for image input (PNG, JPEG, JPG, WebP)
- Validation rules are configurable in validation.py
- OCR preprocessing parameters can be tuned in ocr_processor.py
