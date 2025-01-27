# Matrix Processor Agent

A specialized agent for processing and extracting structured data from insurance guideline matrices using multimodal LLM approaches.

## Setup

### Prerequisites
- Python 3.12+
- Node.js 18+
- Tesseract OCR

### Environment Setup

1. Backend Setup
```bash
cd backend
# Copy environment template
cp .env.template .env
# Update .env with your OpenAI API key
# Copy pytest config template if running tests
cp pytest.ini.template pytest.ini

# Install dependencies
poetry install
```

2. Frontend Setup
```bash
cd frontend
# Copy environment template
cp .env.template .env
# Install dependencies
pnpm install
```

## Development

1. Start Backend
```bash
cd backend
poetry run uvicorn app.main:app --reload
```

2. Start Frontend
```bash
cd frontend
pnpm dev
```

## Configuration

### Required Environment Variables

Backend (.env):
- `OPENAI_API_KEY`: Your OpenAI API key
- `TESSDATA_PREFIX`: Path to Tesseract data
- `TESSERACT_CMD`: Path to Tesseract binary

Frontend (.env):
- `VITE_API_URL`: Backend API URL (default: http://localhost:8000)

## Testing

```bash
cd backend
poetry run pytest
```

## License
MIT
