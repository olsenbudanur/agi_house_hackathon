import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.database import db
from app.models import AgentRegistration, ProviderDetails, InvocationResponse
import json
import sys

# Mock the openai module
mock_openai = MagicMock()
sys.modules['openai'] = mock_openai

client = TestClient(app)

@pytest.fixture
def mock_agents():
    # Register two test agents
    sentiment_agent = {
        "agent_name": "SentimentAnalyzer",
        "description": "Analyzes sentiment in text",
        "version": "1.0.0",
        "capabilities": ["sentiment-analysis", "text-processing"],
        "base_url": "http://localhost:8083",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "analysis_type": {"type": "string", "enum": ["sentiment"]}
            },
            "required": ["text", "analysis_type"]
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "result": {"type": "string"},
                "confidence": {"type": "number"}
            }
        },
        "provider_details": {
            "organization": "Test Org",
            "contact_email": "test@example.com",
            "website": "https://example.com"
        },
        "embedding": [0.1] * 384  # Add mock embedding
    }
    
    summarizer_agent = {
        "agent_name": "TextSummarizer",
        "description": "Generates summaries of text",
        "version": "1.0.0",
        "capabilities": ["summarization", "text-processing"],
        "base_url": "http://localhost:8084",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "analysis_type": {"type": "string", "enum": ["summary"]}
            },
            "required": ["text", "analysis_type"]
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "result": {"type": "string"},
                "confidence": {"type": "number"}
            }
        },
        "provider_details": {
            "organization": "Test Org",
            "contact_email": "test@example.com",
            "website": "https://example.com"
        },
        "embedding": [0.1] * 384  # Add mock embedding
    }
    
    # Register agents
    db.register_agent(sentiment_agent)
    db.register_agent(summarizer_agent)
    
    yield
    
    # Clear database after tests
    db.agents.clear()
    db.tokens.clear()

def test_find_and_invoke_sentiment_analysis(mock_agents):
    """Test that the correct agent is selected for sentiment analysis"""
    # Mock OpenAI response
    mock_openai.ChatCompletion.create.return_value = MagicMock(
        choices=[MagicMock(
            message=MagicMock(
                content=json.dumps({
                    "selected_agent": "SentimentAnalyzer",
                    "reason": "This agent is capable of sentiment analysis",
                    "confidence": "high",
                    "input_parameters": {
                        "text": "I love this product!",
                        "analysis_type": "sentiment"
                    }
                })
            )
        )]
    )
    
    # Mock the invoke_agent function to avoid actual HTTP calls
    async def mock_invoke_agent(*args, **kwargs):
        return InvocationResponse(
            status="success",
            result={"sentiment": "positive"},
            trace_id="test-123"
        )
    
    with patch("app.routes.agents.invoke_agent", side_effect=mock_invoke_agent):
        response = client.post(
            "/api/v1/agents/find-and-invoke",
            json={
                "query": "Can you analyze the sentiment of this text: I love this product!",
                "max_tokens": 1000,
                "temperature": 0.7
            }
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "_agent_selection_metadata" in data["result"]
    assert data["result"]["_agent_selection_metadata"]["selected_agent"] == "SentimentAnalyzer"
    assert data["result"]["_agent_selection_metadata"]["confidence"] == "high"

def test_find_and_invoke_summarization(mock_agents):
    """Test that the correct agent is selected for summarization"""
    # Mock OpenAI response
    mock_openai.ChatCompletion.create.return_value = MagicMock(
        choices=[MagicMock(
            message=MagicMock(
                content=json.dumps({
                    "selected_agent": "TextSummarizer",
                    "reason": "This agent is capable of text summarization",
                    "confidence": "high",
                    "input_parameters": {
                        "text": "This is a long article about AI...",
                        "analysis_type": "summary"
                    }
                })
            )
        )]
    )
    
    # Mock the invoke_agent function to avoid actual HTTP calls
    async def mock_invoke_agent(*args, **kwargs):
        return InvocationResponse(
            status="success",
            result={"summary": "Summary of the article..."},
            trace_id="test-123"
        )
    
    with patch("app.routes.agents.invoke_agent", side_effect=mock_invoke_agent):
        response = client.post(
            "/api/v1/agents/find-and-invoke",
            json={
                "query": "Please summarize this text: This is a long article about AI...",
                "max_tokens": 1000,
                "temperature": 0.7
            }
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "_agent_selection_metadata" in data["result"]
    assert data["result"]["_agent_selection_metadata"]["selected_agent"] == "TextSummarizer"
    assert data["result"]["_agent_selection_metadata"]["confidence"] == "high"

def test_find_and_invoke_no_suitable_agent(mock_agents):
    """Test error handling when no suitable agent is found"""
    # Mock OpenAI response with error
    mock_openai.ChatCompletion.create.return_value = MagicMock(
        choices=[MagicMock(
            message=MagicMock(
                content=json.dumps({
                    "error": "No suitable agent found"
                })
            )
        )]
    )
    
    response = client.post(
        "/api/v1/agents/find-and-invoke",
        json={
            "query": "Can you generate an image of a cat?",
            "max_tokens": 1000,
            "temperature": 0.7
        }
    )
    
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert "error" in data
    assert "No suitable agent found" in data["error"]["message"]

def test_find_and_invoke_invalid_schema(mock_agents):
    """Test error handling when OpenAI generates invalid schema"""
    # Mock OpenAI response with invalid schema
    mock_openai.ChatCompletion.create.return_value = MagicMock(
        choices=[MagicMock(
            message=MagicMock(
                content=json.dumps({
                    "selected_agent": "SentimentAnalyzer",
                    "reason": "This agent can analyze sentiment",
                    "confidence": "high",
                    "input_parameters": {
                        "invalid_field": "This should fail schema validation",
                        "another_invalid": 123
                    }
                })
            )
        )]
    )
    
    response = client.post(
        "/api/v1/agents/find-and-invoke",
        json={
            "query": "Analyze sentiment but with invalid parameters",
            "max_tokens": 1000,
            "temperature": 0.7
        }
    )
    
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert data["error"]["code"] == "VALIDATION_ERROR"
    assert "missing required fields" in data["error"]["message"].lower()
    assert "text" in data["error"]["details"]["missing_fields"]
    assert "analysis_type" in data["error"]["details"]["missing_fields"]

def test_find_and_invoke_openai_error(mock_agents):
    """Test error handling when OpenAI API fails"""
    # Mock OpenAI error
    mock_openai.ChatCompletion.create.side_effect = Exception("OpenAI API error")
    
    response = client.post(
        "/api/v1/agents/find-and-invoke",
        json={
            "query": "This should trigger an OpenAI error",
            "max_tokens": 1000,
            "temperature": 0.7
        }
    )
    
    assert response.status_code == 500
    data = response.json()
    assert data["status"] == "error"
    assert data["error"]["code"] == "INTERNAL_ERROR"
    assert "OpenAI API error" in data["error"]["details"]["error_message"]
