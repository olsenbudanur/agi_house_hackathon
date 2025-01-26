import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database import db
from app.models import AgentRegistration, ProviderDetails
import json

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
        }
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
        }
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
    assert data["result"]["_agent_selection_metadata"]["confidence"] in ["high", "medium", "low"]

def test_find_and_invoke_summarization(mock_agents):
    """Test that the correct agent is selected for summarization"""
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
    assert data["result"]["_agent_selection_metadata"]["confidence"] in ["high", "medium", "low"]

def test_find_and_invoke_no_suitable_agent(mock_agents):
    """Test error handling when no suitable agent is found"""
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
    response = client.post(
        "/api/v1/agents/find-and-invoke",
        json={
            "query": "Analyze sentiment but with invalid parameters",
            "max_tokens": 1000,
            "temperature": 0.7,
            "additional_context": "force_invalid_schema=true"  # Special test flag
        }
    )
    
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert "schema validation failed" in data["error"]["message"].lower()

def test_find_and_invoke_openai_error(mock_agents):
    """Test error handling when OpenAI API fails"""
    response = client.post(
        "/api/v1/agents/find-and-invoke",
        json={
            "query": "This should trigger an OpenAI error",
            "max_tokens": 999999,  # Invalid value to trigger error
            "temperature": 999.9   # Invalid value to trigger error
        }
    )
    
    assert response.status_code == 500
    data = response.json()
    assert data["status"] == "error"
    assert data["error"]["code"] == "OPENAI_API_ERROR"
