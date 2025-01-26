import json
import logging
import requests
import time
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MARKETPLACE_URL = "http://127.0.0.1:8000/api/v1/agents"  # Using IP address instead of localhost with API version prefix
AGENTS = [
    {
        "agent_name": "LoanDocProcessor",
        "description": "Processes and verifies loan documentation and initial applications",
        "version": "1.0.0",
        "capabilities": ["document-verification", "application-review", "borrower-info-collection", "initial-processing"],
        "base_url": "http://127.0.0.1:8090",
        "provider_details": {
            "organization": "Mock Agents Inc",
            "contact_email": "docs@mockagents.com",
            "website": "https://mockagents.com"
        }
    },
    {
        "agent_name": "UnderwritingAnalyzer",
        "description": "Analyzes loan applications and performs underwriting assessments",
        "version": "1.0.0",
        "capabilities": ["credit-verification", "loan-quality-check", "risk-assessment", "conditional-approval"],
        "base_url": "http://127.0.0.1:8091",
        "provider_details": {
            "organization": "Mock Agents Inc",
            "contact_email": "underwriting@mockagents.com",
            "website": "https://mockagents.com"
        }
    },
    {
        "agent_name": "ClosingCoordinator",
        "description": "Coordinates closing process and document preparation",
        "version": "1.0.0",
        "capabilities": ["closing-doc-prep", "closing-scheduling", "clear-to-close", "final-approval"],
        "base_url": "http://127.0.0.1:8092",
        "provider_details": {
            "organization": "Mock Agents Inc",
            "contact_email": "closing@mockagents.com",
            "website": "https://mockagents.com"
        }
    },
    {
        "agent_name": "QualityController",
        "description": "Performs quality control reviews and compliance checks",
        "version": "1.0.0",
        "capabilities": ["post-closing-review", "compliance-check", "qc-audit", "investor-requirements"],
        "base_url": "http://127.0.0.1:8093",
        "provider_details": {
            "organization": "Mock Agents Inc",
            "contact_email": "qc@mockagents.com",
            "website": "https://mockagents.com"
        }
    },
    {
        "agent_name": "FundingManager",
        "description": "Manages loan funding and settlement processes",
        "version": "1.0.0",
        "capabilities": ["fund-disbursement", "settlement-coordination", "wire-transfer", "post-funding-review"],
        "base_url": "http://127.0.0.1:8094",
        "provider_details": {
            "organization": "Mock Agents Inc",
            "contact_email": "funding@mockagents.com",
            "website": "https://mockagents.com"
        }
    },
    {
        "agent_name": "MockImageProcessor",
        "description": "A mock AI agent that processes images for classification and description",
        "version": "1.0.0",
        "capabilities": ["image-processing", "image-classification", "image-description"],
        "base_url": "http://127.0.0.1:8084",
        "provider_details": {
            "organization": "Mock Agents Inc",
            "contact_email": "image@mockagents.com",
            "website": "https://mockagents.com"
        }
    },
    {
        "agent_name": "MockSpeechProcessor",
        "description": "A mock AI agent that processes speech for transcription and language detection",
        "version": "1.0.0",
        "capabilities": ["speech-to-text", "language-detection"],
        "base_url": "http://127.0.0.1:8085",
        "provider_details": {
            "organization": "Mock Agents Inc",
            "contact_email": "speech@mockagents.com",
            "website": "https://mockagents.com"
        }
    }
]

def get_agent_schemas(agent: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch input/output schemas from agent's capabilities endpoint"""
    try:
        url = f"{agent['base_url']}/capabilities"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        return {
            **agent,
            "input_schema": data["input_schema"],
            "output_schema": data["output_schema"]
        }
    except Exception as e:
        logger.error(f"Error fetching schemas for {agent['agent_name']}: {str(e)}")
        return None

def register_agent(agent: Dict[str, Any]) -> bool:
    """Register an agent with the marketplace"""
    try:
        # First, validate we have the required schemas
        if not agent.get("input_schema") or not agent.get("output_schema"):
            logger.error(f"Missing schemas for agent {agent['agent_name']}")
            return False

        url = f"{MARKETPLACE_URL}/register"
        logger.info(f"Full registration URL: {url}")
        logger.info(f"Agent data being sent: {json.dumps(agent, indent=2)}")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Debug log the request
        logger.info(f"Attempting to register {agent['agent_name']} at {url}")
        logger.debug(f"Request headers: {headers}")
        logger.debug(f"Request body: {json.dumps(agent, indent=2)}")
        
        # Make the request with retries
        for attempt in range(3):
            try:
                response = requests.post(
                    url,
                    json=agent,
                    headers=headers,
                    timeout=30,
                    verify=False  # For local development
                )
                response_text = response.text
                logger.info(f"Response status: {response.status_code}")
                logger.info(f"Response text: {response_text}")
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"Successfully registered {agent['agent_name']}")
                    logger.info(f"Agent ID: {data['agent_id']}")
                    logger.info(f"Marketplace Token: {data['marketplace_token']}")
                    return True
                    
                logger.error(f"Failed to register {agent['agent_name']}: {response.status_code}")
                try:
                    error_data = json.loads(response_text)
                    logger.error(f"Error details: {json.dumps(error_data, indent=2)}")
                except:
                    logger.error(f"Raw error response: {response_text}")
                    
                if attempt < 2:  # Don't sleep on last attempt
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed on attempt {attempt + 1}: {str(e)}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    
        return False
        
    except Exception as e:
        logger.error(f"Error registering {agent['agent_name']}: {str(e)}")
        return False

def main():
    """Main registration process"""
    # First, fetch schemas for all agents
    agents_with_schemas = []
    for agent in AGENTS:
        result = get_agent_schemas(agent)
        if result:
            agents_with_schemas.append(result)
    
    if not agents_with_schemas:
        logger.error("Failed to fetch schemas for any agents")
        return
    
    logger.info(f"Successfully fetched schemas for {len(agents_with_schemas)} agents")
    
    # Register each agent
    registration_results = []
    for agent in agents_with_schemas:
        result = register_agent(agent)
        registration_results.append(result)
    
    # Summary
    success_count = sum(1 for result in registration_results if result)
    logger.info(f"\nRegistration Summary:")
    logger.info(f"Total agents: {len(AGENTS)}")
    logger.info(f"Successfully registered: {success_count}")
    logger.info(f"Failed: {len(AGENTS) - success_count}")

if __name__ == "__main__":
    main()
