from datetime import datetime
from typing import Dict, Optional
import uuid
import json
from logging import getLogger

logger = getLogger(__name__)

class InMemoryDB:
    def __init__(self):
        logger.info("Using in-memory storage")
        self.agents: Dict[str, dict] = {}
        self.tokens: Dict[str, str] = {}  # token -> agent_id mapping
        
    def register_agent(self, agent_data: dict) -> tuple[str, str]:
        agent_id = str(uuid.uuid4())
        marketplace_token = str(uuid.uuid4())
        
        # Add registration metadata
        agent_data["registration_timestamp"] = datetime.utcnow()
        agent_data["agent_id"] = agent_id
        
        # Debug log before storage
        logger.info(f"Storing agent data with input schema: {agent_data.get('input_schema')}")
        
        # Store the agent data with its embedding
        self.agents[agent_id] = agent_data
        self.tokens[marketplace_token] = agent_id
        
        # Verify storage
        stored = self.agents[agent_id]
        logger.info(f"Stored agent data input schema: {stored.get('input_schema')}")
        
        # Log storage details
        logger.info(f"Stored agent data for {agent_data['agent_name']}")
        if "embedding" in agent_data:
            logger.info(f"Embedding stored with length: {len(agent_data['embedding'])}")
        else:
            logger.warning("No embedding found in agent data")
        
        logger.info(f"Registered agent {agent_data['agent_name']} with ID {agent_id}")
        logger.debug(f"Current agents in DB: {list(self.get_all_agent_ids())}")
        
        return agent_id, marketplace_token
    
    def get_all_agent_ids(self) -> list[str]:
        """Get all agent IDs from storage."""
        return list(self.agents.keys())

    def get_agent(self, agent_id: str) -> Optional[dict]:
        """Get agent by ID from storage."""
        return self.agents.get(agent_id)
    
    def get_agent_by_token(self, token: str) -> Optional[dict]:
        """Get agent by token from storage."""
        agent_id = self.tokens.get(token)
        return self.agents.get(agent_id) if agent_id else None
    
    def list_agents(self) -> list[dict]:
        """List all agents from storage."""
        return list(self.agents.values())
    
    def verify_token(self, token: str) -> bool:
        """Verify token exists in storage."""
        logger.info(f"Verifying token: {token}")
        logger.debug(f"Available tokens: {list(self.tokens.keys())}")
        result = token in self.tokens
        logger.info(f"Token verification result: {result}")
        return result
        
    def get_token_for_agent(self, agent_id: str) -> Optional[str]:
        """Get the marketplace token for a given agent ID."""
        logger.info(f"Looking up token for agent: {agent_id}")
        # Find the token that maps to this agent_id
        for token, aid in self.tokens.items():
            if aid == agent_id:
                logger.info(f"Found token for agent {agent_id}")
                return token
        logger.warning(f"No token found for agent {agent_id}")
        return None

# Global instance
db = InMemoryDB()
