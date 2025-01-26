from fastapi import APIRouter, HTTPException, Header, Query
from typing import List, Optional, Tuple
import numpy as np
from ..models import (
    AgentRegistration, 
    AgentResponse, 
    HealthCheck, 
    AgentCapabilities, 
    AgentStatus, 
    ProviderDetails,
    InvocationRequest,
    InvocationResponse
)
from ..services.embedding import model as embedding_model
from ..database import db
from ..services.embedding import compute_agent_embedding
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])

@router.post("/register", response_model=AgentResponse)
async def register_agent(agent: AgentRegistration):
    try:
        # Convert model to dict and derive endpoints
        agent_dict = agent.model_dump()
        base_url = str(agent_dict["base_url"]).rstrip("/")  # Remove trailing slash if present
        
        # Derive standard endpoints
        agent_dict["health_check_endpoint"] = f"{base_url}/health"
        agent_dict["invocation_endpoint"] = f"{base_url}/invoke"
        capabilities_url = f"{base_url}/capabilities"
        
        # Fetch and validate capabilities
        try:
            import requests
            logger.info(f"Fetching capabilities from: {capabilities_url}")
            response = requests.get(capabilities_url, timeout=5)
            response.raise_for_status()
            capabilities = response.json()
            
            # Validate capabilities match
            if capabilities.get("name") != agent_dict["agent_name"]:
                logger.warning(f"Agent name mismatch: registered={agent_dict['agent_name']}, endpoint={capabilities.get('name')}")
            
            if capabilities.get("version") != agent_dict["version"]:
                logger.warning(f"Version mismatch: registered={agent_dict['version']}, endpoint={capabilities.get('version')}")
            
            # Update with validated schemas and capabilities
            agent_dict["input_schema"] = capabilities.get("input_schema", agent_dict["input_schema"])
            agent_dict["output_schema"] = capabilities.get("output_schema", agent_dict["output_schema"])
            agent_dict["capabilities"] = capabilities.get("capabilities", agent_dict["capabilities"])
            
            # Store rate limits if provided
            if "rate_limits" in capabilities:
                agent_dict["rate_limits"] = capabilities["rate_limits"]
            
            logger.info("Successfully validated agent capabilities")
            
        except Exception as e:
            logger.error(f"Failed to fetch or validate capabilities: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to validate agent capabilities: {str(e)}"
            )
        
        # Compute embedding with detailed logging
        logger.info(f"Computing embedding for agent: {agent_dict['agent_name']}")
        try:
            embedding = compute_agent_embedding(agent_dict)
            if not embedding:
                raise ValueError("Empty embedding generated")
            logger.info(f"Successfully generated embedding of length {len(embedding)}")
            agent_dict["embedding"] = embedding
        except Exception as e:
            logger.error(f"Failed to compute embedding: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to compute embedding: {str(e)}"
            )
        
        # Register agent with embedding
        agent_id, marketplace_token = db.register_agent(agent_dict)
        
        # Verify embedding was stored
        stored_agent = db.get_agent(agent_id)
        if not stored_agent.get("embedding"):
            logger.error("Embedding was not stored properly")
            raise HTTPException(
                status_code=500,
                detail="Failed to store agent embedding"
            )
        
        logger.info(f"Successfully registered agent {agent_dict['agent_name']} with embedding")
        return AgentResponse(
            agent_id=agent_id,
            registration_timestamp=datetime.utcnow(),
            marketplace_token=marketplace_token
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to register agent: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to register agent: {str(e)}"
        )

@router.get("/search", response_model=List[AgentCapabilities])
async def semantic_search(
    query: str,
    threshold: float = Query(0.3, ge=0.0, le=1.0, description="Minimum similarity threshold")
) -> List[AgentCapabilities]:
    """
    Search for agents using semantic similarity with the query.
    Returns agents sorted by relevance, filtered by similarity threshold.
    """
    if not query:
        return []
    
    try:
        logger.info(f"Processing semantic search query: {query}")
        
        # Generate query embedding
        query_embedding = embedding_model.encode(query).tolist()
        logger.info(f"Generated query embedding of length {len(query_embedding)}")
        
        # Get all agents
        agents = db.list_agents()
        logger.info(f"Found {len(agents)} agents in database")
        results: List[Tuple[dict, float]] = []
        
        for agent in agents:
            agent_emb = agent.get("embedding")
            if not agent_emb:
                logger.warning(f"No embedding found for agent {agent.get('agent_name', 'unknown')}")
                continue
            
            logger.info(f"Processing agent: {agent.get('agent_name')} with embedding length {len(agent_emb)}")
                
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, agent_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(agent_emb)
            )
            
            similarity = float(similarity)  # Convert to Python float
            logger.info(f"Similarity score for {agent.get('agent_name')}: {similarity:.4f}")
            
            if similarity >= threshold:
                logger.info(f"Adding agent {agent.get('agent_name')} to results with score {similarity:.4f}")
                results.append((agent, similarity))
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Convert to AgentCapabilities models
        return [
            AgentCapabilities(
                name=agent["agent_name"],
                version=agent["version"],
                capabilities=agent["capabilities"],
                input_schema=agent["input_schema"],
                output_schema=agent["output_schema"],
                rate_limits={"requests_per_second": 10, "burst_limit": 20},
                description=agent["description"],
                provider_details=ProviderDetails(**agent["provider_details"]),
                health_check_endpoint=agent["health_check_endpoint"],
                invocation_endpoint=agent["invocation_endpoint"],
                agent_id=agent.get("agent_id"),
                registration_timestamp=agent.get("registration_timestamp"),
                embedding=agent.get("embedding")
            )
            for agent, _ in results
        ]
    except Exception as e:
        logger.error(f"Semantic search failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to perform semantic search: {str(e)}"
        )

@router.get("/list", response_model=List[AgentCapabilities])
async def list_agents(
    capability: Optional[str] = Query(None, description="Filter agents by capability"),
    min_version: Optional[str] = Query(None, description="Filter agents by minimum version"),
    status: Optional[AgentStatus] = Query(None, description="Filter agents by health status")
):
    """List available agents with optional filtering."""
    from packaging import version
    from logging import getLogger
    
    logger = getLogger(__name__)
    agents = db.list_agents()
    logger.info(f"Total agents in DB: {len(agents)}")
    filtered_agents = []
    
    for agent in agents:
        try:
            logger.debug(f"Processing agent: {agent.get('agent_name')} ({agent.get('agent_id')})")
            
            # Apply capability filter
            if capability:
                agent_capabilities = set(agent.get("capabilities", []))
                if capability not in agent_capabilities:
                    logger.debug(f"Agent {agent.get('agent_name')} filtered out by capability: {capability} not in {agent_capabilities}")
                    continue
            
            # Apply version filter
            if min_version:
                try:
                    agent_version = version.parse(agent.get("version", "0.0.0"))
                    min_ver = version.parse(min_version)
                    logger.debug(f"Comparing versions: {agent_version} >= {min_ver}")
                    if agent_version < min_ver:
                        logger.debug(f"Agent {agent.get('agent_name')} filtered out by version")
                        continue
                except version.InvalidVersion as e:
                    logger.warning(f"Invalid version for agent {agent.get('agent_name')}: {e}")
                    continue
            
            # Check agent health
            agent_health = AgentStatus.HEALTHY  # In production, we would check the health endpoint
            if status and status != agent_health:
                logger.debug(f"Agent {agent.get('agent_name')} filtered out by status")
                continue
            
            # Convert provider details
            provider_details = ProviderDetails(**agent["provider_details"])
            
            # Create agent capabilities response
            agent_capabilities = AgentCapabilities(
                name=agent["agent_name"],
                version=agent["version"],
                capabilities=agent["capabilities"],
                input_schema=agent["input_schema"],
                output_schema=agent["output_schema"],
                rate_limits={"requests_per_second": 10, "burst_limit": 20},
                description=agent["description"],
                provider_details=provider_details,
                health_check_endpoint=agent["health_check_endpoint"],
                invocation_endpoint=agent["invocation_endpoint"],
                agent_id=agent.get("agent_id"),
                registration_timestamp=agent.get("registration_timestamp"),
                embedding=agent.get("embedding")  # Include embedding in response
            )
            filtered_agents.append(agent_capabilities)
            logger.debug(f"Added agent {agent['agent_name']} to filtered results")
            
        except Exception as e:
            logger.error(f"Failed to process agent {agent.get('agent_name')}: {str(e)}")
            continue
    
    logger.info(f"Returning {len(filtered_agents)} filtered agents")
    return filtered_agents

@router.get("/{agent_id}/health", response_model=HealthCheck)
async def get_agent_health(agent_id: str):
    agent = db.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    # In a real implementation, we would make a request to the agent's health check endpoint
    return HealthCheck(
        status="healthy",
        last_updated=datetime.utcnow(),
        metrics={"uptime": 100, "success_rate": 0.99, "average_response_time": 0.1}
    )

@router.post("/{agent_id}/invoke", response_model=InvocationResponse)
async def invoke_agent(
    agent_id: str,
    request: InvocationRequest,
    marketplace_token: Optional[str] = Header(None)
):
    """Invoke an agent with the given input."""
    from logging import getLogger
    logger = getLogger(__name__)
    
    logger.debug(f"Attempting to invoke agent with ID: {agent_id}")
    agent = db.get_agent(agent_id)
    if not agent:
        logger.error(f"Agent not found with ID: {agent_id}")
        logger.debug(f"Available agent IDs: {db.get_all_agent_ids()}")
        raise HTTPException(status_code=404, detail=f"Agent not found with ID: {agent_id}")

    # Verify marketplace token if provided
    if marketplace_token and not db.verify_token(marketplace_token):
        raise HTTPException(
            status_code=401,
            detail="Invalid marketplace token"
        )

    try:
        import requests
        from jsonschema import validate, ValidationError

        # Get the agent's invocation endpoint
        invocation_endpoint = agent.get("invocation_endpoint")
        if not invocation_endpoint:
            raise HTTPException(
                status_code=500,
                detail="Agent invocation endpoint not found"
            )

        logger.info(f"Forwarding request to agent endpoint: {invocation_endpoint}")
        
        # Forward the request to the agent
        try:
            invocation_res = requests.post(
                invocation_endpoint,
                json={
                    "input": request.input,
                    "trace_id": request.trace_id,
                    "timeout": request.timeout
                },
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                timeout=request.timeout or 30
            )
            
            # Check if the request was successful
            invocation_res.raise_for_status()
            
            # Parse the response
            agent_resp = invocation_res.json()
            logger.info(f"Agent response received: {agent_resp}")
            
            # Convert to InvocationResponse model
            return InvocationResponse(**agent_resp)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to invoke agent: {str(e)}")
            return InvocationResponse(
                status="error",
                error={
                    "code": "INVOCATION_ERROR",
                    "message": "Failed to invoke agent",
                    "details": {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "agent_id": agent_id
                    }
                },
                trace_id=request.trace_id
            )
            
    except Exception as e:
        logger.error(f"Error in invoke_agent: {str(e)}")
        return InvocationResponse(
            status="error",
            error={
                "code": "INTERNAL_ERROR",
                "message": str(e),
                "details": {
                    "error_type": type(e).__name__,
                    "agent_id": agent_id
                }
            },
            trace_id=request.trace_id
        )
