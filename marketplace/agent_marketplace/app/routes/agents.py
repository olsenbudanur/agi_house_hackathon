from fastapi import APIRouter, HTTPException, Header, Query
from typing import List, Optional, Tuple
import numpy as np
import os
import openai
import jsonschema
from jsonschema import validate, ValidationError
import logging
import json
import uuid
from ..models import (
    AgentRegistration, 
    AgentResponse, 
    HealthCheck, 
    AgentCapabilities, 
    AgentStatus, 
    ProviderDetails,
    InvocationRequest,
    InvocationResponse,
    UniversalAgentRequest
)
from ..services.embedding import model as embedding_model
from ..database import db
from ..services.embedding import compute_agent_embedding
from datetime import datetime

# Initialize logger
logger = logging.getLogger(__name__)

# Configure OpenAI client using environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    logger.error("OpenAI API key not found in environment")
    raise ValueError("OpenAI API key not configured")
logger.info("OpenAI API key configured from environment")

router = APIRouter(tags=["agents"])

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
            # Always include marketplace token in headers
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-Marketplace-Token": marketplace_token or ""  # Send empty string if None
            }
            logger.info(f"Making request to {invocation_endpoint} with token: {marketplace_token}")
            logger.info(f"Request payload: {request.input}")

            invocation_res = requests.post(
                invocation_endpoint,
                json={
                    "input": request.input,
                    "trace_id": request.trace_id,
                    "timeout": request.timeout
                },
                headers=headers,
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

@router.post("/find-and-invoke", response_model=InvocationResponse)
async def find_and_invoke_agent(
    request: UniversalAgentRequest,
    marketplace_token: Optional[str] = Header(None, description="Optional marketplace token for agent authentication")
):
    """Find the most suitable agent for the given query and invoke it."""
    logger = logging.getLogger(__name__)
    
    # Load all agents in the database
    agents = db.list_agents()
    if not agents:
        raise HTTPException(
            status_code=404,
            detail="No agents available in the marketplace"
        )
        
    try:
        
        # Create a system prompt to find the most suitable agent
        prompt = "You are an AI agent selector tasked with matching user queries to the most appropriate agent and formatting their input correctly. Your task is to:\n"
        prompt += "1. Carefully analyze the user's query to understand their intent\n"
        prompt += "2. Select the most suitable agent based on capabilities and requirements\n"
        prompt += "3. Extract or infer required parameters from the query to match the agent's input schema\n"
        prompt += "4. Format the input exactly according to the schema, ensuring all required fields are present\n"
        prompt += "5. For sentiment analysis tasks, prefer agents with explicit sentiment-analysis capability\n"
        prompt += "6. For text analysis, use the text from the query or a default if none provided\n\n"
        prompt += "Special Instructions for Sentiment Analysis:\n"
        prompt += "- When user asks for sentiment analysis, look for agents with 'sentiment-analysis' capability\n"
        prompt += "- For NewsAnalyzer, use 'sentiment' as analysis_type and extract keywords from the query\n"
        prompt += "- Default timeframe to '1h' if not specified\n\n"
        
        prompt += "Available agents:\n"
        for agent in agents:
            prompt += f"- Name: {agent['agent_name']}\n"
            prompt += f"  Description: {agent['description']}\n"
            prompt += f"  Capabilities: {', '.join(agent['capabilities'])}\n"
            prompt += f"  Input Schema (MUST MATCH EXACTLY):\n"
            
            # Pretty print the schema with indentation for clarity
            import json
            schema_str = json.dumps(agent['input_schema'], indent=4)
            # Add indentation to each line
            schema_lines = [f"    {line}" for line in schema_str.split("\n")]
            prompt += "\n".join(schema_lines) + "\n\n"
            
            # Add example input if it's the MockTextProcessor
            if agent['agent_name'] == "MockTextProcessor":
                prompt += "  Example valid input:\n"
                prompt += "    {\n"
                prompt += '      "text": "I love this product!",\n'
                prompt += '      "analysis_type": "sentiment"\n'
                prompt += "    }\n\n"
        
        # Add user query and context
        prompt += f"\nUser Query: {request.query}\n"
        if request.additional_context:
            prompt += f"Additional Context: {request.additional_context}\n"
        
        prompt += "\nIMPORTANT: You must respond in the following JSON format:\n"
        prompt += "{\n"
        prompt += '  "selected_agent": "exact_name_of_selected_agent",\n'
        prompt += '  "reason": "detailed explanation of why this agent was selected and how it matches the user\'s needs",\n'
        prompt += '  "confidence": "high/medium/low - how confident you are in this selection",\n'
        prompt += '  "input_parameters": { parameters exactly matching the agent\'s input schema }\n'
        prompt += "}\n\n"
        prompt += "Rules:\n"
        prompt += "1. The selected_agent MUST exactly match one of the provided agent names\n"
        prompt += "2. The input_parameters MUST exactly match the schema of the selected agent\n"
        prompt += "3. If you cannot find a suitable agent, respond with {\"error\": \"No suitable agent found\"}\n"
        prompt += "4. If you cannot construct valid input parameters, respond with {\"error\": \"Cannot construct valid input\"}\n"
        prompt += "5. For text analysis, always extract the actual text to analyze from the user's query\n"
        prompt += "6. For the MockTextProcessor, always use either 'sentiment' or 'summary' for analysis_type\n"
        
        # Define the JSON response schema
        json_schema = {
            "type": "object",
            "properties": {
                "selected_agent": {"type": "string"},
                "reason": {"type": "string"},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "input_parameters": {"type": "object"},
                "error": {"type": "string"}
            },
            "required": ["selected_agent", "reason", "confidence", "input_parameters"]
        }

        # Call OpenAI API with JSON mode using configured client
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found in environment")
                
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",  # Using a model that supports JSON mode
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "Remember to respond ONLY with valid JSON matching the specified format."}
                ],
                temperature=request.temperature or 0.7,
                max_tokens=request.max_tokens or 1000,
                response_format={"type": "json_object"}  # Ensure JSON response
            )
            
            if not response.choices or not response.choices[0].message or not response.choices[0].message.content:
                raise ValueError("Empty response from OpenAI API")
            
            response_content = response.choices[0].message.content
            logger.info(f"OpenAI API response received: {response_content}")
            
            try:
                selection = json.loads(response_content)
                logger.info(f"Parsed OpenAI selection: {selection}")
                
                # Log available agents for debugging
                logger.info(f"Available agents: {[agent['agent_name'] for agent in agents]}")
                logger.info(f"Agent capabilities: {[(agent['agent_name'], agent['capabilities']) for agent in agents]}")
                logger.info(f"Query being processed: {request.query}")
                
                # Validate selection format
                if not isinstance(selection, dict):
                    raise ValueError("OpenAI response is not a JSON object")
                    
                if "error" in selection:
                    raise HTTPException(
                        status_code=404,
                        detail={
                            "status": "error",
                            "error": {
                                "code": "NO_SUITABLE_AGENT",
                                "message": selection["error"],
                                "details": {
                                    "error_type": "AgentSelectionError",
                                    "query": request.query
                                }
                            },
                            "trace_id": str(uuid.uuid4())
                        }
                    )
                    
                required_fields = ["selected_agent", "reason", "confidence", "input_parameters"]
                missing_fields = [field for field in required_fields if field not in selection]
                if missing_fields:
                    raise ValueError(f"Missing required fields in OpenAI response: {missing_fields}")
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse OpenAI response as JSON: {str(e)}")
                raise ValueError(f"Invalid JSON in OpenAI response: {str(e)}")
                
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "status": "error",
                    "error": {
                        "code": "OPENAI_API_ERROR",
                        "message": "Failed to process OpenAI API response",
                        "details": {
                            "error_type": type(e).__name__,
                            "error_message": str(e)
                        }
                    },
                    "trace_id": str(uuid.uuid4())
                }
            )
        
        # Get response (guaranteed to be valid JSON in JSON mode)
        selection = json.loads(response.choices[0].message.content)
        logger.info(f"OpenAI selection response: {selection}")
        
        # Generate trace ID for this request
        trace_id = str(uuid.uuid4())
        
        # Check for error response
        if "error" in selection:
            raise HTTPException(
                status_code=404,
                detail={
                    "status": "error",
                    "error": {
                        "code": "NO_SUITABLE_AGENT",
                        "message": selection["error"],
                        "details": {
                            "error_type": "AgentSelectionError",
                            "query": request.query
                        }
                    },
                    "trace_id": trace_id
                }
            )
        
        # Find the selected agent
        selected_agent = None
        for agent in agents:
            if agent["agent_name"] == selection["selected_agent"]:
                selected_agent = agent
                break
        
        if not selected_agent:
            raise HTTPException(
                status_code=404,
                detail=f"Selected agent {selection['selected_agent']} not found"
            )
        
        # Get the agent's token
        agent_token = db.get_token_for_agent(selected_agent["agent_id"])
        if not agent_token:
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve agent token"
            )
        
        # Validate response against our schema
        try:
            jsonschema.validate(instance=selection, schema=json_schema)
            
            # Validate input parameters against agent's schema
            try:
                jsonschema.validate(
                    instance=selection["input_parameters"],
                    schema=selected_agent["input_schema"]
                )
                
                # Create invocation request
                trace_id = str(uuid.uuid4())
                invocation_request = InvocationRequest(
                    input=selection["input_parameters"],
                    trace_id=trace_id,
                    timeout=request.timeout or 30
                )
                
                # Invoke the agent with the validated parameters
                response = await invoke_agent(
                    agent_id=selected_agent["agent_id"],
                    request=invocation_request,
                    marketplace_token=agent_token
                )
                
                # Add agent selection metadata to the response
                if response.result:
                    response.result["_agent_selection_metadata"] = {
                        "selected_agent": selection["selected_agent"],
                        "confidence": selection["confidence"],
                        "reason": selection["reason"]
                    }
                
                return response
                
            except jsonschema.ValidationError as e:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "status": "error",
                        "error": {
                            "code": "VALIDATION_ERROR",
                            "message": "Generated input parameters do not match agent schema",
                            "details": {
                                "error_type": "ValidationError",
                                "validation_error": str(e),
                                "schema": selected_agent["input_schema"],
                                "input": selection["input_parameters"]
                            }
                        },
                        "trace_id": trace_id
                    }
                )
        
        except Exception as e:
            logger.error(f"Error in find_and_invoke_agent: {str(e)}")
            
            if isinstance(e, ValidationError):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "status": "error",
                        "error": {
                            "code": "VALIDATION_ERROR",
                            "message": "Generated input parameters do not match agent schema",
                            "details": {
                                "error_type": "ValidationError",
                                "error_message": str(e),
                                "schema": selected_agent["input_schema"] if selected_agent else None
                            }
                        },
                        "trace_id": trace_id
                    }
                )
            raise HTTPException(
                status_code=500,
                detail={
                    "status": "error",
                    "error": {
                        "code": "INVOCATION_ERROR",
                        "message": "Failed to invoke selected agent",
                        "details": {
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "selected_agent": selection.get("selected_agent"),
                            "confidence": selection.get("confidence"),
                            "reason": selection.get("reason")
                        }
                    },
                    "trace_id": trace_id
                }
            )

    except Exception as e:
        trace_id = str(uuid.uuid4())
        logger.error(f"Error in find_and_invoke_agent: {str(e)}")
        
        # If it's already an HTTPException with a properly formatted error response, re-raise it
        if isinstance(e, HTTPException) and isinstance(e.detail, dict) and "error" in e.detail:
            raise e
        
        error_response = {
            "status": "error",
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "details": {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "query": request.query,
                    "additional_context": request.additional_context
                }
            },
            "trace_id": trace_id
        }
        
        if "openai" in str(type(e)).lower():
            error_response["error"]["code"] = "OPENAI_API_ERROR"
            error_response["error"]["message"] = "OpenAI API error occurred"
            error_response["error"]["details"] = {
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
            raise HTTPException(status_code=500, detail=error_response)
        elif isinstance(e, HTTPException):
            error_response["error"]["code"] = "REQUEST_ERROR"
            error_response["error"]["message"] = str(e.detail)
            error_response["error"]["details"]["error_type"] = "HTTPException"
            error_response["error"]["details"]["status_code"] = e.status_code
            raise HTTPException(status_code=e.status_code, detail=error_response)
        else:
            raise HTTPException(status_code=500, detail=error_response)
