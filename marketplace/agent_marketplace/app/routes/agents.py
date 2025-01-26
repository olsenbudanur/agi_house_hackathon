from fastapi import APIRouter, HTTPException, Header, Query
from typing import List, Optional, Tuple
import numpy as np
import os
import openai
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
import logging

# Initialize logger
logger = logging.getLogger(__name__)

# Configure OpenAI API key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    logger.warning("OPENAI_API_KEY environment variable not set. find-and-invoke endpoint will not work.")

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

@router.post("/find-and-invoke", response_model=InvocationResponse)
async def find_and_invoke_agent(request: UniversalAgentRequest):
    """Find the most suitable agent for the given query and invoke it."""
    from logging import getLogger
    import openai
    import uuid
    from jsonschema import validate, ValidationError
    
    logger = getLogger(__name__)
    
    # Import required modules
    import openai
    from jsonschema import validate, ValidationError
    import json
    import uuid
    
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
        prompt += "4. Format the input exactly according to the schema, ensuring all required fields are present\n\n"
        
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

        # Call OpenAI API with JSON mode
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt}
            ],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            response_format={"type": "json_object", "schema": json_schema}
        )
        
        # Get response (guaranteed to be valid JSON in JSON mode)
        selection = json.loads(response.choices[0].message.content)
            
        # Check for error responses from the LLM
        if "error" in selection:
            trace_id = str(uuid.uuid4())
            raise HTTPException(
                status_code=400,
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
        
        # Extract required fields
        selected_agent_name = selection.get("selected_agent")
        input_parameters = selection.get("input_parameters")
        confidence = selection.get("confidence", "low")
        reason = selection.get("reason", "No reason provided")
        
        if not selected_agent_name or not input_parameters:
            trace_id = str(uuid.uuid4())
            raise HTTPException(
                status_code=400,
                detail={
                    "status": "error",
                    "error": {
                        "code": "INVALID_LLM_RESPONSE",
                        "message": "Missing required fields in LLM response",
                        "details": {
                            "error_type": "ValidationError",
                            "missing_fields": [
                                "selected_agent" if not selected_agent_name else None,
                                "input_parameters" if not input_parameters else None
                            ]
                        }
                    },
                    "trace_id": trace_id
                }
            )
        
        # Log the selection details
        logger.info(f"Selected agent: {selected_agent_name}")
        logger.info(f"Selection confidence: {confidence}")
        logger.info(f"Selection reason: {reason}")
        
        # Find the selected agent
        selected_agent = next(
            (a for a in agents if a["agent_name"] == selected_agent_name),
            None
        )
        
        if not selected_agent:
            trace_id = str(uuid.uuid4())
            raise HTTPException(
                status_code=400,
                detail={
                    "status": "error",
                    "error": {
                        "code": "AGENT_NOT_FOUND",
                        "message": f"Selected agent '{selected_agent_name}' not found",
                        "details": {
                            "error_type": "AgentSelectionError",
                            "selected_agent": selected_agent_name,
                            "available_agents": [a["agent_name"] for a in agents]
                        }
                    },
                    "trace_id": trace_id
                }
            )
            
        # Extract and validate schema
        input_schema = selected_agent["input_schema"]
        logger.debug(f"Agent input schema: {json.dumps(input_schema, indent=2)}")
        trace_id = str(uuid.uuid4())
        
        # Pre-validation checks
        if not isinstance(input_parameters, dict):
            raise HTTPException(
                status_code=400,
                detail={
                    "status": "error",
                    "error": {
                        "code": "VALIDATION_ERROR",
                        "message": "Schema validation failed: input parameters must be a JSON object",
                        "details": {
                            "error_type": "TypeError",
                            "received_type": str(type(input_parameters))
                        }
                    },
                    "trace_id": trace_id
                }
            )
        
        # Full schema validation
        try:
            # Check required fields first
            if "required" in input_schema:
                missing_fields = [
                    field for field in input_schema["required"]
                    if field not in input_parameters
                ]
                if missing_fields:
                    error_response = {
                        "status": "error",
                        "error": {
                            "code": "VALIDATION_ERROR",
                            "message": "Missing required fields in input parameters",
                            "details": {
                                "error_type": "ValidationError",
                                "missing_fields": missing_fields,
                                "schema": input_schema,
                                "received": input_parameters
                            }
                        },
                        "trace_id": trace_id
                    }
                    raise HTTPException(status_code=400, detail=error_response)
            
            # Then do full schema validation
            validate(instance=input_parameters, schema=input_schema)
            logger.info("Input parameters validated successfully")
        except ValidationError as e:
            error_response = {
                "status": "error",
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Missing required fields in input parameters",
                    "details": {
                        "error_type": "ValidationError",
                        "validation_error": str(e),
                        "schema": input_schema,
                        "received": input_parameters
                    }
                },
                "trace_id": trace_id
            }
            raise HTTPException(status_code=400, detail=error_response)
            
            # Log the matched fields for debugging
            if "properties" in input_schema:
                matched_fields = {
                    field: input_parameters.get(field)
                    for field in input_schema["properties"]
                    if field in input_parameters
                }
                logger.debug(f"Matched fields: {json.dumps(matched_fields, indent=2)}")
            
        except ValidationError as e:
            logger.error(f"Schema validation failed: {str(e)}")
            # Enhanced error message with schema comparison
            trace_id = str(uuid.uuid4())
            error_response = {
                "status": "error",
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Schema validation failed: input parameters do not match agent schema",
                    "details": {
                        "error_type": "ValidationError",
                        "validation_error": str(e),
                        "received_parameters": input_parameters,
                        "expected_schema": input_schema
                    }
                },
                "trace_id": trace_id
            }
            raise HTTPException(status_code=400, detail=error_response)
            
        # Prepare invocation context
        trace_id = str(uuid.uuid4())
        timeout = min(request.timeout or 30, 300)  # Cap at 5 minutes
        
        # Create invocation request with metadata
        invoke_request = InvocationRequest(
            input=input_parameters,
            trace_id=trace_id,
            timeout=timeout
        )
        
        # Log the invocation details
        logger.info(f"Invoking agent: {selected_agent['agent_name']} (ID: {selected_agent['agent_id']})")
        logger.info(f"Trace ID: {trace_id}")
        logger.info(f"Confidence level: {confidence}")
        logger.info(f"Selection reason: {reason}")
        logger.debug(f"Input parameters: {json.dumps(input_parameters, indent=2)}")
        
        try:
            # Invoke the selected agent
            response = await invoke_agent(
                agent_id=selected_agent["agent_id"],
                request=InvocationRequest(
                    input=input_parameters,
                    trace_id=trace_id,
                    timeout=timeout
                )
            )

            # Add metadata to successful response
            if response.status == "success":
                if not response.result:
                    response.result = {}
                response.result.update({
                    "_agent_selection_metadata": {
                        "selected_agent": selected_agent_name,
                        "confidence": confidence,
                        "reason": reason
                    }
                })

            return response
            
        except Exception as e:
            logger.error(f"Agent invocation failed: {str(e)}")
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
                                "schema": input_schema
                            }
                        },
                        "trace_id": trace_id
                    }
                )
            else:
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
                                "selected_agent": selected_agent_name,
                                "confidence": confidence,
                                "reason": reason
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
        
        if "InvalidRequestError" in str(type(e)):
            error_response["error"]["code"] = "OPENAI_REQUEST_ERROR"
            error_response["error"]["message"] = "Invalid request to OpenAI API"
            error_response["error"]["details"]["error_type"] = "InvalidRequestError"
            raise HTTPException(status_code=400, detail=error_response)
        elif isinstance(e, HTTPException):
            error_response["error"]["code"] = "REQUEST_ERROR"
            error_response["error"]["message"] = str(e.detail)
            error_response["error"]["details"]["error_type"] = "HTTPException"
            error_response["error"]["details"]["status_code"] = e.status_code
            raise HTTPException(status_code=e.status_code, detail=error_response)
        else:
            raise HTTPException(status_code=500, detail=error_response)
    except HTTPException:
        raise
    except Exception as e:
        trace_id = str(uuid.uuid4())
        logger.error(f"Error in find_and_invoke_agent: {str(e)}")
        
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
        
        if isinstance(e, openai.error.OpenAIError):
            error_response["error"]["code"] = "OPENAI_API_ERROR"
            error_response["error"]["message"] = "OpenAI API error occurred"
            error_response["error"]["details"] = {
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
            raise HTTPException(status_code=500, detail=error_response)
        else:
            raise HTTPException(
                status_code=500,
                detail={
                    "status": "error",
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "An unexpected error occurred while processing your request",
                        "details": {
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "query": request.query,
                            "additional_context": request.additional_context
                        }
                    },
                    "trace_id": trace_id
                }
            )
