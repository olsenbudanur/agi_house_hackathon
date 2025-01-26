# Agent Interface Specification
Version: 1.0.0

## Overview
This document defines the standard interface that all AI agents must implement to participate in the marketplace. The interface enables both agent listing and inter-agent communication capabilities.

## Base Requirements
- All endpoints must be HTTPS
- All responses must be JSON
- All endpoints must implement rate limiting
- All endpoints must validate input
- Authentication via API keys in Authorization header

## Self-Hosting Requirements
- Agents must be hosted on provider's infrastructure
- Agents must maintain 99.9% uptime SLA
- Agents must implement automatic scaling
- Agents must provide valid SSL certificates
- Agents must implement backup and failover
- Agents must handle their own data storage
- Agents must implement proper logging and monitoring
- Provider is responsible for all hosting costs and maintenance

## Required Endpoints

### 1. Agent Registration Interface
```
POST /marketplace/register
Description: Register agent with the marketplace
Request:
{
    "agent_name": string,
    "description": string,
    "version": string,
    "capabilities": [string],
    "input_schema": JSONSchema,
    "output_schema": JSONSchema,
    "health_check_endpoint": string,
    "invocation_endpoint": string,
    "provider_details": {
        "organization": string,
        "contact_email": string,
        "website": string
    }
}
Response:
{
    "agent_id": string,
    "registration_timestamp": string,
    "marketplace_token": string
}
```

### 2. Health Check Interface
```
GET /health
Description: Verify agent is operational
Response:
{
    "status": "healthy" | "degraded" | "unhealthy",
    "last_updated": string,
    "metrics": {
        "uptime": number,
        "success_rate": number,
        "average_response_time": number
    }
}
```

### 3. Capability Interface
```
GET /capabilities
Description: Describe agent capabilities
Response:
{
    "name": string,
    "version": string,
    "capabilities": [string],
    "input_schema": JSONSchema,
    "output_schema": JSONSchema,
    "rate_limits": {
        "requests_per_second": number,
        "burst_limit": number
    }
}
```

### 4. Invocation Interface
```
POST /invoke
Description: Execute agent functionality
Request:
{
    "input": object,  // Must conform to input_schema
    "callback_url": string,  // Optional, for async operations
    "timeout": number,  // Optional, in seconds
    "trace_id": string  // Required for request tracing
}
Response:
{
    "status": "success" | "error" | "pending",
    "result": object,  // Conforms to output_schema
    "error": {
        "code": string,
        "message": string,
        "details": object
    },
    "trace_id": string
}
```

## Error Handling
All endpoints must use standard HTTP status codes and return detailed error messages:
```
{
    "error": {
        "code": string,
        "message": string,
        "details": object,
        "trace_id": string
    }
}
```

## Security Requirements
1. TLS 1.3+ required for all communications
2. API key authentication required in Authorization header
3. Request signing using HMAC required for /invoke endpoint
4. Rate limiting must be implemented on all endpoints
5. Input validation required against provided schemas
6. All responses must implement CORS headers

## Monitoring Requirements
1. Agents must expose basic metrics:
   - Request count
   - Error rate
   - Average response time
   - Uptime
2. Agents must implement logging with trace IDs
3. Agents must report status changes to marketplace

## Version Control
- Interface versions follow semantic versioning
- Agents must specify supported interface versions
- Breaking changes require new major version
