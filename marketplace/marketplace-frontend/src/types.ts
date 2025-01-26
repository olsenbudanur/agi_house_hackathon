export interface Agent {
  name: string
  version: string
  description: string
  capabilities: string[]
  input_schema: Record<string, any>
  output_schema: Record<string, any>
  rate_limits: {
    requests_per_second: number
    burst_limit: number
  }
  status?: 'healthy' | 'degraded' | 'unhealthy'
  base_url: string  // Base URL for the agent
  provider_details: ProviderDetails
  agent_id: string
  registration_timestamp?: string
}

export interface ProviderDetails {
  organization: string
  contact_email: string
  website: string
}

export interface AgentRegistration {
  agent_name: string
  description: string
  version: string
  capabilities: string[]
  input_schema: Record<string, any>
  output_schema: Record<string, any>
  base_url: string  // Base URL from which we'll derive /capabilities, /health, and /invoke endpoints
  provider_details: ProviderDetails
}

export interface AgentResponse {
  agent_id: string
  registration_timestamp: string
  marketplace_token: string
}
