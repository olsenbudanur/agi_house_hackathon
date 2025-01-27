import { Agent, AgentRegistration } from '../types'

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api/v1"

export interface InvocationRequest {
  input: Record<string, any>
  trace_id: string
  timeout?: number
}

export interface InvocationResponse {
  status: 'success' | 'error' | 'pending'
  result?: Record<string, any>
  error?: {
    code: string
    message: string
    details: Record<string, any>
  }
  trace_id: string
}

export async function listAgents(): Promise<Agent[]> {
  try {
    const response = await fetch(`${API_URL}/agents/list`)
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    const data = await response.json()
    return data
  } catch (error) {
    console.error('Failed to fetch agents:', error)
    throw error
  }
}

export async function semanticSearch(query: string, threshold: number = 0.3): Promise<Agent[]> {
  try {
    console.log('doing semantic search:', query)
    const response = await fetch(
      `${API_URL}/agents/search?query=${encodeURIComponent(query)}&threshold=${threshold}`
    )
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    const data = await response.json()
    return data
  } catch (error) {
    console.error('Failed to perform semantic search:', error)
    throw error
  }
}

export async function registerAgent(agent: AgentRegistration): Promise<{ agent_id: string; marketplace_token: string }> {
  try {
    const response = await fetch(`${API_URL}/agents/register`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(agent),
    })
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    const data = await response.json()
    return data
  } catch (error) {
    console.error('Failed to register agent:', error)
    throw error
  }
}

export async function checkAgentHealth(agentId: string): Promise<{ status: string; metrics: Record<string, number> }> {
  try {
    const response = await fetch(`${API_URL}/agents/${agentId}/health`)
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    const data = await response.json()
    return data
  } catch (error) {
    console.error('Failed to check agent health:', error)
    throw error
  }
}

export async function invokeAgent(agentId: string, request: InvocationRequest): Promise<InvocationResponse> {
  try {
    const response = await fetch(`${API_URL}/agents/${agentId}/invoke`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
      credentials: 'omit'
    })
    
    const data = await response.json()
    
    if (!response.ok) {
      throw new Error(data.detail || 'Failed to invoke agent')
    }
    
    return data
  } catch (error) {
    console.error('Failed to invoke agent:', error)
    throw error
  }
}
