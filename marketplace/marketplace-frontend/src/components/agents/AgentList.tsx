import { useEffect, useState, useRef, useCallback } from "react"
import { AgentCard } from "./AgentCard"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { AlertCircle, RefreshCw, Search, TrendingUp } from "lucide-react"
import { Agent } from "../../types"
import { listAgents, semanticSearch } from "../../lib/api"

export function AgentList() {
  const [agents, setAgents] = useState<Agent[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [searchTerm, setSearchTerm] = useState("")
  const [statusFilter, setStatusFilter] = useState<string>("all")
  const [semanticMode, setSemanticMode] = useState(false)
  const [isSearching, setIsSearching] = useState(false)
  const searchTimerRef = useRef<NodeJS.Timeout>()

  const debouncedSearch = useCallback((query: string) => {
    if (searchTimerRef.current) {
      clearTimeout(searchTimerRef.current)
    }
    searchTimerRef.current = setTimeout(() => {
      performSearch(query)
    }, 300)
  }, [])

  const fetchAgents = async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await listAgents()
      setAgents(data)
    } catch (error) {
      const errorMessage = error instanceof Error
        ? `Failed to fetch agents: ${error.message}`
        : 'An unexpected error occurred while fetching agents'
      
      console.error('Fetch error:', error)
      setError(errorMessage)
    } finally {
      setLoading(false)
    }
  }

  const performSearch = async (query: string) => {
    if (!query) {
      await fetchAgents()
      return
    }

    console.log(`Performing ${semanticMode ? 'semantic' : 'basic'} search for:`, query)

    try {
      setIsSearching(true)
      setError(null)
      console.log('Performing semantic search...');
      const results = await semanticSearch(query);
      console.log('Semantic search results:', results);
      setAgents(results)
    } catch (error) {
      const errorMessage = error instanceof Error 
        ? `Search failed: ${error.message}`
        : 'An unexpected error occurred while searching agents'
      
      console.error('Search error:', error)
      setError(errorMessage)
      
      // Reset agents to initial state on error
      fetchAgents()
    } finally {
      setIsSearching(false)
    }
  }

  useEffect(() => {
    fetchAgents()
  }, [])

  const filteredAgents = agents.filter(agent => 
    statusFilter === "all" || agent.status === statusFilter
  )

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="flex gap-4">
          <div className="w-64 h-10 bg-muted rounded animate-pulse" />
          <div className="w-48 h-10 bg-muted rounded animate-pulse" />
        </div>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="h-48 bg-muted rounded-lg animate-pulse" />
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
      
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder={semanticMode ? "Describe what you're looking for..." : "Search agents..."}
            value={searchTerm}
            onChange={(e) => {
              setSearchTerm(e.target.value)
              debouncedSearch(e.target.value)
            }}
            className="pl-9"
          />
        </div>
        <Select value={statusFilter} onValueChange={setStatusFilter}>
          <SelectTrigger className="w-48">
            <SelectValue placeholder="Filter by status" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Status</SelectItem>
            <SelectItem value="healthy">Healthy</SelectItem>
            <SelectItem value="degraded">Degraded</SelectItem>
            <SelectItem value="unhealthy">Unhealthy</SelectItem>
          </SelectContent>
        </Select>
        <Button variant="outline" size="icon" onClick={fetchAgents}>
          <RefreshCw className="h-4 w-4" />
        </Button>
      </div>

      {isSearching ? (
        <div className="text-center py-12">
          <p className="text-muted-foreground">Searching for agents...</p>
        </div>
      ) : filteredAgents.length === 0 ? (
        <div className="text-center py-12">
          <p className="text-muted-foreground">No agents found matching your criteria</p>
        </div>
      ) : (
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredAgents.map((agent) => (
            <AgentCard 
              key={agent.agent_id || agent.name}
              name={agent.name}
              version={agent.version}
              description={agent.description}
              capabilities={agent.capabilities}
              status={agent.status}
              agent_id={agent.agent_id}
              input_schema={agent.input_schema}
            />
          ))}
        </div>
      )}
    </div>
  )
}
