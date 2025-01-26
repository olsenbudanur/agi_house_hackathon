import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Bot, Activity } from "lucide-react"
import { useState } from "react"
import { InvokeAgent } from "./InvokeAgent"

interface AgentCardProps {
  name: string
  version: string
  description: string
  capabilities: string[]
  status?: "healthy" | "degraded" | "unhealthy"
  agent_id?: string
  input_schema?: Record<string, any>
}

export function AgentCard({ name, version, description, capabilities, status, agent_id = "", input_schema = {} }: AgentCardProps) {
  const [showInvoke, setShowInvoke] = useState(false)
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Bot className="w-6 h-6" />
            <CardTitle>{name}</CardTitle>
          </div>
          <Badge variant={status === "healthy" ? "default" : "destructive"}>
            {status}
          </Badge>
        </div>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex flex-wrap gap-2">
          {capabilities.map((capability) => (
            <Badge key={capability} variant="secondary">
              {capability}
            </Badge>
          ))}
        </div>
      </CardContent>
      <CardFooter className="flex justify-between">
        <div className="flex items-center space-x-2">
          <Activity className="w-4 h-4" />
          <span className="text-sm text-muted-foreground">v{version}</span>
        </div>
        <Button onClick={() => setShowInvoke(true)} size="sm">
          <Bot className="w-4 h-4 mr-2" />
          Invoke
        </Button>
      </CardFooter>
      
      {showInvoke && (
        <div className="fixed inset-0 bg-background/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <InvokeAgent
            agentId={agent_id}
            agentName={name}
            inputSchema={input_schema}
            onClose={() => setShowInvoke(false)}
          />
        </div>
      )}
    </Card>
  )
}
