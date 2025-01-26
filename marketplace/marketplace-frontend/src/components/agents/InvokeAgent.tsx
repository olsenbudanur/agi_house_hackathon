import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { useToast } from "@/components/ui/use-toast"
import { Bot, AlertCircle } from "lucide-react"
import { v4 as uuidv4 } from 'uuid'
import { invokeAgent } from '../../lib/api'

interface InvokeAgentProps {
  agentId: string
  agentName: string
  inputSchema: Record<string, any>
  onClose: () => void
}

export function InvokeAgent({ agentId, agentName, inputSchema, onClose }: InvokeAgentProps) {
  const { toast } = useToast()
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [response, setResponse] = useState<any>(null)

  const handleInvoke = async () => {
    setLoading(true)
    setError(null)
    setResponse(null)

    try {
      // Generate trace ID
      const trace_id = uuidv4()

      // Parse input JSON
      const parsedInput = JSON.parse(input)
      
      // Create invocation request
      const response = await invokeAgent(agentId, {
        input: parsedInput,
        trace_id,
        timeout: 30
      })
    
      setResponse(response)
      toast({
        title: "Agent Invoked Successfully",
        description: `Trace ID: ${trace_id}`,
      })
    } catch (err) {
      const message = err instanceof Error ? 
        err.message : 
        "Failed to invoke agent"
      
      setError(message)
      toast({
        variant: "destructive",
        title: "Invocation Failed",
        description: err instanceof SyntaxError ? 
          "Invalid JSON input. Please check your syntax." : 
          message,
      })
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card className="w-full max-w-2xl mx-auto">
      <CardHeader>
        <div className="flex items-center space-x-2">
          <Bot className="w-6 h-6" />
          <CardTitle>Invoke {agentName}</CardTitle>
        </div>
      </CardHeader>

      {error && (
        <div className="px-6">
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        </div>
      )}

      <CardContent className="space-y-4">
        <div className="space-y-2">
          <label className="text-sm font-medium">
            Input JSON (must match schema):
            <pre className="mt-1 text-xs bg-muted p-2 rounded">
              {JSON.stringify(inputSchema, null, 2)}
            </pre>
          </label>
          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="{}"
            className="font-mono"
            rows={8}
            disabled={loading}
          />
        </div>

        {response && (
          <div className="space-y-2">
            <h3 className="text-sm font-medium">Response:</h3>
            <pre className="bg-muted p-2 rounded overflow-auto max-h-60">
              {JSON.stringify(response, null, 2)}
            </pre>
          </div>
        )}
      </CardContent>

      <CardFooter className="flex justify-between">
        <Button variant="outline" onClick={onClose} disabled={loading}>
          Close
        </Button>
        <Button onClick={handleInvoke} disabled={loading}>
          {loading ? (
            <>
              <span className="animate-spin mr-2">⚪</span>
              Invoking...
            </>
          ) : (
            <>
              <Bot className="mr-2" />
              Invoke Agent
            </>
          )}
        </Button>
      </CardFooter>
    </Card>
  )
}
