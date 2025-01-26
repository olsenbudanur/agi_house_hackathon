import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Bot, AlertCircle, CheckCircle2 } from "lucide-react"
import { useToast } from "@/components/ui/use-toast"
import { AgentRegistration, AgentResponse } from "../../types"
import { registerAgent } from "../../lib/api"

interface RegisterAgentFormData extends Omit<AgentRegistration, 'provider_details'> {
  organization: string
  contact_email: string
  website: string
}

export function RegisterAgent() {
  const { toast } = useToast()
  const [formData, setFormData] = useState<RegisterAgentFormData>({
    agent_name: "",
    description: "",
    version: "",
    capabilities: [],
    base_url: "",
    organization: "",
    contact_email: "",
    website: "",
    input_schema: {},
    output_schema: {}
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<AgentResponse | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setSuccess(null)

    try {
      const registrationData: AgentRegistration = {
        ...formData,
        provider_details: {
          organization: formData.organization,
          contact_email: formData.contact_email,
          website: formData.website
        }
      }

      const response = await registerAgent(registrationData)
      setSuccess({
        ...response,
        registration_timestamp: new Date().toISOString()
      })
      toast({
        title: "Registration Successful",
        description: "Your agent has been registered successfully.",
      })
      // Reset form after successful registration
      setFormData({
        agent_name: "",
        description: "",
        version: "",
        capabilities: [],
        base_url: "",
        organization: "",
        contact_email: "",
        website: "",
        input_schema: {},
        output_schema: {}
      })
    } catch (error) {
      setError("Failed to register agent. Please check your inputs and try again.")
      toast({
        variant: "destructive",
        title: "Registration Failed",
        description: "There was an error registering your agent.",
      })
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card className="max-w-2xl mx-auto">
      <CardHeader>
        <div className="flex items-center space-x-2">
          <Bot className="w-6 h-6" />
          <CardTitle>Register Your Agent</CardTitle>
        </div>
        <CardDescription>
          List your self-hosted AI agent in the marketplace
        </CardDescription>
      </CardHeader>
      {error && (
        <div className="px-6">
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        </div>
      )}
      {success && (
        <div className="px-6">
          <Alert>
            <CheckCircle2 className="h-4 w-4" />
            <AlertDescription>
              Agent registered successfully! Your agent ID is: {success.agent_id}
              <br />
              Please save your marketplace token: {success.marketplace_token}
            </AlertDescription>
          </Alert>
        </div>
      )}
      <form onSubmit={handleSubmit}>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="agent_name">Agent Name</Label>
            <Input
              id="agent_name"
              value={formData.agent_name}
              onChange={(e) => setFormData({ ...formData, agent_name: e.target.value })}
              required
              disabled={loading}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="description">Description</Label>
            <Textarea
              id="description"
              value={formData.description}
              onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              required
              disabled={loading}
            />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="version">Version</Label>
              <Input
                id="version"
                value={formData.version}
                onChange={(e) => setFormData({ ...formData, version: e.target.value })}
                required
                disabled={loading}
                placeholder="1.0.0"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="capabilities">Capabilities (comma-separated)</Label>
              <Input
                id="capabilities"
                value={formData.capabilities.join(", ")}
                onChange={(e) => setFormData({ ...formData, capabilities: e.target.value.split(",").map(c => c.trim()) })}
                required
                disabled={loading}
                placeholder="text, image, audio"
              />
            </div>
          </div>
          <div className="space-y-2">
            <Label htmlFor="base_url">Base URL</Label>
            <Input
              id="base_url"
              type="url"
              value={formData.base_url}
              onChange={(e) => setFormData({ ...formData, base_url: e.target.value })}
              required
              disabled={loading}
              placeholder="https://your-agent.com"
            />
          </div>
          <div className="space-y-4">
            <h3 className="text-lg font-medium">Provider Details</h3>
            <div className="space-y-2">
              <Label htmlFor="organization">Organization</Label>
              <Input
                id="organization"
                value={formData.organization}
                onChange={(e) => setFormData({ ...formData, organization: e.target.value })}
                required
                disabled={loading}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="contact_email">Contact Email</Label>
              <Input
                id="contact_email"
                type="email"
                value={formData.contact_email}
                onChange={(e) => setFormData({ ...formData, contact_email: e.target.value })}
                required
                disabled={loading}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="website">Website</Label>
              <Input
                id="website"
                type="url"
                value={formData.website}
                onChange={(e) => setFormData({ ...formData, website: e.target.value })}
                required
                disabled={loading}
                placeholder="https://your-organization.com"
              />
            </div>
          </div>
        </CardContent>
        <CardFooter>
          <Button type="submit" className="w-full" disabled={loading}>
            {loading ? (
              <>
                <span className="animate-spin mr-2">⚪</span>
                Registering...
              </>
            ) : (
              <>
                <Bot className="mr-2" />
                Register Agent
              </>
            )}
          </Button>
        </CardFooter>
      </form>
    </Card>
  )
}
