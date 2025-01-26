import { useState } from 'react'
import { NavigationMenu, NavigationMenuItem, NavigationMenuList } from "@/components/ui/navigation-menu"
import { Button } from "@/components/ui/button"
import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Bot, Server, Activity, List, PlusCircle } from "lucide-react"
import { AgentList } from './components/agents/AgentList'
import { RegisterAgent } from './components/agents/RegisterAgent'
import { Toaster } from "@/components/ui/toaster"

function App() {
  const [currentView, setCurrentView] = useState<'home' | 'list' | 'register'>('home')

  const renderContent = () => {
    switch (currentView) {
      case 'list':
        return <AgentList />
      case 'register':
        return <RegisterAgent />
      default:
        return (
          <div className="grid gap-6">
            {/* Hero Section */}
            <section className="text-center">
              <h2 className="text-3xl font-bold mb-4">Self-Hosted AI Agent Marketplace</h2>
              <p className="text-muted-foreground mb-6">
                Discover and connect with AI agents through our standardized interface
              </p>
              <div className="flex justify-center gap-4">
                <Button size="lg" onClick={() => setCurrentView('register')}>
                  <PlusCircle className="mr-2" />
                  Register Your Agent
                </Button>
                <Button size="lg" variant="outline" onClick={() => setCurrentView('list')}>
                  <List className="mr-2" />
                  View Agents
                </Button>
              </div>
            </section>

            {/* Features */}
            <section className="grid md:grid-cols-3 gap-6 mt-12">
              <Card>
                <CardHeader>
                  <Bot className="w-8 h-8 mb-2" />
                  <CardTitle>Self-Hosted Agents</CardTitle>
                  <CardDescription>
                    Host and manage your own AI agents while making them discoverable
                  </CardDescription>
                </CardHeader>
              </Card>
              <Card>
                <CardHeader>
                  <Server className="w-8 h-8 mb-2" />
                  <CardTitle>Standardized Interface</CardTitle>
                  <CardDescription>
                    Consistent API endpoints and communication protocols
                  </CardDescription>
                </CardHeader>
              </Card>
              <Card>
                <CardHeader>
                  <Activity className="w-8 h-8 mb-2" />
                  <CardTitle>Health Monitoring</CardTitle>
                  <CardDescription>
                    Real-time health checks and performance metrics
                  </CardDescription>
                </CardHeader>
              </Card>
            </section>
          </div>
        )
    }
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Navigation */}
      <header className="border-b">
        <div className="container mx-auto px-4 py-3">
          <NavigationMenu>
            <NavigationMenuList className="flex items-center justify-between w-full">
              <NavigationMenuItem>
                <h1 className="text-xl font-bold cursor-pointer" onClick={() => setCurrentView('home')}>
                  AI Agent Marketplace
                </h1>
              </NavigationMenuItem>
              <div className="flex gap-4">
                <NavigationMenuItem>
                  <Button variant="ghost" onClick={() => setCurrentView('list')}>
                    <List className="mr-2" />
                    View Agents
                  </Button>
                </NavigationMenuItem>
                <NavigationMenuItem>
                  <Button onClick={() => setCurrentView('register')}>
                    <PlusCircle className="mr-2" />
                    Register Agent
                  </Button>
                </NavigationMenuItem>
              </div>
            </NavigationMenuList>
          </NavigationMenu>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        {renderContent()}
      </main>
      <Toaster />
    </div>
  )
}

export default App
