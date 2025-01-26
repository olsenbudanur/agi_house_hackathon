import { useState, useEffect } from 'react'
import { Upload, AlertCircle } from 'lucide-react'
import { Alert, AlertDescription, AlertTitle } from './components/ui/alert'
import { Button } from './components/ui/button'
import { Card } from './components/ui/card'
import { Textarea } from './components/ui/textarea'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './components/ui/table'

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [result, setResult] = useState<string>('')
  const [error, setError] = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [backendStatus, setBackendStatus] = useState<{
    status: 'checking' | 'connected' | 'error';
    openai?: boolean;
    message?: string;
  }>({ status: 'checking' })

  // Check backend connection on mount
  useEffect(() => {
    fetch(`${import.meta.env.VITE_API_URL}/healthz`)
      .then(response => response.json())
      .then(data => {
        if (data.status === 'ok') {
          setBackendStatus({
            status: 'connected',
            openai: data.openai_status === 'connected',
            message: `API v${data.api_version}`
          })
        } else {
          setBackendStatus({
            status: 'error',
            message: data.message || 'Unknown error'
          })
        }
      })
      .catch((error) => {
        setBackendStatus({
          status: 'error',
          message: error.message
        })
      })
  }, [])

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
      setError('')
    }
  }

  const handleSubmit = async () => {
    if (!file) {
      setError('Please select a file first')
      return
    }

    setLoading(true)
    setError('')
    
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/process-matrix`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Failed to process matrix')
      }

      const data = await response.json()
      // Pretty print the JSON response with validation results
      setResult(JSON.stringify(data, null, 2))
      
      // Show validation errors if any
      if (data.validation && !data.validation.is_valid) {
        setError(`Validation Errors:\n${data.validation.errors.join('\n')}`)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-100 py-8 px-4">
      <div className="max-w-4xl mx-auto space-y-8">
        <div className="text-center">
          <h1 className="text-3xl font-bold">Insurance Matrix Processor</h1>
          <p className="text-gray-600 mt-2">Upload an insurance guideline matrix to extract structured data</p>
          <div className="mt-2">
            {backendStatus.status === 'checking' && (
              <p className="text-yellow-600">Checking backend connection...</p>
            )}
            {backendStatus.status === 'connected' && (
              <div className="space-y-1">
                <p className="text-green-600">✓ Backend connected ({backendStatus.message})</p>
                {backendStatus.openai && (
                  <p className="text-green-600">✓ OpenAI API connected</p>
                )}
                {!backendStatus.openai && (
                  <p className="text-red-600">⚠ OpenAI API not connected</p>
                )}
              </div>
            )}
            {backendStatus.status === 'error' && (
              <p className="text-red-600">⚠ Backend error: {backendStatus.message}</p>
            )}
          </div>
        </div>

        <Card className="p-6">
          <div className="space-y-4">
            <div className="flex items-center justify-center w-full">
              <label className="flex flex-col items-center justify-center w-full min-h-[16rem] border-2 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100 transition-colors">
                <div className="flex flex-col items-center justify-center py-6">
                  <Upload className="w-12 h-12 mb-4 text-gray-400" />
                  <p className="mb-2 text-sm text-gray-500">
                    <span className="font-semibold">Click to upload</span> or drag and drop
                  </p>
                  <p className="text-xs text-gray-500">PNG, JPEG, or JPG files</p>
                </div>
                <input
                  type="file"
                  className="hidden"
                  accept=".png,.jpg,.jpeg"
                  onChange={handleFileChange}
                />
              </label>
            </div>

            {file && (
              <div className="text-sm text-gray-500">
                Selected file: {file.name}
              </div>
            )}

            <Button
              onClick={handleSubmit}
              disabled={loading}
              className="w-full"
            >
              {loading ? 'Processing...' : 'Process Matrix'}
            </Button>

            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {result && (
              <div className="mt-4 space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold">Extracted Matrix Data</h3>
                  {JSON.parse(result).processing_method && (
                    <div className="text-sm text-gray-500">
                      Processed using: {JSON.parse(result).processing_method.name}
                    </div>
                  )}
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Card className="p-4">
                    <h4 className="font-semibold mb-2">Program Details</h4>
                    <div className="space-y-2">
                      <div>
                        <span className="text-gray-600">Program Name:</span>{' '}
                        {JSON.parse(result).data.program_name}
                      </div>
                      <div>
                        <span className="text-gray-600">Effective Date:</span>{' '}
                        {JSON.parse(result).data.effective_date}
                      </div>
                    </div>
                  </Card>

                  <Card className="p-4">
                    <h4 className="font-semibold mb-2">Validation Status</h4>
                    <div className="space-y-2">
                      {JSON.parse(result).validation.is_valid ? (
                        <Alert>
                          <AlertCircle className="h-4 w-4" />
                          <AlertTitle>Valid Matrix Data</AlertTitle>
                          <AlertDescription>
                            All validation checks passed successfully.
                          </AlertDescription>
                        </Alert>
                      ) : (
                        <Alert variant="destructive">
                          <AlertCircle className="h-4 w-4" />
                          <AlertTitle>Validation Issues Found</AlertTitle>
                          <AlertDescription>
                            <ul className="list-disc pl-4">
                              {JSON.parse(result).validation.errors.map((error: string, index: number) => (
                                <li key={index}>{error}</li>
                              ))}
                            </ul>
                          </AlertDescription>
                        </Alert>
                      )}
                      {JSON.parse(result).validation.warnings?.length > 0 && (
                        <Alert variant="warning">
                          <AlertCircle className="h-4 w-4" />
                          <AlertTitle>Warnings</AlertTitle>
                          <AlertDescription>
                            <ul className="list-disc pl-4">
                              {JSON.parse(result).validation.warnings.map((warning: string, index: number) => (
                                <li key={index}>{warning}</li>
                              ))}
                            </ul>
                          </AlertDescription>
                        </Alert>
                      )}
                    </div>
                  </Card>

                  <Card className="p-4 md:col-span-2">
                    <h4 className="font-semibold mb-2">LTV Requirements</h4>
                    <div className="overflow-x-auto">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Property Type</TableHead>
                            <TableHead>Transaction</TableHead>
                            <TableHead>Max LTV</TableHead>
                            <TableHead>Min FICO</TableHead>
                            <TableHead>Max Loan</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {['primary_residence', 'second_home', 'investment'].map((propertyType) => (
                            ['purchase', 'rate_and_term', 'cash_out'].map((transactionType) => {
                              const requirements = JSON.parse(result).data.ltv_requirements[propertyType][transactionType];
                              return (
                                <TableRow key={`${propertyType}-${transactionType}`}>
                                  <TableCell className="font-medium">
                                    {propertyType.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                                  </TableCell>
                                  <TableCell>
                                    {transactionType.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                                  </TableCell>
                                  <TableCell>{requirements.max_ltv}</TableCell>
                                  <TableCell>{requirements.min_fico}</TableCell>
                                  <TableCell>${requirements.max_loan.toLocaleString()}</TableCell>
                                </TableRow>
                              );
                            })
                          ))}
                        </TableBody>
                      </Table>
                    </div>
                  </Card>

                  <Card className="p-4 md:col-span-2">
                    <h4 className="font-semibold mb-2">Credit Requirements</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <p><span className="font-medium">Minimum FICO:</span> {JSON.parse(result).data.credit_requirements.minimum_fico}</p>
                        <p><span className="font-medium">Maximum DTI:</span> {JSON.parse(result).data.credit_requirements.maximum_dti}%</p>
                      </div>
                      <div>
                        <p className="font-medium mb-1">Credit Events Waiting Periods:</p>
                        <ul className="list-disc pl-4">
                          <li>Bankruptcy: {JSON.parse(result).data.credit_requirements.credit_events.bankruptcy}</li>
                          <li>Foreclosure: {JSON.parse(result).data.credit_requirements.credit_events.foreclosure}</li>
                          <li>Short Sale: {JSON.parse(result).data.credit_requirements.credit_events.short_sale}</li>
                        </ul>
                      </div>
                    </div>
                  </Card>

                  <Card className="p-4 md:col-span-2">
                    <h4 className="font-semibold mb-2">Property & Documentation Requirements</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <h5 className="font-medium mb-1">Property Types</h5>
                        <div className="space-y-2">
                          <div>
                            <p className="text-sm text-gray-600">Eligible:</p>
                            <ul className="list-disc pl-4">
                              {JSON.parse(result).data.property_requirements.eligible_types.map((type: string, index: number) => (
                                <li key={index}>{type}</li>
                              ))}
                            </ul>
                          </div>
                          <div>
                            <p className="text-sm text-gray-600">Ineligible:</p>
                            <ul className="list-disc pl-4">
                              {JSON.parse(result).data.property_requirements.ineligible_types.map((type: string, index: number) => (
                                <li key={index}>{type}</li>
                              ))}
                            </ul>
                          </div>
                        </div>
                      </div>
                      <div>
                        <h5 className="font-medium mb-1">Documentation Requirements</h5>
                        <div className="space-y-2">
                          <div>
                            <p className="text-sm text-gray-600">Required Documents:</p>
                            <ul className="list-disc pl-4">
                              {JSON.parse(result).data.income_documentation.required_documents.map((doc: string, index: number) => (
                                <li key={index}>{doc}</li>
                              ))}
                            </ul>
                          </div>
                          <div>
                            <p className="text-sm text-gray-600">Self-Employed Requirements:</p>
                            <ul className="list-disc pl-4">
                              {JSON.parse(result).data.income_documentation.self_employed.requirements.map((req: string, index: number) => (
                                <li key={index}>{req}</li>
                              ))}
                            </ul>
                          </div>
                        </div>
                      </div>
                    </div>
                  </Card>

                  <Card className="p-4 md:col-span-2">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-semibold">Raw JSON Data</h4>
                      <div className="text-sm text-gray-500">
                        {JSON.parse(result)._metadata && (
                          <span>Processed: {new Date(JSON.parse(result)._metadata.processing_timestamp).toLocaleString()}</span>
                        )}
                      </div>
                    </div>
                    <Textarea
                      value={JSON.stringify(JSON.parse(result), null, 2)}
                      readOnly
                      className="min-h-screen/2 font-mono text-sm"
                    />
                  </Card>
                </div>
              </div>
            )}
          </div>
        </Card>
      </div>
    </div>
  )
}

export default App
