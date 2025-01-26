#!/bin/bash

# Kill any existing processes on the ports
pkill -f uvicorn

# Wait for processes to die
sleep 2

# Start each agent on its designated port
cd /home/ubuntu/mock_agent

# Mortgage Loan Origination Agents
poetry run uvicorn app.main_loan_doc:app --host 0.0.0.0 --port 8090 &  # Loan Document Processor
poetry run uvicorn app.main_underwriting:app --host 0.0.0.0 --port 8091 &  # Underwriting Analyzer
poetry run uvicorn app.main_closing:app --host 0.0.0.0 --port 8092 &  # Closing Coordinator
poetry run uvicorn app.main_quality:app --host 0.0.0.0 --port 8093 &  # Quality Controller
poetry run uvicorn app.main_funding:app --host 0.0.0.0 --port 8094 &  # Funding Manager

# Wait for all agents to start
sleep 5

echo "All agents started. Testing connectivity..."
for port in 8090 8091 8092 8093 8094; do
    echo "Testing agent on port $port..."
    curl -s http://localhost:$port/health > /dev/null
    if [ $? -eq 0 ]; then
        echo "Agent on port $port is healthy"
    else
        echo "Warning: Agent on port $port is not responding"
    fi
done
