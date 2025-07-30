#!/bin/bash

# SOUL Project Web Application Startup Script

echo "==================================="
echo "SOUL Project - Web Application"
echo "==================================="
echo

# Check if we're in the correct directory
if [ ! -f "src/webapp.py" ]; then
    echo "Error: src/webapp.py not found. Please run this script from the project root directory."
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    else
        echo "Virtual environment not found. Please run 'uv sync' first."
        exit 1
    fi
fi

echo "Starting SOUL Project Web Application..."
echo "Web interface will be available at: http://localhost:8000"
echo "API documentation available at: http://localhost:8000/docs"
echo
echo "Press Ctrl+C to stop the server"
echo

# Start the application
cd src && uvicorn webapp:app --host 0.0.0.0 --port 8000 --reload 