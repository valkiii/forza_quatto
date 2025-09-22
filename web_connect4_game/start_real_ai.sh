#!/bin/bash

# Quick start script for Connect 4 with Real AI
# This script starts both the Flask API and web server

echo "🚀 Starting Connect 4 with Real AI..."
echo "======================================="

# Check if we're in the right directory
if [ ! -f "index.html" ]; then
    echo "❌ Error: Please run this script from the web_connect4_game directory"
    exit 1
fi

# Check if Python requirements are installed
if ! python -c "import flask, torch" 2>/dev/null; then
    echo "⚠️  Installing Python dependencies..."
    cd api
    pip install -r requirements.txt
    cd ..
fi

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        echo "⚠️  Port $port is already in use"
        return 1
    else
        return 0
    fi
}

# Start Flask API in background
echo "🤖 Starting Flask API server..."
if check_port 5000; then
    cd api
    python app.py &
    API_PID=$!
    cd ..
    echo "   API server started (PID: $API_PID)"
    echo "   Health check: http://localhost:5000"
else
    echo "   Port 5000 in use, API may already be running"
fi

# Wait a moment for API to start
sleep 3

# Start web server
echo "🌐 Starting web server..."
if check_port 8000; then
    python -m http.server 8000 &
    WEB_PID=$!
    echo "   Web server started (PID: $WEB_PID)"
    echo "   Game URL: http://localhost:8000"
else
    echo "   Port 8000 in use, web server may already be running"
fi

# Wait another moment
sleep 2

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎮 Open your browser and go to: http://localhost:8000"
echo "🔧 API health check: http://localhost:5000"
echo "📊 Ensemble info: http://localhost:5000/api/ensemble/info"
echo ""
echo "🛑 To stop the servers:"
echo "   kill $API_PID $WEB_PID"
echo "   or press Ctrl+C in this terminal"
echo ""

# Keep script running and wait for interrupt
trap 'echo ""; echo "🛑 Stopping servers..."; kill $API_PID $WEB_PID 2>/dev/null; echo "✅ Servers stopped"; exit' INT

echo "ℹ️  Press Ctrl+C to stop all servers"
wait