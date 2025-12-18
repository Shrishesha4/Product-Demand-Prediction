#!/bin/bash

# Quick start script for E-commerce Demand Forecasting GUI

echo "🚀 Starting E-commerce Demand Forecasting GUI..."
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Error: Virtual environment not found. Please create it first:"
    echo "   python -m venv .venv"
    echo "   source .venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Check if node_modules exists
if [ ! -d "gui/node_modules" ]; then
    echo "❌ Error: Node modules not found. Installing..."
    cd gui
    npm install
    cd ..
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup INT TERM

# Start backend
echo "📡 Starting FastAPI backend on http://localhost:8000..."
source .venv/bin/activate
cd be
python -m uvicorn main:app --host 0.0.0.0 --port 8000 > /dev/null 2>&1 &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 3

# Start frontend
echo "🎨 Starting Svelte frontend on http://localhost:5173..."
cd gui
npm run dev > /dev/null 2>&1 &
FRONTEND_PID=$!
cd ..

# Wait for frontend to start
sleep 3

echo ""
echo "✅ Both servers are running!"
echo ""
echo "   Frontend: http://localhost:5173"
echo "   Backend:  http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Keep script running
wait
