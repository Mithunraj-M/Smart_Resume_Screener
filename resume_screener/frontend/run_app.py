#!/usr/bin/env python3
"""
Launcher script for the Smart Resume Screener application.
This script helps start both the backend API and frontend interface.
"""

import subprocess
import sys
import time
import requests
import os
from pathlib import Path

def check_backend_status():
    """Check if the backend API is running."""
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        return response.status_code == 200
    except:
        return False

def start_backend():
    """Start the backend API server."""
    print("Starting backend API server...")
    backend_dir = Path(__file__).parent.parent / "backend"
    
    try:
        # Start backend in a subprocess
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "300"],
            cwd=backend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for backend to start
        print("Waiting for backend to start...")
        for i in range(30):  # Wait up to 30 seconds
            if check_backend_status():
                print("✅ Backend API is running on http://localhost:8000")
                return process
            time.sleep(1)
        
        print("❌ Backend failed to start within 30 seconds")
        return None
        
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return None

def start_frontend():
    """Start the Streamlit frontend."""
    print("Starting Streamlit frontend...")
    frontend_dir = Path(__file__).parent
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ], cwd=frontend_dir)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")

def main():
    """Main launcher function."""
    print("🚀 Smart Resume Screener Launcher")
    print("=" * 40)
    
    # Check if backend is already running
    if check_backend_status():
        print("✅ Backend API is already running")
        backend_process = None
    else:
        backend_process = start_backend()
        if not backend_process:
            print("❌ Cannot start application without backend")
            return
    
    try:
        print("\n🌐 Starting frontend interface...")
        print("Frontend will be available at: http://localhost:8501")
        print("Backend API is available at: http://localhost:8000")
        print("\nPress Ctrl+C to stop both services")
        print("=" * 40)
        
        start_frontend()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
    finally:
        if backend_process:
            print("Stopping backend...")
            backend_process.terminate()
            backend_process.wait()

if __name__ == "__main__":
    main()
