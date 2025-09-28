#!/usr/bin/env python3
"""
Simple script to run the FastAPI email classification application
"""

import uvicorn
from main import app

if __name__ == "__main__":
    print("🚀 Starting Email Classification API...")
    print("📧 Upload emails or paste text to categorize them")
    print("🌐 Open your browser to: http://localhost:8000")
    print("📚 API docs available at: http://localhost:8000/docs")
    print("=" * 60)
    
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
