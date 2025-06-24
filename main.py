#!/usr/bin/env python3
"""
Simple startup script for Railway deployment
This helps Railway detect this as a Python application
"""

if __name__ == "__main__":
    import sys
    import os
    
    # Add the current directory to Python path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Import and run the Flask app
    from api.app import app
    
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
