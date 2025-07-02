#!/usr/bin/env python3
"""
Simple startup script for Railway deployment
This helps Railway detect this as a Python application
"""

if __name__ == "__main__":
    import sys
    import os
    
    # Add the current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    # Also add the api directory to path for direct imports
    api_dir = os.path.join(current_dir, 'api')
    sys.path.insert(0, api_dir)
    
    # Import and run the Flask app
    from api.app import app
    
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
