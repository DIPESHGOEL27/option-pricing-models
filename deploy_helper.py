#!/usr/bin/env python3
"""
Deployment Helper Script

This script helps prepare the Option Pricing Platform for deployment
by setting up git repository and providing deployment instructions.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {description} failed")
            print(f"Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False

def check_git_status():
    """Check if git is initialized and configured"""
    print("\n📋 Checking Git status...")
    
    # Check if git is initialized
    if not os.path.exists('.git'):
        print("⚠️  Git repository not initialized")
        return False
    
    # Check git config
    try:
        result = subprocess.run(['git', 'config', 'user.name'], capture_output=True, text=True)
        if result.returncode != 0:
            print("⚠️  Git user.name not configured")
            return False
        
        result = subprocess.run(['git', 'config', 'user.email'], capture_output=True, text=True)
        if result.returncode != 0:
            print("⚠️  Git user.email not configured")
            return False
        
        print("✅ Git is properly configured")
        return True
    except:
        print("❌ Git not found or not configured")
        return False

def initialize_git():
    """Initialize git repository if not already done"""
    if not check_git_status():
        print("\n🚀 Initializing Git repository...")
        
        # Initialize git
        if not run_command("git init", "Initializing git repository"):
            return False
        
        # Create .gitignore if it doesn't exist
        gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Project specific
market_data.db
*.log
test_results/
"""
        
        if not os.path.exists('.gitignore'):
            with open('.gitignore', 'w') as f:
                f.write(gitignore_content.strip())
            print("✅ Created .gitignore file")
        
        return True
    
    return True

def prepare_for_deployment():
    """Prepare the project for deployment"""
    print("🚀 Advanced Option Pricing Platform - Deployment Helper")
    print("=" * 60)
    
    # Check current directory
    current_dir = Path.cwd()
    print(f"📁 Current directory: {current_dir}")
    
    # Check if we're in the right directory
    if not os.path.exists('requirements.txt') or not os.path.exists('vercel.json'):
        print("❌ This doesn't appear to be the Option Pricing Platform directory")
        print("Please run this script from the project root directory")
        return False
    
    # Initialize git if needed
    if not initialize_git():
        return False
    
    # Stage all files
    if not run_command("git add .", "Staging all files"):
        return False
    
    # Create initial commit if no commits exist
    result = subprocess.run(['git', 'log', '--oneline'], capture_output=True, text=True)
    if result.returncode != 0:  # No commits exist
        if not run_command('git commit -m "Initial commit: Advanced Option Pricing Platform v2.0"', 
                          "Creating initial commit"):
            return False
    else:
        print("✅ Git repository already has commits")
    
    # Display next steps
    print("\n🎉 Repository prepared successfully!")
    print("\n📋 Next Steps for Deployment:")
    print("\n1. Create a new repository on GitHub/GitLab")
    print("2. Add remote origin:")
    print("   git remote add origin <your-repository-url>")
    print("\n3. Push to remote:")
    print("   git branch -M main")
    print("   git push -u origin main")
    print("\n4. Deploy to Vercel:")
    print("   - Install Vercel CLI: npm i -g vercel")
    print("   - Login: vercel login")
    print("   - Deploy: vercel")
    print("\n5. Or deploy via Vercel Dashboard:")
    print("   - Connect your GitHub/GitLab repository")
    print("   - Vercel will auto-detect Python and deploy")
    
    print(f"\n🔗 Your app will be available at: https://<your-app-name>.vercel.app")
    
    return True

def test_deployment_readiness():
    """Test if the application is ready for deployment"""
    print("\n🧪 Testing deployment readiness...")
    
    # Test Python imports
    try:
        sys.path.append('api')
        from option_pricing import AdvancedOptionPricer
        pricer = AdvancedOptionPricer()
        result = pricer.black_scholes(100, 100, 0.25, 0.05, 0.2, 'call')
        print(f"✅ Core option pricing working: {result:.4f}")
    except Exception as e:
        print(f"❌ Core option pricing failed: {e}")
        return False
    
    # Test Flask app import
    try:
        from app import app
        print("✅ Flask app imports successfully")
    except Exception as e:
        print(f"❌ Flask app import failed: {e}")
        return False
    
    print("✅ Application is ready for deployment!")
    return True

if __name__ == "__main__":
    success = prepare_for_deployment()
    
    if success:
        # Test deployment readiness
        test_deployment_readiness()
        
        print("\n🎊 All checks passed! Ready to deploy to Vercel.")
        print("\nSee DEPLOYMENT_GUIDE.md for detailed instructions.")
    else:
        print("\n❌ Deployment preparation failed. Please check the errors above.")
        sys.exit(1)
