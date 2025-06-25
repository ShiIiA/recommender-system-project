#!/usr/bin/env python3
"""
Performance-optimized launcher for the Recipe Recommender App
"""
import subprocess
import sys
import os

def main():
    """Launch the app with performance optimizations"""
    print("🚀 Launching Ghibli Recipe Garden with performance optimizations...")
    
    # Set environment variables for better performance
    env = os.environ.copy()
    env['STREAMLIT_SERVER_MAX_UPLOAD_SIZE'] = '200'
    env['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
    env['STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION'] = 'false'
    
    # Launch streamlit with optimizations
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", "8501",
        "--server.address", "localhost",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 