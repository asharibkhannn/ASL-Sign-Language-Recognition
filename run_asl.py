#!/usr/bin/env python3
"""
ASL Sign Detector Launcher
This script ensures that the ASL Sign Detector runs with Python 3
"""

import sys
import os
import subprocess

def main():
    """Launch the ASL Sign Detector application with Python 3"""
    print("Launching ASL Sign Detector...")
    
    # Get the path to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the main application
    app_path = os.path.join(script_dir, "asl_camera_gui.py")
    
    # Try to use python3 command first
    try:
        subprocess.run(["python3", app_path], check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        # Fall back to python command if python3 fails
        try:
            print("Python3 command failed, trying 'python'...")
            # Check if python is Python 3.x
            version = subprocess.check_output(["python", "--version"]).decode().strip()
            if "Python 3" in version:
                subprocess.run(["python", app_path], check=True)
            else:
                print(f"Warning: {version} detected, but Python 3.x is required.")
                print("Please install Python 3 or run directly with: python3 asl_camera_gui.py")
        except (subprocess.SubprocessError, FileNotFoundError):
            print("Error: Could not launch the application.")
            print("Please ensure Python 3 is installed and available in your PATH.")
            print("You can run the application manually with: python3 asl_camera_gui.py")

if __name__ == "__main__":
    main() 