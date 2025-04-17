#!/usr/bin/env python3
"""
Setup script for AI Learning Notebooks
This script helps set up the environment for running the notebooks.
"""

import subprocess
import sys
import os
import platform

def check_python_version():
    """Check if Python version is 3.6 or higher."""
    print("Checking Python version...")
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 6):
        print("ERROR: Python 3.6 or higher is required.")
        print(f"Current version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        sys.exit(1)
    else:
        print(f"✓ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")

def install_requirements():
    """Install required packages from requirements.txt."""
    print("\nInstalling required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ All packages installed successfully!")
    except subprocess.CalledProcessError:
        print("ERROR: Failed to install required packages.")
        sys.exit(1)

def check_dependencies():
    """Check if required packages are importable."""
    print("\nVerifying dependencies...")
    dependencies = [
        "torch", "transformers", "matplotlib", "numpy", "seaborn", 
        "datasets", "notebook", "ipywidgets", "tqdm", "sklearn"
    ]
    
    all_good = True
    for package in dependencies:
        try:
            __import__(package)
            print(f"✓ {package} is installed.")
        except ImportError:
            print(f"✗ {package} is NOT installed.")
            all_good = False
    
    if all_good:
        print("\n✓ All dependencies verified!")
    else:
        print("\n✗ Some dependencies are missing. Please check the errors above.")

def configure_gpu():
    """Check and configure GPU if available."""
    print("\nChecking for GPU availability...")
    try:
        import torch
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "None"
            print(f"✓ GPU is available!")
            print(f"  • Number of devices: {device_count}")
            print(f"  • Device name: {device_name}")
        else:
            print("✗ GPU is NOT available. Notebooks will run on CPU only, which may be slower.")
            
    except ImportError:
        print("✗ Could not check GPU. PyTorch may not be installed correctly.")

def main():
    """Main setup function."""
    print("=" * 60)
    print("AI Learning Notebooks Setup")
    print("=" * 60)
    
    check_python_version()
    install_requirements()
    check_dependencies()
    configure_gpu()
    
    print("\n" + "=" * 60)
    print("Setup complete! You can now run the notebooks using:")
    print("jupyter notebook")
    print("=" * 60)

if __name__ == "__main__":
    main() 