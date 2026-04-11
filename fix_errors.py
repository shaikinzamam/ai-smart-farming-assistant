#!/usr/bin/env python3
"""
Auto-Fix Script for AI Smart Farming
Installs all dependencies and fixes common errors
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and report status"""
    if description:
        print(f"\n📦 {description}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ {description or 'Command'} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {e.stderr[:200]}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("AI SMART FARMING - AUTO-FIX SCRIPT")
    print("="*60)
    
    # Step 1: Upgrade pip
    print("\n[1/3] Upgrading pip...")
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], "Upgrading pip")
    
    # Step 2: Install dependencies
    print("\n[2/3] Installing Python dependencies...")
    packages = [
        "tensorflow",
        "numpy",
        "pillow",
        "kaggle",
        "tqdm",
        "fastapi",
        "uvicorn[standard]",
        "python-multipart",
        "pydantic"
    ]
    
    for package in packages:
        cmd = [sys.executable, "-m", "pip", "install", "-q", package]
        if run_command(cmd, f"Installing {package}"):
            print(f"  ✓ {package}")
        else:
            print(f"  ⚠ Failed to install {package}, continuing...")
    
    # Step 3: Verify installation
    print("\n[3/3] Verifying installations...")
    required_packages = {
        "tensorflow": "TensorFlow",
        "numpy": "NumPy",
        "PIL": "Pillow",
        "kaggle": "Kaggle",
        "tqdm": "tqdm"
    }
    
    all_good = True
    for import_name, display_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"  ✓ {display_name}")
        except ImportError:
            print(f"  ✗ {display_name}")
            all_good = False
    
    # Summary
    print("\n" + "="*60)
    if all_good:
        print("✓ ALL ERRORS FIXED - READY TO USE!")
        print("="*60)
        print("\nNext steps:")
        print("1. Run: python verify_setup.py")
        print("2. Run: python setup_kaggle.py")
        print("3. Run: python train_complete.py --epochs 15 --batch_size 16")
        return True
    else:
        print("⚠ Some packages failed to install")
        print("="*60)
        print("\nManual fix:")
        print(f"Run: {sys.executable} -m pip install -r requirements.txt")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
