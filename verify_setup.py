#!/usr/bin/env python3
"""
Quick Setup Verification Script
Run this first to check if your environment is ready
"""

import subprocess
import sys
from pathlib import Path
import json

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"\n✓ Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ⚠ Warning: Python 3.8+ recommended")
        return False
    return True

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✓ {package_name} is installed")
        return True
    except ImportError:
        print(f"✗ {package_name} is NOT installed")
        return False

def check_kaggle_config():
    """Check if Kaggle is configured"""
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    
    if kaggle_json.exists():
        try:
            with open(kaggle_json) as f:
                data = json.load(f)
                if 'username' in data and 'key' in data:
                    print(f"✓ Kaggle configured (username: {data['username']})")
                    return True
        except:
            pass
    
    # Check project directory
    project_kaggle = Path.cwd() / 'kaggle.json'
    if project_kaggle.exists():
        print("⚠ kaggle.json found in project directory")
        print("  (should be in ~/.kaggle/kaggle.json)")
        return True
    
    print("✗ Kaggle not configured")
    print("  Setup: Place kaggle.json from Kaggle.com → Account → API")
    return False

def check_kaggle_cli():
    """Check if Kaggle CLI works"""
    try:
        result = subprocess.run(
            ['kaggle', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"✓ Kaggle CLI: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        print("✗ Kaggle CLI not found in PATH")
    except Exception as e:
        print(f"✗ Kaggle CLI error: {e}")
    
    return False

def check_dataset():
    """Check if dataset exists"""
    dataset_dir = Path.cwd() / 'dataset'
    
    if not dataset_dir.exists():
        print("✗ Dataset directory not found")
        return False
    
    required_dirs = [
        'tomato/early_blight',
        'tomato/late_blight',
        'tomato/leaf_mold',
        'tomato/healthy',
        'banana/sigatoka',
        'banana/panama_disease',
        'banana/healthy',
    ]
    
    missing = []
    for subdir in required_dirs:
        full_path = dataset_dir / subdir
        if not full_path.exists():
            missing.append(subdir)
    
    if missing:
        print(f"✗ Dataset incomplete - missing {len(missing)} directories")
        for m in missing[:3]:
            print(f"    {m}")
        if len(missing) > 3:
            print(f"    ... and {len(missing)-3} more")
        return False
    
    print(f"✓ Dataset structure complete ({len(required_dirs)} classes found)")
    return True

def main():
    print("\n" + "="*60)
    print("AI SMART FARMING - SETUP VERIFICATION")
    print("="*60)
    
    checks = [
        ("Python Version", check_python_version),
        ("TensorFlow", lambda: check_package("tensorflow", "tensorflow")),
        ("NumPy", lambda: check_package("numpy")),
        ("Pillow", lambda: check_package("Pillow", "PIL")),
        ("Kaggle Package", lambda: check_package("kaggle")),
        ("Kaggle Configuration", check_kaggle_config),
        ("Kaggle CLI", check_kaggle_cli),
        ("Dataset", check_dataset),
    ]
    
    results = {}
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"✗ Error: {e}")
            results[check_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if results.get("Dataset"):
        print("\n✓ Ready to train: python train_complete.py")
    else:
        print("\n⚠ Missing dataset. Run setup first:")
        print("  python setup_kaggle.py")
        print("\nOr use the master orchestrator:")
        print("  python run_complete_pipeline.py")
    
    return all(results.values())

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
