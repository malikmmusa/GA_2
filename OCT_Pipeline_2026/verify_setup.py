#!/usr/bin/env python3
"""
Utility script to verify the OCT Pipeline setup.
Run this after installation to check everything is configured correctly.
"""

import sys
from pathlib import Path
import importlib.util

def check_mark(condition):
    return "✓" if condition else "✗"

def check_installation():
    """Check if all dependencies are installed."""
    print("=" * 60)
    print("OCT Pipeline 2026 - Setup Verification")
    print("=" * 60)
    print()
    
    # Check Python version
    print("1. Python Version:")
    py_version = sys.version_info
    py_ok = py_version.major == 3 and py_version.minor >= 9
    print(f"   {check_mark(py_ok)} Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    if not py_ok:
        print("   ⚠️  Warning: Python 3.9+ recommended")
    print()
    
    # Check required packages
    print("2. Required Packages:")
    required_packages = [
        'torch', 'torchvision', 'cv2', 'PIL', 'numpy', 
        'pandas', 'matplotlib', 'tqdm', 'albumentations'
    ]
    
    all_installed = True
    for package in required_packages:
        pkg_name = 'opencv-python' if package == 'cv2' else package
        spec = importlib.util.find_spec(package)
        installed = spec is not None
        all_installed = all_installed and installed
        status = check_mark(installed)
        print(f"   {status} {package}")
    print()
    
    # Check directory structure
    print("3. Directory Structure:")
    base_dir = Path(__file__).parent
    required_dirs = [
        'data/raw',
        'data/processed/b_scans',
        'data/processed/en_face',
        'data/csv',
        'src/models',
        'src/utils',
        'notebooks'
    ]
    
    all_dirs_exist = True
    for dir_path in required_dirs:
        full_path = base_dir / dir_path
        exists = full_path.exists()
        all_dirs_exist = all_dirs_exist and exists
        print(f"   {check_mark(exists)} {dir_path}/")
    print()
    
    # Check source files
    print("4. Source Files:")
    required_files = [
        'src/01_split_data.py',
        'src/02_train_disc.py',
        'src/03_train_fovea.py',
        'src/04_train_ga.py',
        'src/models/unet.py',
        'src/utils/gaussian_utils.py'
    ]
    
    all_files_exist = True
    for file_path in required_files:
        full_path = base_dir / file_path
        exists = full_path.exists()
        all_files_exist = all_files_exist and exists
        print(f"   {check_mark(exists)} {file_path}")
    print()
    
    # Check data
    print("5. Data Status:")
    raw_images = list((base_dir / 'data' / 'raw').glob('*.jpg'))
    raw_images.extend(list((base_dir / 'data' / 'raw').glob('*.png')))
    print(f"   • Raw images: {len(raw_images)}")
    
    b_scans = list((base_dir / 'data' / 'processed' / 'b_scans').glob('*.jpg'))
    en_faces = list((base_dir / 'data' / 'processed' / 'en_face').glob('*.jpg'))
    print(f"   • B-scans: {len(b_scans)}")
    print(f"   • En face: {len(en_faces)}")
    
    csv_files = list((base_dir / 'data' / 'csv').glob('*.csv'))
    print(f"   • Label CSVs: {len(csv_files)}")
    for csv_file in csv_files:
        print(f"     - {csv_file.name}")
    print()
    
    # Check models
    print("6. Trained Models:")
    model_dir = base_dir / 'models'
    if model_dir.exists():
        model_files = list(model_dir.glob('*.pth'))
        if model_files:
            for model_file in model_files:
                size_mb = model_file.stat().st_size / (1024 * 1024)
                print(f"   ✓ {model_file.name} ({size_mb:.1f} MB)")
        else:
            print("   ℹ️  No trained models yet (run training scripts)")
    else:
        print("   ℹ️  Models directory will be created during training")
    print()
    
    # Summary
    print("=" * 60)
    print("Summary:")
    print("=" * 60)
    
    if all_installed and all_dirs_exist and all_files_exist:
        print("✓ Setup Complete! All checks passed.")
        print()
        print("Next Steps:")
        print("  1. Place composite images in data/raw/")
        print("  2. Create label CSVs in data/csv/")
        print("  3. Run: python src/01_split_data.py")
        print("  4. Train models using scripts 02-04")
    else:
        print("⚠️  Some issues detected. Please review the checks above.")
        if not all_installed:
            print("   • Install missing packages: pip install -r requirements.txt")
        if not all_dirs_exist or not all_files_exist:
            print("   • Some files/directories are missing")
    print()
    
    return all_installed and all_dirs_exist and all_files_exist


if __name__ == "__main__":
    success = check_installation()
    sys.exit(0 if success else 1)
