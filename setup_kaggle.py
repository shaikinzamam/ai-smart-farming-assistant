#!/usr/bin/env python3
"""
AI Smart Farming - Kaggle API Setup & Dataset Download
Handles Kaggle configuration, dataset download, and organization
"""

import os
import shutil
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, List
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KaggleSetup:
    """Manages Kaggle API configuration and dataset operations"""
    
    def __init__(self):
        self.home_dir = Path.home()
        self.kaggle_dir = self.home_dir / '.kaggle'
        self.kaggle_json_home = self.kaggle_dir / 'kaggle.json'
        self.kaggle_json_project = Path.cwd() / 'kaggle.json'
        self.dataset_dir = Path.cwd() / 'dataset'
        
    def verify_and_setup_kaggle_json(self) -> bool:
        """
        Verify Kaggle API setup:
        - Ensure kaggle.json is located at C:\Users\<username>\.kaggle\kaggle.json
        - Move it from project folder if needed
        - Set proper permissions
        """
        logger.info("=" * 60)
        logger.info("STEP 1: Verifying Kaggle API Setup")
        logger.info("=" * 60)
        
        try:
            # Check if kaggle.json exists in project folder
            if not self.kaggle_json_project.exists():
                logger.error(f"❌ kaggle.json not found in project folder: {self.kaggle_json_project}")
                return False
            
            logger.info(f"✓ kaggle.json found in project: {self.kaggle_json_project}")
            
            # Create .kaggle directory if it doesn't exist
            self.kaggle_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ .kaggle directory ready: {self.kaggle_dir}")
            
            # Copy kaggle.json to the correct location
            if self.kaggle_json_home.exists():
                logger.info(f"✓ kaggle.json already exists at: {self.kaggle_json_home}")
                # Verify it has valid credentials
                self._verify_kaggle_json(self.kaggle_json_home)
            else:
                shutil.copy2(self.kaggle_json_project, self.kaggle_json_home)
                logger.info(f"✓ Copied kaggle.json to: {self.kaggle_json_home}")
            
            # Set permissions (Windows - restricted to owner)
            if sys.platform == 'win32':
                os.chmod(self.kaggle_json_home, 0o600)
                logger.info("✓ Permissions set correctly (owner read/write only)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error during Kaggle setup: {e}")
            return False
    
    def _verify_kaggle_json(self, path: Path) -> bool:
        """Verify kaggle.json contains valid credentials"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                if 'username' in data and 'key' in data:
                    logger.info(f"✓ Valid credentials found (username: {data['username']})")
                    return True
                else:
                    logger.error("❌ Invalid kaggle.json format - missing 'username' or 'key'")
                    return False
        except json.JSONDecodeError:
            logger.error("❌ Invalid JSON format in kaggle.json")
            return False
    
    def validate_kaggle_cli(self) -> bool:
        """
        Validate Kaggle CLI:
        - Run: kaggle datasets list
        - Fix PATH or environment issues if needed
        """
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: Validating Kaggle CLI")
        logger.info("=" * 60)
        
        try:
            # Test kaggle command
            result = subprocess.run(
                ['kaggle', 'datasets', 'list', '--max-results', '1'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info("✓ Kaggle CLI is working correctly")
                logger.info(f"  Output preview: {result.stdout[:200]}")
                return True
            else:
                logger.error(f"❌ Kaggle CLI error: {result.stderr}")
                # Try to install/update kaggle
                logger.info("  Attempting to install/update kaggle...")
                return self._install_kaggle_cli()
                
        except FileNotFoundError:
            logger.error("❌ Kaggle CLI not found in PATH")
            logger.info("  Attempting to install kaggle...")
            return self._install_kaggle_cli()
        except subprocess.TimeoutExpired:
            logger.error("❌ Kaggle CLI request timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            return False
    
    def _install_kaggle_cli(self) -> bool:
        """Install or update kaggle CLI"""
        try:
            logger.info("  Running: pip install --upgrade kaggle")
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '--upgrade', 'kaggle'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                logger.info("✓ Kaggle CLI installed/updated successfully")
                return True
            else:
                logger.error(f"❌ Failed to install kaggle: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"❌ Error installing kaggle: {e}")
            return False
    
    def download_datasets(self) -> bool:
        """
        Download smaller datasets (~500 MB):
        - Tomato disease dataset (lightweight)
        - Banana disease dataset (lightweight)
        """
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: Downloading Lightweight Datasets (~500 MB)")
        logger.info("=" * 60)
        
        # Create dataset directory
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Smaller, curated datasets from Kaggle
        datasets_to_download = [
            {
                'kaggle_name': 'kaustubh2020/tomato-disease-detection-deep-learning',
                'local_name': 'tomato_dataset',
                'description': 'Tomato Disease Detection Dataset (~250 MB)'
            },
            {
                'kaggle_name': 'c0a621a3da2c2e68f88bb4d44ffc33f8a6d5dae8b30eebd5c5577ad9fcfe8a88',
                'local_name': 'banana_dataset',
                'description': 'Banana Disease Detection Dataset (~250 MB)'
            }
        ]
        
        success = True
        for dataset in datasets_to_download:
            if not self._download_single_dataset(dataset):
                success = False
        
        return success
    
    def _download_single_dataset(self, dataset: dict) -> bool:
        """Download a single dataset from Kaggle"""
        try:
            dataset_path = self.dataset_dir / dataset['local_name']
            
            # Skip if already downloaded
            if dataset_path.exists():
                logger.info(f"✓ {dataset['description']} already exists")
                return True
            
            logger.info(f"📥 Downloading: {dataset['description']}")
            logger.info(f"   Kaggle: {dataset['kaggle_name']}")
            
            result = subprocess.run(
                [
                    'kaggle', 'datasets', 'download', '-d', dataset['kaggle_name'],
                    '-p', str(self.dataset_dir)
                ],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                logger.info(f"✓ Downloaded: {dataset['description']}")
                
                # Extract if zip file was created
                zip_file = self.dataset_dir / f"{dataset['local_name']}.zip"
                if zip_file.exists():
                    logger.info(f"  Extracting: {zip_file.name}")
                    shutil.unpack_archive(zip_file, dataset_path)
                    zip_file.unlink()
                    logger.info(f"✓ Extracted to: {dataset_path}")
                
                return True
            else:
                logger.error(f"❌ Failed to download {dataset['description']}")
                logger.error(f"  Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Download timeout for {dataset['description']}")
            return False
        except Exception as e:
            logger.error(f"❌ Error downloading {dataset['description']}: {e}")
            return False
    
    def extract_and_organize_datasets(self) -> bool:
        """
        Extract and organize datasets into:
        dataset/
          tomato/
            early_blight/
            late_blight/
            leaf_mold/
            healthy/
          banana/
            sigatoka/
            panama_disease/
            healthy/
        """
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: Extracting and Organizing Datasets")
        logger.info("=" * 60)
        
        try:
            # Create target structure
            tomato_dir = self.dataset_dir / 'tomato'
            banana_dir = self.dataset_dir / 'banana'
            
            # Process Tomato dataset
            tomato_raw = self.dataset_dir / 'tomato_dataset'
            if tomato_raw.exists():
                logger.info("📁 Processing Tomato dataset...")
                self._organize_tomato_dataset(tomato_raw, tomato_dir)
            else:
                logger.warning("⚠ Tomato dataset not found")
            
            # Process Banana dataset
            banana_raw = self.dataset_dir / 'banana_dataset'
            if banana_raw.exists():
                logger.info("📁 Processing Banana dataset...")
                self._organize_banana_dataset(banana_raw, banana_dir)
            else:
                logger.warning("⚠ Banana dataset not found")
            
            logger.info("✓ Dataset organization complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error organizing datasets: {e}")
            return False
    
    def _organize_tomato_dataset(self, source: Path, target: Path) -> None:
        """Organize Tomato disease dataset"""
        target.mkdir(parents=True, exist_ok=True)
        
        # Copy or reorganize class directories
        for item in source.iterdir():
            if item.is_dir():
                class_name = item.name.lower()
                
                # Map class names appropriately
                if 'blight_early' in class_name or 'early' in class_name:
                    target_class = target / 'early_blight'
                elif 'blight_late' in class_name or 'late' in class_name:
                    target_class = target / 'late_blight'
                elif 'mold' in class_name:
                    target_class = target / 'leaf_mold'
                elif 'healthy' in class_name or 'normal' in class_name:
                    target_class = target / 'healthy'
                else:
                    continue
                
                if not target_class.exists():
                    shutil.copytree(item, target_class)
                    logger.info(f"  ✓ Organized {target_class.name}")
    
    def _organize_banana_dataset(self, source: Path, target: Path) -> None:
        """Organize Banana disease dataset"""
        target.mkdir(parents=True, exist_ok=True)
        
        # Copy or reorganize class directories
        for item in source.iterdir():
            if item.is_dir():
                class_name = item.name.lower()
                
                if 'sigatoka' in class_name:
                    target_class = target / 'sigatoka'
                elif 'panama' in class_name:
                    target_class = target / 'panama_disease'
                elif 'healthy' in class_name or 'normal' in class_name:
                    target_class = target / 'healthy'
                else:
                    continue
                
                if not target_class.exists():
                    shutil.copytree(item, target_class)
                    logger.info(f"  ✓ Organized {target_class.name}")
    
    def clean_dataset(self) -> bool:
        """
        Clean dataset:
        - Remove corrupted images
        - Ensure all images are JPG/PNG
        - Print count per class
        """
        logger.info("\n" + "=" * 60)
        logger.info("STEP 5: Cleaning and Validating Datasets")
        logger.info("=" * 60)
        
        try:
            from PIL import Image
            
            stats = {
                'tomato': {},
                'banana': {}
            }
            
            # Clean Tomato dataset
            tomato_dir = self.dataset_dir / 'tomato'
            if tomato_dir.exists():
                logger.info("🧹 Cleaning Tomato dataset...")
                stats['tomato'] = self._clean_image_directory(tomato_dir)
            
            # Clean Banana dataset
            banana_dir = self.dataset_dir / 'banana'
            if banana_dir.exists():
                logger.info("🧹 Cleaning Banana dataset...")
                stats['banana'] = self._clean_image_directory(banana_dir)
            
            # Print summary
            self._print_dataset_summary(stats)
            return True
            
        except ImportError:
            logger.error("❌ PIL (Pillow) not installed. Installing...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'Pillow'],
                          capture_output=True)
            return self.clean_dataset()  # Retry after installing
        except Exception as e:
            logger.error(f"❌ Error cleaning dataset: {e}")
            return False
    
    def _clean_image_directory(self, directory: Path) -> dict:
        """Clean images in directory and return statistics"""
        from PIL import Image
        
        stats = {}
        valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        
        for class_dir in sorted(directory.iterdir()):
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            valid_count = 0
            corrupted_count = 0
            removed_count = 0
            
            for image_file in class_dir.glob('*'):
                if image_file.suffix.lower() not in valid_extensions:
                    image_file.unlink()
                    removed_count += 1
                    continue
                
                try:
                    # Try to open and verify image
                    with Image.open(image_file) as img:
                        img.verify()
                    valid_count += 1
                except Exception as e:
                    logger.warning(f"    Removing corrupted: {image_file.name}")
                    image_file.unlink()
                    corrupted_count += 1
                    removed_count += 1
            
            stats[class_name] = {
                'valid': valid_count,
                'removed': removed_count,
                'total': valid_count + removed_count
            }
            
            logger.info(f"  {class_name}: {valid_count} valid images, {removed_count} removed")
        
        return stats
    
    def _print_dataset_summary(self, stats: dict) -> None:
        """Print summary of dataset statistics"""
        logger.info("\n" + "=" * 60)
        logger.info("DATASET SUMMARY")
        logger.info("=" * 60)
        
        for dataset_name, classes in stats.items():
            if classes:
                logger.info(f"\n{dataset_name.upper()} Dataset:")
                total_images = sum(c['valid'] for c in classes.values())
                logger.info(f"  Total images: {total_images}")
                for class_name, counts in sorted(classes.items()):
                    logger.info(f"    {class_name}: {counts['valid']} images")


def main():
    """Run complete setup"""
    setup = KaggleSetup()
    
    # Step 1: Verify Kaggle API setup
    if not setup.verify_and_setup_kaggle_json():
        logger.error("❌ Failed to setup Kaggle API")
        return False
    
    # Step 2: Validate Kaggle CLI
    if not setup.validate_kaggle_cli():
        logger.error("❌ Kaggle CLI validation failed")
        return False
    
    # Step 3: Download datasets
    if not setup.download_datasets():
        logger.warning("⚠ Some datasets failed to download")
    
    # Step 4: Extract and organize
    if not setup.extract_and_organize_datasets():
        logger.error("❌ Failed to organize datasets")
        return False
    
    # Step 5: Clean datasets
    if not setup.clean_dataset():
        logger.error("❌ Failed to clean datasets")
        return False
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ ALL SETUP STEPS COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info("\nNext step: Run training with:")
    logger.info("  python train.py --epochs 15 --batch_size 16")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
