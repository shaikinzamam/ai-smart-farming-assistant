#!/usr/bin/env python3
"""
AI Smart Farming - Complete Training Pipeline
Integrates setup, data preparation, model training, and evaluation
"""

import argparse
import json
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Complete training pipeline for AI Smart Farming"""
    
    def __init__(self, data_dir: Path = None, epochs: int = 15, batch_size: int = 16):
        self.base_dir = Path(__file__).resolve().parent
        self.data_dir = data_dir or self.base_dir / 'dataset'
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_dir = self.base_dir / 'models'
        self.log_dir = self.base_dir / 'logs'
        
        # Create necessary directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def precheck_dependencies(self) -> bool:
        """Pre-check all required dependencies"""
        logger.info("=" * 60)
        logger.info("PRE-CHECK: Verifying Dependencies")
        logger.info("=" * 60)
        
        required_packages = [
            ('tensorflow', 'TensorFlow'),
            ('numpy', 'NumPy'),
            ('PIL', 'Pillow'),
            ('kaggle', 'Kaggle CLI'),
        ]
        
        missing = []
        for package, name in required_packages:
            try:
                __import__(package)
                logger.info(f"✓ {name} is installed")
            except ImportError:
                logger.warning(f"✗ {name} is NOT installed")
                missing.append(package)
        
        if missing:
            logger.info(f"\nInstalling missing packages: {', '.join(missing)}")
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install'] + missing,
                capture_output=True
            )
            if result.returncode == 0:
                logger.info("✓ Missing packages installed")
                return True
            else:
                logger.error("✗ Failed to install packages")
                return False
        
        return True
    
    def verify_dataset_structure(self) -> bool:
        """Verify dataset has the expected structure"""
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: Verifying Dataset Structure")
        logger.info("=" * 60)
        
        required_dirs = {
            "tomato/early_blight",
            "tomato/late_blight", 
            "tomato/leaf_mold",
            "tomato/healthy",
            "banana/sigatoka",
            "banana/panama_disease",
            "banana/healthy",
        }
        
        missing = []
        for subdir in required_dirs:
            full_path = self.data_dir / subdir
            if not full_path.exists():
                missing.append(subdir)
            else:
                image_count = len(list(full_path.glob('*.jpg'))) + \
                             len(list(full_path.glob('*.png')))
                logger.info(f"  ✓ {subdir}: {image_count} images")
        
        if missing:
            logger.error(f"\n✗ Missing directories:")
            for m in missing:
                logger.error(f"    {m}")
            logger.error(f"\nYou need to run setup_kaggle.py first to download datasets")
            return False
        
        logger.info("✓ Dataset structure verified")
        return True
    
    def prepare_training_data(self) -> Dict[str, int]:
        """
        Prepare datasets using Keras ImageDataGenerator
        Returns class distribution
        """
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: Preparing Training Data")
        logger.info("=" * 60)
        
        try:
            import numpy as np
            import tensorflow as tf
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            
            # Create data generators
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest',
                validation_split=0.2
            )
            
            logger.info(f"Loading images from: {self.data_dir}")
            
            # Load all images into a single generator
            image_dataset = train_datagen.flow_from_directory(
                self.data_dir,
                target_size=(224, 224),
                batch_size=self.batch_size,
                class_mode='categorical',
                subset='training'
            )
            
            val_dataset = train_datagen.flow_from_directory(
                self.data_dir,
                target_size=(224, 224),
                batch_size=self.batch_size,
                class_mode='categorical',
                subset='validation'
            )
            
            # Store class information
            class_indices = image_dataset.class_indices
            self.class_indices = class_indices
            self.num_classes = len(class_indices)
            
            logger.info(f"✓ Data prepared successfully")
            logger.info(f"  Classes: {self.num_classes}")
            logger.info(f"  Training batches: {len(image_dataset)}")
            logger.info(f"  Validation batches: {len(val_dataset)}")
            logger.info(f"\nClass Distribution:")
            for class_name, idx in sorted(class_indices.items(), key=lambda x: x[1]):
                logger.info(f"  {idx}: {class_name}")
            
            return image_dataset, val_dataset
            
        except Exception as e:
            logger.error(f"✗ Error preparing data: {e}")
            return None, None
    
    def build_model(self) -> 'tf.keras.Model':
        """Build MobileNetV2 transfer learning model"""
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: Building Model Architecture")
        logger.info("=" * 60)
        
        try:
            import tensorflow as tf
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras import layers, models
            
            # Load pre-trained MobileNetV2
            logger.info("Loading pre-trained MobileNetV2 (ImageNet weights)...")
            base_model = MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=False,
                weights='imagenet'
            )
            
            # Freeze base model initially
            base_model.trainable = False
            
            # Build custom head
            logger.info("Building custom classification head...")
            model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(self.num_classes, activation='softmax')
            ])
            
            # Compile
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
            )
            
            logger.info("✓ Model built successfully")
            model.summary()
            
            return model
            
        except Exception as e:
            logger.error(f"✗ Error building model: {e}")
            return None
    
    def train_model(self, model, train_data, val_data) -> bool:
        """Train the model with callbacks"""
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: Training Model")
        logger.info("=" * 60)
        logger.info(f"Configuration:")
        logger.info(f"  Epochs: {self.epochs}")
        logger.info(f"  Batch Size: {self.batch_size}")
        logger.info(f"  Learning Rate: 0.001 (Adam default)")
        
        try:
            import tensorflow as tf
            from tensorflow.keras.callbacks import (
                ModelCheckpoint,
                EarlyStopping,
                ReduceLROnPlateau,
                TensorBoard,
            )
            
            callbacks = [
                ModelCheckpoint(
                    str(self.model_dir / 'plant_model.h5'),
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                ),
                EarlyStopping(
                    monitor='val_accuracy',
                    patience=5,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-7,
                    verbose=1
                ),
                TensorBoard(
                    log_dir=str(self.log_dir),
                    histogram_freq=1,
                    write_graph=True
                )
            ]
            
            logger.info("\n📚 Starting training (Phase 1: Frozen Base Model)")
            logger.info("-" * 60)
            
            history = model.fit(
                train_data,
                epochs=self.epochs,
                validation_data=val_data,
                callbacks=callbacks,
                verbose=1
            )
            
            logger.info("\n✓ Training completed")
            
            # Save metadata
            self._save_metadata(history)
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def evaluate_model(self, model, val_data) -> Dict:
        """Evaluate model on validation set"""
        logger.info("\n" + "=" * 60)
        logger.info("STEP 5: Evaluating Model")
        logger.info("=" * 60)
        
        try:
            results = model.evaluate(val_data, verbose=1)
            
            eval_dict = {
                'loss': float(results[0]),
                'accuracy': float(results[1]),
                'top_2_accuracy': float(results[2]) if len(results) > 2 else None
            }
            
            logger.info(f"\n✓ Evaluation Results:")
            logger.info(f"  Loss: {eval_dict['loss']:.4f}")
            logger.info(f"  Accuracy: {eval_dict['accuracy']:.4f} ({eval_dict['accuracy']*100:.2f}%)")
            if eval_dict['top_2_accuracy']:
                logger.info(f"  Top-2 Accuracy: {eval_dict['top_2_accuracy']:.4f}")
            
            return eval_dict
            
        except Exception as e:
            logger.error(f"✗ Evaluation failed: {e}")
            return {}
    
    def _save_metadata(self, history=None):
        """Save model metadata"""
        metadata = {
            'model_path': str(self.model_dir / 'plant_model.h5'),
            'classes': self.class_indices,
            'num_classes': self.num_classes,
            'image_size': (224, 224),
            'training_config': {
                'epochs': self.epochs,
                'batch_size': self.batch_size,
            }
        }
        
        if history:
            metadata['training_history'] = {
                'accuracy': [float(a) for a in history.history.get('accuracy', [])],
                'val_accuracy': [float(a) for a in history.history.get('val_accuracy', [])],
                'loss': [float(l) for l in history.history.get('loss', [])],
                'val_loss': [float(l) for l in history.history.get('val_loss', [])],
            }
        
        metadata_path = self.model_dir / 'model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"\n✓ Metadata saved to: {metadata_path}")
    
    def run_complete_pipeline(self) -> bool:
        """Execute the complete training pipeline"""
        logger.info("\n")
        logger.info("╔" + "=" * 58 + "╗")
        logger.info("║" + " " * 58 + "║")
        logger.info("║" + "  AI SMART FARMING - COMPLETE TRAINING PIPELINE".center(58) + "║")
        logger.info("║" + " " * 58 + "║")
        logger.info("╚" + "=" * 58 + "╝")
        
        # Pre-check
        if not self.precheck_dependencies():
            logger.error("✗ Dependency check failed")
            return False
        
        # Verify dataset
        if not self.verify_dataset_structure():
            logger.error("\n⚠ Dataset not found. Run setup_kaggle.py first:")
            logger.error("   python setup_kaggle.py")
            return False
        
        # Prepare data
        train_data, val_data = self.prepare_training_data()
        if train_data is None:
            logger.error("✗ Data preparation failed")
            return False
        
        # Build model
        model = self.build_model()
        if model is None:
            logger.error("✗ Model building failed")
            return False
        
        # Train model
        if not self.train_model(model, train_data, val_data):
            logger.error("✗ Training failed")
            return False
        
        # Evaluate model
        eval_results = self.evaluate_model(model, val_data)
        
        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("✓ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"\n📦 Model saved to: {self.model_dir / 'plant_model.h5'}")
        logger.info(f"📊 Logs saved to: {self.log_dir}")
        logger.info(f"📋 Metadata saved to: {self.model_dir / 'model_metadata.json'}")
        
        if eval_results:
            logger.info(f"\n📈 Final Metrics:")
            logger.info(f"   Validation Accuracy: {eval_results['accuracy']*100:.2f}%")
            logger.info(f"   Validation Loss: {eval_results['loss']:.4f}")
        
        logger.info(f"\n🚀 Next steps:")
        logger.info(f"   1. Start the backend server:")
        logger.info(f"      python -m backend.main")
        logger.info(f"   2. Open the frontend:")
        logger.info(f"      Open frontend/index.html in your browser")
        
        return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='AI Smart Farming - Complete Training Pipeline'
    )
    parser.add_argument(
        '--data_dir',
        type=Path,
        default=None,
        help='Path to dataset directory (default: ./dataset)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=15,
        help='Number of training epochs (default: 15)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for training (default: 16)'
    )
    
    args = parser.parse_args()
    
    pipeline = TrainingPipeline(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    success = pipeline.run_complete_pipeline()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
