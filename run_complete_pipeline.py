#!/usr/bin/env python3
"""
AI Smart Farming - Master Orchestration Script
Runs complete setup → training pipeline with error handling and recovery
"""

import subprocess
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MasterOrchestrator:
    """Orchestrates complete AI Smart Farming setup and training"""
    
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.dataset_dir = self.base_dir / 'dataset'
    
    def run_step(self, script: str, args: list = None, description: str = None) -> Tuple[bool, str]:
        """
        Run a step script with error handling
        Returns: (success, output_message)
        """
        script_path = self.base_dir / script
        
        if not script_path.exists():
            return False, f"Script not found: {script}"
        
        cmd = [sys.executable, str(script_path)]
        if args:
            cmd.extend(args)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {description or script}")
        logger.info(f"{'='*60}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=False,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                return True, f"✓ {description or script} completed"
            else:
                return False, f"✗ {description or script} failed (exit code: {result.returncode})"
                
        except subprocess.TimeoutExpired:
            return False, f"✗ {description or script} timed out"
        except Exception as e:
            return False, f"✗ Error running {description or script}: {e}"
    
    def verify_setup(self) -> bool:
        """Quick verification that setup is complete"""
        logger.info(f"\n{'='*60}")
        logger.info("Verifying Setup")
        logger.info(f"{'='*60}")
        
        checks = {
            'Kaggle config': self.base_dir / '..' / '.kaggle' / 'kaggle.json',
            'Models directory': self.base_dir / 'models',
            'Dataset directory': self.dataset_dir,
        }
        
        all_good = True
        for check_name, path in checks.items():
            if path.exists():
                logger.info(f"✓ {check_name}: {path}")
            else:
                logger.warning(f"⚠ {check_name} not ready: {path}")
                all_good = False
        
        return all_good
    
    def run_full_pipeline(self, epochs: int = 15, batch_size: int = 16) -> bool:
        """
        Execute complete pipeline:
        1. Kaggle setup and dataset download
        2. Data preparation and cleaning
        3. Model training
        4. Evaluation and results
        """
        
        logger.info("\n")
        logger.info("╔" + "="*58 + "╗")
        logger.info("║" + " "*58 + "║")
        logger.info("║" + "  AI SMART FARMING - MASTER ORCHESTRATOR".center(58) + "║")
        logger.info("║" + " "*58 + "║")
        logger.info("║" + "  Complete Setup → Download → Train → Evaluate".center(58) + "║")
        logger.info("║" + " "*58 + "║")
        logger.info("╚" + "="*58 + "╝")
        
        steps_completed = 0
        total_steps = 2
        
        # Step 1: Kaggle Setup & Download
        success, msg = self.run_step(
            'setup_kaggle.py',
            description='[1/2] Kaggle Setup & Dataset Download'
        )
        logger.info(msg)
        if success:
            steps_completed += 1
        else:
            logger.error("⚠ Setup failed - attempting to continue with existing data...")
        
        # Step 2: Complete Training Pipeline
        train_args = [
            '--epochs', str(epochs),
            '--batch_size', str(batch_size)
        ]
        
        success, msg = self.run_step(
            'train_complete.py',
            args=train_args,
            description='[2/2] Complete Training Pipeline'
        )
        logger.info(msg)
        if success:
            steps_completed += 1
        else:
            logger.error("✗ Training failed")
            return False
        
        # Final summary
        logger.info("\n" + "="*60)
        logger.info("FINAL SUMMARY")
        logger.info("="*60)
        logger.info(f"Steps completed: {steps_completed}/{total_steps}")
        
        if steps_completed == total_steps:
            logger.info("\n✓ ALL STEPS COMPLETED SUCCESSFULLY!")
            logger.info("\n🎉 Your AI Smart Farming model is ready!")
            logger.info("\nTo start the application:")
            logger.info("  1. Start backend: python -m backend.main")
            logger.info("  2. Open frontend: frontend/index.html")
            return True
        else:
            logger.error("\n✗ Some steps failed - see above for details")
            return False


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='AI Smart Farming Master Orchestrator'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=15,
        help='Training epochs (default: 15)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size (default: 16)'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify setup without running training'
    )
    
    args = parser.parse_args()
    orchestrator = MasterOrchestrator()
    
    if args.verify_only:
        success = orchestrator.verify_setup()
        sys.exit(0 if success else 1)
    
    success = orchestrator.run_full_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
