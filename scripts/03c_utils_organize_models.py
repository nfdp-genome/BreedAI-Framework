#!/usr/bin/env python3
"""
File: 03c_utils_organize_models.py
Purpose: Organize and summarize trained models from array jobs
"""

import pandas as pd
import numpy as np
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

def organize_trained_models(models_dir, output_summary):
    """Organize trained models and create summary"""
    
    models_path = Path(models_dir)
    
    if not models_path.exists():
        print(f"Models directory not found: {models_dir}")
        return
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("Organizing trained models...")
    
    # Collect all models and their performance
    all_models = {}
    performance_summary = []
    
    for trait_dir in models_path.iterdir():
        if trait_dir.is_dir():
            trait_name = trait_dir.name.replace('_', ' ')
            trait_models = {}
            
            # Find all models for this trait
            for model_file in trait_dir.glob('*.joblib'):
                algorithm_name = model_file.stem
                
                # Find corresponding performance file
                perf_file = trait_dir / f"{algorithm_name}_performance.json"
                if perf_file.exists():
                    with open(perf_file, 'r') as f:
                        performance = json.load(f)
                    
                    trait_models[algorithm_name] = {
                        'model_file': str(model_file),
                        'performance_file': str(perf_file),
                        'performance': performance
                    }
                    
                    # Add to overall performance summary
                    summary_entry = performance.copy()
                    summary_entry['trait'] = trait_name
                    performance_summary.append(summary_entry)
                else:
                    logger.warning(f"No performance file found for {algorithm_name} on {trait_name}")
            
            if trait_models:
                all_models[trait_name] = trait_models
                logger.info(f"Found {len(trait_models)} models for {trait_name}")
    
    # Create comprehensive summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_traits': len(all_models),
        'total_models': sum(len(models) for models in all_models.values()),
        'traits': list(all_models.keys()),
        'models_by_trait': {trait: len(models) for trait, models in all_models.items()},
        'performance_summary': performance_summary
    }
    
    # Save summary
    with open(output_summary, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Training summary saved: {output_summary}")
    logger.info(f"Total: {summary['total_models']} models for {summary['total_traits']} traits")
    
    # Create performance DataFrame and save as CSV
    if performance_summary:
        perf_df = pd.DataFrame(performance_summary)
        csv_file = Path(output_summary).parent / "deployment_performance_summary.csv"
        perf_df.to_csv(csv_file, index=False)
        logger.info(f"Performance CSV saved: {csv_file}")
        
        # Print top performers
        logger.info("\nTop 10 performing models:")
        top_10 = perf_df.nlargest(10, 'cv_r2_mean')
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            logger.info(f"{i:2d}. {row['algorithm']:<15} on {row['trait']:<20}: CV R² = {row['cv_r2_mean']:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Organize trained models')
    parser.add_argument('--models_dir', required=True, help='Directory with trained models')
    parser.add_argument('--output_summary', required=True, help='Output summary file (JSON)')
    
    args = parser.parse_args()
    
    organize_trained_models(args.models_dir, args.output_summary)

if __name__ == "__main__":
    main()