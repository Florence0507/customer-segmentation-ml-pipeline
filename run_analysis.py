#!/usr/bin/env python3
"""
Complete Analysis Pipeline
Runs the entire regularized regression analysis from start to finish.

Usage:
    python run_analysis.py [--no-plots] [--output-dir results]
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from preprocessing import preprocess_pipeline
from model_training import full_training_pipeline
from visualization import create_all_visualizations


def main():
    parser = argparse.ArgumentParser(
        description='Run complete regularized regression analysis'
    )
    parser.add_argument(
        '--no-plots', 
        action='store_true',
        help='Skip generating visualizations'
    )
    parser.add_argument(
        '--output-dir',
        default='results',
        help='Directory to save results (default: results)'
    )
    parser.add_argument(
        '--data-path',
        default='data/job_satisfaction_data.csv',
        help='Path to dataset (default: data/job_satisfaction_data.csv)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("REGULARIZED REGRESSION ANALYSIS PIPELINE")
    print("="*70)
    
    # Step 1: Preprocessing
    print("\n[1/3] Preprocessing data...")
    data = preprocess_pipeline(args.data_path, verbose=True)
    
    # Step 2: Model Training
    print("\n[2/3] Training and evaluating models...")
    results = full_training_pipeline(data)
    
    # Step 3: Visualization
    if not args.no_plots:
        print(f"\n[3/3] Generating visualizations (output: {args.output_dir})...")
        create_all_visualizations(results, data, output_dir=args.output_dir)
    else:
        print("\n[3/3] Skipping visualizations (--no-plots specified)")
    
    # Print summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("="*70)
    print("\nBest Model on Test Set:")
    best_model = results['test_results'].loc[results['test_results']['R²'].idxmax()]
    print(f"  {best_model['Model']}")
    print(f"  Alpha: {best_model['Alpha']}")
    print(f"  R²:    {best_model['R²']:.4f}")
    print(f"  MSE:   {best_model['MSE']:.4f}")
    print(f"  MAE:   {best_model['MAE']:.4f}")
    
    print("\nTop 3 Most Important Features:")
    top_features = results['coefficients'].head(3)
    for idx, row in top_features.iterrows():
        print(f"  {row['Feature']}: {row['Ridge']:.3f}")
    
    if not args.no_plots:
        print(f"\nVisualizations saved to: {args.output_dir}/")
    
    print("\n" + "="*70)
    print("Success! Analysis pipeline completed.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
