#!/usr/bin/env python3
"""
File: 02d_phase1_summary_benchmarking.py
Purpose: Create comprehensive benchmarking summary and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json
from datetime import datetime

def create_benchmarking_summary(input_dir, output_dir):
    """Create comprehensive benchmarking summary"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find the most recent results file
    results_files = list(input_path.glob("combined_benchmarking_results_*.csv"))
    if not results_files:
        print("No benchmarking results found!")
        return
    
    latest_results = max(results_files, key=lambda x: x.stat().st_mtime)
    print(f"Processing results from: {latest_results}")
    
    # Load results
    results_df = pd.read_csv(latest_results)
    
    # Generate visualizations
    print("Creating visualizations...")
    
    # 1. Performance comparison by trait
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Genomic Prediction Benchmarking Summary', fontsize=16, fontweight='bold')
    
    # Performance by trait
    ax1 = axes[0, 0]
    trait_performance = results_df.groupby('trait')['test_pearson_r'].agg(['mean', 'max']).reset_index()
    x_pos = np.arange(len(trait_performance))
    
    ax1.bar(x_pos - 0.2, trait_performance['mean'], 0.4, label='Mean', alpha=0.7)
    ax1.bar(x_pos + 0.2, trait_performance['max'], 0.4, label='Best', alpha=0.7)
    ax1.set_xlabel('Trait')
    ax1.set_ylabel('Pearson Correlation')
    ax1.set_title('Performance by Trait')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(trait_performance['trait'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Algorithm performance
    ax2 = axes[0, 1]
    alg_performance = results_df.groupby('algorithm')['test_pearson_r'].mean().sort_values(ascending=False)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(alg_performance)))
    bars = ax2.barh(range(len(alg_performance)), alg_performance.values, color=colors, alpha=0.8)
    ax2.set_yticks(range(len(alg_performance)))
    ax2.set_yticklabels(alg_performance.index, fontsize=8)
    ax2.set_xlabel('Mean Pearson Correlation')
    ax2.set_title('Algorithm Performance Ranking')
    ax2.grid(True, alpha=0.3)
    
    # Runtime comparison
    ax3 = axes[1, 0]
    runtime_data = results_df.groupby('algorithm')['fit_time'].mean().sort_values()
    
    bars = ax3.bar(range(len(runtime_data)), runtime_data.values, alpha=0.7)
    ax3.set_xlabel('Algorithm')
    ax3.set_ylabel('Mean Fit Time (seconds)')
    ax3.set_title('Algorithm Runtime Comparison')
    ax3.set_xticks(range(len(runtime_data)))
    ax3.set_xticklabels(runtime_data.index, rotation=45, ha='right')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Performance vs Runtime
    ax4 = axes[1, 1]
    alg_summary = results_df.groupby('algorithm').agg({
        'test_pearson_r': 'mean',
        'fit_time': 'mean'
    }).reset_index()
    
    scatter = ax4.scatter(alg_summary['fit_time'], alg_summary['test_pearson_r'], 
                         s=100, alpha=0.7, c=range(len(alg_summary)), cmap='viridis')
    
    for i, row in alg_summary.iterrows():
        ax4.annotate(row['algorithm'][:8], (row['fit_time'], row['test_pearson_r']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax4.set_xlabel('Mean Fit Time (seconds)')
    ax4.set_ylabel('Mean Pearson Correlation')
    ax4.set_title('Performance vs Runtime Trade-off')
    ax4.set_xscale('log')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = output_path / f"benchmarking_summary_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary report
    print("Creating summary report...")
    
    report = {
        'summary': {
            'timestamp': timestamp,
            'total_combinations': len(results_df),
            'traits_tested': results_df['trait'].nunique(),
            'algorithms_tested': results_df['algorithm'].nunique(),
            'mean_performance': results_df['test_pearson_r'].mean(),
            'best_performance': results_df['test_pearson_r'].max(),
            'total_runtime': results_df['fit_time'].sum()
        },
        'best_performers': {
            'overall_best': {
                'combination': results_df.loc[results_df['test_pearson_r'].idxmax(), 
                                             ['algorithm', 'trait', 'test_pearson_r']].to_dict(),
            },
            'best_by_trait': results_df.groupby('trait').apply(
                lambda x: x.loc[x['test_pearson_r'].idxmax(), 
                               ['algorithm', 'test_pearson_r']].to_dict()
            ).to_dict(),
            'best_algorithms': results_df.groupby('algorithm')['test_pearson_r'].agg(['mean', 'max']).sort_values('mean', ascending=False).head(10).to_dict()
        },
        'efficiency_analysis': {
            'fastest_algorithms': results_df.groupby('algorithm')['fit_time'].mean().sort_values().head(5).to_dict(),
            'best_efficiency': alg_summary.assign(
                efficiency=alg_summary['test_pearson_r'] / alg_summary['fit_time']
            ).sort_values('efficiency', ascending=False).head(5)[['algorithm', 'efficiency']].to_dict()
        }
    }
    
    # Save report
    report_file = output_path / f"benchmarking_report_{timestamp}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Create text summary
    summary_file = output_path / f"benchmarking_summary_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write("GENOMIC PREDICTION BENCHMARKING SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Results File: {latest_results.name}\n\n")
        
        f.write("OVERVIEW:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total trait-algorithm combinations: {len(results_df)}\n")
        f.write(f"Traits tested: {results_df['trait'].nunique()}\n")
        f.write(f"Algorithms tested: {results_df['algorithm'].nunique()}\n")
        f.write(f"Mean Pearson correlation: {results_df['test_pearson_r'].mean():.4f}\n")
        f.write(f"Best Pearson correlation: {results_df['test_pearson_r'].max():.4f}\n")
        f.write(f"Total computation time: {results_df['fit_time'].sum():.2f} seconds\n\n")
        
        f.write("TOP 10 ALGORITHM-TRAIT COMBINATIONS:\n")
        f.write("-" * 40 + "\n")
        top_10 = results_df.nlargest(10, 'test_pearson_r')
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            f.write(f"{i:2d}. {row['algorithm']:<15} on {row['trait']:<15}: r = {row['test_pearson_r']:.4f}\n")
        
        f.write("\nTOP 5 ALGORITHMS (by mean performance):\n")
        f.write("-" * 40 + "\n")
        top_algs = results_df.groupby('algorithm')['test_pearson_r'].mean().sort_values(ascending=False).head(5)
        for i, (alg, score) in enumerate(top_algs.items(), 1):
            f.write(f"{i:2d}. {alg:<20}: mean r = {score:.4f}\n")
        
        f.write("\nFASTEST 5 ALGORITHMS:\n")
        f.write("-" * 25 + "\n")
        fastest = results_df.groupby('algorithm')['fit_time'].mean().sort_values().head(5)
        for i, (alg, time) in enumerate(fastest.items(), 1):
            f.write(f"{i:2d}. {alg:<20}: {time:.2f} seconds\n")
    
    print(f"Summary completed!")
    print(f"Visualization: {plot_file}")
    print(f"Report: {report_file}")
    print(f"Text summary: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description='Create benchmarking summary')
    parser.add_argument('--input_dir', required=True, help='Directory with benchmarking results')
    parser.add_argument('--output_dir', required=True, help='Output directory for summary')
    
    args = parser.parse_args()
    
    create_benchmarking_summary(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()