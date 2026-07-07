#!/usr/bin/env python3
"""
File: 01a_utils_calculate_gmatrix.py
Purpose: Calculate Genomic Relationship Matrix (G-matrix) from genotype data
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

def calculate_gmatrix(X, method='vanRaden', standardize=True, save_intermediate=False, output_dir=None):
    """
    Calculate Genomic Relationship Matrix (G-matrix) from genotype data.
    
    Parameters:
    -----------
    X : numpy.ndarray or pandas.DataFrame
        Genotype matrix (n_animals x n_markers)
        Values should be 0, 1, 2 (homozygous ref, heterozygous, homozygous alt)
        or allele dosages
    
    method : str, default='vanRaden'
        Method to use for G-matrix calculation:
        - 'vanRaden': VanRaden (2008) method: G = (X - P)(X - P)' / sum(2p(1-p))
        - 'centered': Centered method: G = (X - P)(X - P)' / n_markers
        - 'scaled': Scaled method: G = (X - P)(X - P)' / (n_markers * mean_var)
    
    standardize : bool, default=True
        Whether to standardize the genotype matrix before calculation
    
    save_intermediate : bool, default=False
        Whether to save intermediate calculations (allele frequencies, etc.)
    
    output_dir : str or Path, optional
        Directory to save intermediate files if save_intermediate=True
    
    Returns:
    --------
    G : numpy.ndarray
        Genomic Relationship Matrix (n_animals x n_animals)
    
    info : dict
        Dictionary containing calculation metadata:
        - method: method used
        - n_animals: number of animals
        - n_markers: number of markers
        - allele_frequencies: array of allele frequencies
        - scaling_factor: scaling factor used
        - computation_time: time taken for calculation
    """
    start_time = datetime.now()
    
    # Convert to numpy array if needed
    if isinstance(X, pd.DataFrame):
        animal_ids = X.index.tolist()
        X = X.values
    else:
        animal_ids = None
        X = np.asarray(X)
    
    n_animals, n_markers = X.shape
    
    logging.info(f"Calculating G-matrix: {n_animals} animals x {n_markers} markers")
    logging.info(f"Method: {method}")
    
    # Handle missing values (replace with mean allele dosage)
    if np.any(np.isnan(X)):
        logging.warning("Missing values detected - replacing with column means")
        for col in range(n_markers):
            col_data = X[:, col]
            if np.any(np.isnan(col_data)):
                mean_val = np.nanmean(col_data)
                X[np.isnan(col_data), col] = mean_val if not np.isnan(mean_val) else 1.0
    
    # Calculate allele frequencies (p) - assuming 0,1,2 coding
    # p = mean(X) / 2
    p = np.mean(X, axis=0) / 2.0
    
    # Ensure allele frequencies are in valid range [0, 1]
    p = np.clip(p, 0.0, 1.0)
    
    # Calculate P matrix (expected genotype values: 2p)
    P = 2.0 * p  # Shape: (n_markers,)
    P_matrix = np.tile(P, (n_animals, 1))  # Shape: (n_animals, n_markers)
    
    # Center the genotype matrix: X - 2p
    X_centered = X - P_matrix
    
    # Calculate G-matrix based on method
    if method == 'vanRaden':
        # VanRaden (2008) method
        # G = (X - P)(X - P)' / sum(2p(1-p))
        scaling_factor = np.sum(2.0 * p * (1.0 - p))
        
        if scaling_factor == 0:
            logging.warning("Scaling factor is zero, using alternative scaling")
            scaling_factor = n_markers
        
        G = np.dot(X_centered, X_centered.T) / scaling_factor
        
    elif method == 'centered':
        # Centered method: divide by number of markers
        G = np.dot(X_centered, X_centered.T) / n_markers
        
    elif method == 'scaled':
        # Scaled method: divide by n_markers * mean variance
        mean_var = np.mean(2.0 * p * (1.0 - p))
        scaling_factor = n_markers * mean_var if mean_var > 0 else n_markers
        G = np.dot(X_centered, X_centered.T) / scaling_factor
        
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'vanRaden', 'centered', 'scaled'")
    
    # Standardize if requested (make diagonal elements approximately 1)
    if standardize:
        # Adjust diagonal to be approximately 1 (common practice)
        diag_values = np.diag(G)
        mean_diag = np.mean(diag_values)
        if mean_diag > 0:
            G = G / mean_diag
            logging.info(f"Standardized G-matrix (mean diagonal = {mean_diag:.4f})")
    
    computation_time = (datetime.now() - start_time).total_seconds()
    
    # Prepare info dictionary
    info = {
        'method': method,
        'n_animals': n_animals,
        'n_markers': n_markers,
        'allele_frequencies': p,
        'scaling_factor': scaling_factor if method == 'vanRaden' else None,
        'computation_time': computation_time,
        'standardized': standardize,
        'animal_ids': animal_ids
    }
    
    # Save intermediate files if requested
    if save_intermediate and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save allele frequencies
        freq_file = output_dir / 'allele_frequencies.csv'
        freq_df = pd.DataFrame({'marker': range(n_markers), 'allele_frequency': p})
        freq_df.to_csv(freq_file, index=False)
        logging.info(f"Saved allele frequencies to {freq_file}")
        
        # Save G-matrix
        gmatrix_file = output_dir / 'Gmatrix.csv'
        gmatrix_df = pd.DataFrame(G, index=animal_ids if animal_ids else range(n_animals),
                                 columns=animal_ids if animal_ids else range(n_animals))
        gmatrix_df.to_csv(gmatrix_file)
        logging.info(f"Saved G-matrix to {gmatrix_file}")
        
        # Save metadata
        metadata_file = output_dir / 'gmatrix_metadata.json'
        import json
        metadata = info.copy()
        metadata['allele_frequencies'] = p.tolist()  # Convert to list for JSON
        metadata['computation_time'] = computation_time
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logging.info(f"Saved metadata to {metadata_file}")
    
    logging.info(f"G-matrix calculation completed in {computation_time:.2f} seconds")
    logging.info(f"G-matrix shape: {G.shape}")
    logging.info(f"G-matrix diagonal range: [{np.min(np.diag(G)):.4f}, {np.max(np.diag(G)):.4f}]")
    logging.info(f"G-matrix off-diagonal range: [{np.min(G[~np.eye(G.shape[0], dtype=bool)]):.4f}, "
                 f"{np.max(G[~np.eye(G.shape[0], dtype=bool)]):.4f}]")
    
    return G, info


def calculate_gmatrix_from_file(X_file, output_file=None, method='vanRaden', 
                                standardize=True, save_intermediate=False, output_dir=None):
    """
    Calculate G-matrix from a genotype file.
    
    Parameters:
    -----------
    X_file : str or Path
        Path to genotype CSV file (animals as rows, markers as columns)
    
    output_file : str or Path, optional
        Path to save the G-matrix CSV file
    
    method : str, default='vanRaden'
        Method for G-matrix calculation
    
    standardize : bool, default=True
        Whether to standardize the G-matrix
    
    save_intermediate : bool, default=False
        Whether to save intermediate calculations
    
    output_dir : str or Path, optional
        Directory for intermediate files
    
    Returns:
    --------
    G : numpy.ndarray
        Genomic Relationship Matrix
    """
    logging.info(f"Loading genotype data from {X_file}")
    X_df = pd.read_csv(X_file, index_col=0)
    
    logging.info(f"Loaded {X_df.shape[0]} animals x {X_df.shape[1]} markers")
    
    # Calculate G-matrix
    G, info = calculate_gmatrix(
        X_df, 
        method=method, 
        standardize=standardize,
        save_intermediate=save_intermediate,
        output_dir=output_dir
    )
    
    # Save G-matrix if output file specified
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        G_df = pd.DataFrame(G, index=X_df.index, columns=X_df.index)
        G_df.to_csv(output_file)
        logging.info(f"Saved G-matrix to {output_file}")
    
    return G, info


def main():
    parser = argparse.ArgumentParser(
        description='Calculate Genomic Relationship Matrix (G-matrix) from genotype data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate G-matrix from genotype file
  python 01a_utils_calculate_gmatrix.py --X_file data/Geno.csv --output_file results/Gmatrix.csv
  
  # Use different method and save intermediate files
  python 01a_utils_calculate_gmatrix.py --X_file data/Geno.csv --output_file results/Gmatrix.csv \\
      --method centered --save_intermediate --output_dir results/gmatrix_intermediate
        """
    )
    
    parser.add_argument('--X_file', required=True, help='Path to genotype CSV file')
    parser.add_argument('--output_file', help='Path to save G-matrix CSV file')
    parser.add_argument('--method', default='vanRaden', 
                       choices=['vanRaden', 'centered', 'scaled'],
                       help='Method for G-matrix calculation (default: vanRaden)')
    parser.add_argument('--no_standardize', action='store_true',
                       help='Do not standardize the G-matrix')
    parser.add_argument('--save_intermediate', action='store_true',
                       help='Save intermediate calculations (allele frequencies, etc.)')
    parser.add_argument('--output_dir', help='Directory for intermediate files')
    parser.add_argument('--log_level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    try:
        # Calculate G-matrix
        G, info = calculate_gmatrix_from_file(
            X_file=args.X_file,
            output_file=args.output_file,
            method=args.method,
            standardize=not args.no_standardize,
            save_intermediate=args.save_intermediate,
            output_dir=args.output_dir
        )
        
        # Print summary
        print("\n" + "="*60)
        print("G-MATRIX CALCULATION SUMMARY")
        print("="*60)
        print(f"Method: {info['method']}")
        print(f"Animals: {info['n_animals']}")
        print(f"Markers: {info['n_markers']}")
        print(f"Computation time: {info['computation_time']:.2f} seconds")
        print(f"Standardized: {info['standardized']}")
        if info['scaling_factor']:
            print(f"Scaling factor: {info['scaling_factor']:.6f}")
        print(f"G-matrix shape: {G.shape}")
        print(f"Diagonal range: [{np.min(np.diag(G)):.4f}, {np.max(np.diag(G)):.4f}]")
        print("="*60)
        
        return 0
        
    except Exception as e:
        logging.error(f"Error calculating G-matrix: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

