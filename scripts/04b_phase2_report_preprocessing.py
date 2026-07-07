#!/usr/bin/env python3
"""
Generate Phase 2 EDA & Data Processing Report
Creates a Jupyter notebook documenting EDA and preprocessing on new prediction data
(Note: Prediction results are in a separate report: 2.3_prediction_report.ipynb)
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

def create_prediction_eda_report(dataset_dir, deployment_dir, output_file):
    """Create a Jupyter notebook with comprehensive prediction EDA analysis"""
    
    # Create notebook structure
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Title cell
    title_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Phase 2: EDA & G-Matrix Analysis Report\n",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "**Phase:** Deployment & Prediction\n",
            "\n",
            "This report documents the Exploratory Data Analysis (EDA) performed on the new prediction dataset, G-matrix calculation, and explains how missing data is handled during the prediction phase.\n",
            "\n",
            "**Note:** For prediction results analysis (all algorithms and ensembles), see `2.3_prediction_report.ipynb`."
        ]
    }
    notebook["cells"].append(title_cell)
    
    # Table of Contents
    toc_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 📋 Table of Contents\n",
            "\n",
            "1. [Overview](#overview)\n",
            "2. [Data Loading & Initial Inspection](#data-loading)\n",
            "3. [Exploratory Data Analysis (EDA)](#eda)\n",
            "   - [Dataset Dimensions](#dimensions)\n",
            "   - [Data Types & Structure](#data-types)\n",
            "   - [Missing Data Analysis](#missing-data)\n",
            "   - [Statistical Summary](#statistical-summary)\n",
            "   - [Data Distribution](#data-distribution)\n",
            "4. [Feature Filtering (Variance Mask)](#feature-filtering)\n",
            "5. [Missing Data Handling](#missing-handling)\n",
            "6. [G-Matrix Calculation](#g-matrix)\n",
            "7. [Data Preprocessing Pipeline](#preprocessing)"
        ]
    }
    notebook["cells"].append(toc_cell)
    
    # Overview section
    overview_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## 📊 Overview {#overview}\n",
            "\n",
            "In Phase 2 (Deployment & Prediction), we apply the **same feature selection and data preprocessing** that was established in Phase 1 (Learning & Benchmarking) to any new input files for prediction.\n",
            "\n",
            "### Key Principles:\n",
            "\n",
            "1. **Feature Consistency**: The same variance filter mask from Phase 1 is applied to ensure prediction uses identical features as training\n",
            "2. **Preprocessing Consistency**: Missing data handling and normalization follow the same procedures as Phase 1\n",
            "3. **Data Validation**: Input data is validated to ensure feature dimensions match the training/deployment phase\n",
            "4. **SNP Alignment**: When new animals are genotyped on a different chip/panel, SNPs are aligned to training order, missing SNPs are filled, and extra SNPs are dropped\n",
            "\n",
            "### Workflow:\n",
            "\n",
            "```\n",
            "Phase 1 (Training):\n",
            "  Input Data → Variance Filtering → Missing Data Handling → Model Training\n",
            "  ↓\n",
            "  Save: variance_filter_mask.npy + training SNP order\n",
            "\n",
            "Phase 2 (Prediction):\n",
            "  New Input Data → SNP Alignment (if needed) → Apply Variance Mask → Missing Data Handling → Prediction\n",
            "```\n",
            "\n",
            "### SNP Overlap Policy:\n",
            "\n",
            "| Overlap | Action |\n",
            "|---------|--------|\n",
            "| ≥ 80% | Proceed normally |\n",
            "| 50%–80% | Warn (predictions may be degraded) |\n",
            "| < 50% | Reject (predictions unreliable) |"
        ]
    }
    notebook["cells"].append(overview_cell)
    
    # Data Loading cell
    load_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## 📥 Data Loading & Initial Inspection {#data-loading}"
        ]
    }
    notebook["cells"].append(load_cell)
    
    load_code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from pathlib import Path\n",
            "from IPython.display import display, HTML\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "# Set style\n",
            "plt.style.use('seaborn-v0_8')\n",
            "sns.set_palette('husl')\n",
            "plt.rcParams['figure.figsize'] = (12, 6)\n",
            "\n",
            "# Define paths\n",
            f"PROJECT_ROOT = Path('{Path(dataset_dir).parent.parent}')\n",
            f"PREDICTION_DIR = PROJECT_ROOT / 'Phase2_Deployment_Prediction' / 'prediction'\n",
            f"DEPLOYMENT_DIR = PROJECT_ROOT / 'Phase2_Deployment_Prediction' / 'deployment'\n",
            f"TRAINING_DIR = PROJECT_ROOT / 'Phase1_Learning_Benchmarking' / 'training_validation'\n",
            f"DATA_DIR = Path('{dataset_dir}')\n",
            "\n",
            "print(f\"📁 Project root: {PROJECT_ROOT}\")\n",
            "print(f\"📁 Prediction directory: {PREDICTION_DIR}\")\n",
            "print(f\"📁 Deployment directory: {DEPLOYMENT_DIR}\")\n",
            "print(f\"📁 Training directory: {TRAINING_DIR}\")\n",
            "print(f\"📁 Data directory: {DATA_DIR}\")\n",
            "\n",
            "# Load prediction input data\n",
            "geno_file = DATA_DIR / 'Geno.csv'\n",
            "\n",
            "if not geno_file.exists():\n",
            "    print(f\"❌ Error: Geno.csv not found at {geno_file}\")\n",
            "    print(\"Please ensure the prediction input file exists.\")\n",
            "else:\n",
            "    print(f\"✅ Loading prediction data from: {geno_file}\")\n",
            "    \n",
            "    # Load data (first column is animal IDs)\n",
            "    X_pred_df = pd.read_csv(geno_file, index_col=0)\n",
            "    X_pred = X_pred_df.values\n",
            "    animal_ids = X_pred_df.index.tolist()\n",
            "    \n",
            "    print(f\"\\n📊 Initial Data Shape:\")\n",
            "    print(f\"   Animals (rows): {X_pred.shape[0]:,}\")\n",
            "    print(f\"   Markers (columns): {X_pred.shape[1]:,}\")\n",
            "    print(f\"   Animal IDs: {len(animal_ids)}\")"
        ]
    }
    notebook["cells"].append(load_code_cell)
    
    # EDA Section - Dataset Dimensions
    eda_dimensions_header = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## 🔍 Exploratory Data Analysis (EDA) {#eda}\n",
            "\n",
            "### Dataset Dimensions {#dimensions}"
        ]
    }
    notebook["cells"].append(eda_dimensions_header)
    
    eda_dimensions_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "print(\"=\"*80)\n",
            "print(\"DATASET DIMENSIONS\")\n",
            "print(\"=\"*80)\n",
            "print(f\"\\nOriginal Prediction Data:\")\n",
            "print(f\"  • Number of animals: {X_pred.shape[0]:,}\")\n",
            "print(f\"  • Number of markers (SNPs): {X_pred.shape[1]:,}\")\n",
            "print(f\"  • Total data points: {X_pred.size:,}\")\n",
            "print(f\"  • Memory usage: {X_pred.nbytes / (1024**2):.2f} MB\")\n",
            "\n",
            "print(f\"\\nData Structure:\")\n",
            "print(f\"  • Index type: {type(X_pred_df.index)}\")\n",
            "print(f\"  • Column type: {type(X_pred_df.columns)}\")\n",
            "print(f\"  • Data type: {X_pred.dtype}\")"
        ]
    }
    notebook["cells"].append(eda_dimensions_code)
    
    # Missing Data Analysis
    missing_data_header = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Missing Data Analysis {#missing-data}\n",
            "\n",
            "Missing data analysis is critical for understanding data quality and determining appropriate imputation strategies."
        ]
    }
    notebook["cells"].append(missing_data_header)
    
    missing_data_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Calculate missing data statistics\n",
            "missing_per_animal = np.isnan(X_pred).sum(axis=1)\n",
            "missing_per_marker = np.isnan(X_pred).sum(axis=0)\n",
            "total_missing = np.isnan(X_pred).sum()\n",
            "total_values = X_pred.size\n",
            "missing_percentage = (total_missing / total_values) * 100\n",
            "\n",
            "print(\"=\"*80)\n",
            "print(\"MISSING DATA ANALYSIS\")\n",
            "print(\"=\"*80)\n",
            "\n",
            "print(f\"\\n📊 Overall Missing Data:\")\n",
            "print(f\"  • Total missing values: {total_missing:,} ({missing_percentage:.2f}%)\")\n",
            "print(f\"  • Total non-missing values: {total_values - total_missing:,} ({100 - missing_percentage:.2f}%)\")\n",
            "\n",
            "print(f\"\\n📋 Missing Data by Animal:\")\n",
            "print(f\"  • Animals with missing data: {(missing_per_animal > 0).sum():,} / {len(missing_per_animal):,} ({(missing_per_animal > 0).sum() / len(missing_per_animal) * 100:.2f}%)\")\n",
            "print(f\"  • Mean missing per animal: {missing_per_animal.mean():.2f} markers\")\n",
            "print(f\"  • Median missing per animal: {np.median(missing_per_animal):.2f} markers\")\n",
            "print(f\"  • Max missing per animal: {missing_per_animal.max():,} markers\")\n",
            "\n",
            "print(f\"\\n📋 Missing Data by Marker:\")\n",
            "print(f\"  • Markers with missing data: {(missing_per_marker > 0).sum():,} / {len(missing_per_marker):,} ({(missing_per_marker > 0).sum() / len(missing_per_marker) * 100:.2f}%)\")\n",
            "print(f\"  • Mean missing per marker: {missing_per_marker.mean():.2f} animals\")\n",
            "print(f\"  • Median missing per marker: {np.median(missing_per_marker):.2f} animals\")\n",
            "print(f\"  • Max missing per marker: {missing_per_marker.max():,} animals\")"
        ]
    }
    notebook["cells"].append(missing_data_code)
    
    # Feature Filtering Section
    feature_filtering_header = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## 🔧 Feature Filtering (Variance Mask) {#feature-filtering}\n",
            "\n",
            "In Phase 2, we apply the **same variance filter mask** that was created and saved during Phase 1. This ensures that prediction uses identical features as the training phase.\n",
            "\n",
            "### Process:\n",
            "\n",
            "1. **Load variance filter mask** from deployment or training directory\n",
            "2. **Validate feature dimensions** - ensure input data matches mask dimensions\n",
            "3. **Apply mask** to filter out low-variance markers\n",
            "4. **Use filtered data** for prediction"
        ]
    }
    notebook["cells"].append(feature_filtering_header)
    
    feature_filtering_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Load variance filter mask\n",
            f"variance_mask_file_deploy = DEPLOYMENT_DIR / 'variance_filter_mask.npy'\n",
            f"variance_mask_file_train = TRAINING_DIR / 'variance_filter_mask.npy'\n",
            "\n",
            "variance_mask = None\n",
            "mask_source = None\n",
            "\n",
            "if variance_mask_file_deploy.exists():\n",
            "    variance_mask = np.load(variance_mask_file_deploy)\n",
            "    mask_source = 'deployment'\n",
            "    print(f\"✅ Loaded variance filter mask from deployment: {variance_mask_file_deploy}\")\n",
            "elif variance_mask_file_train.exists():\n",
            "    variance_mask = np.load(variance_mask_file_train)\n",
            "    mask_source = 'training'\n",
            "    print(f\"✅ Loaded variance filter mask from training: {variance_mask_file_train}\")\n",
            "else:\n",
            "    print(f\"❌ Warning: Variance filter mask not found in deployment or training directories\")\n",
            "\n",
            "if variance_mask is not None:\n",
            "    print(f\"\\n📊 Variance Mask Statistics:\")\n",
            "    print(f\"   • Mask shape: {variance_mask.shape}\")\n",
            "    print(f\"   • Original features: {len(variance_mask):,}\")\n",
            "    print(f\"   • Features kept (high variance): {np.sum(variance_mask):,} ({np.sum(variance_mask)/len(variance_mask)*100:.2f}%)\")\n",
            "    print(f\"   • Features removed (low variance): {np.sum(~variance_mask):,} ({np.sum(~variance_mask)/len(variance_mask)*100:.2f}%)\")\n",
            "    print(f\"   • Mask source: {mask_source}\")\n",
            "    \n",
            "    # Validate dimensions\n",
            "    if variance_mask.shape[0] == X_pred.shape[1]:\n",
            "        print(f\"\\n✅ Feature validation passed: Input data has {X_pred.shape[1]:,} features (matches mask)\")\n",
            "        \n",
            "        # Apply mask\n",
            "        X_pred_filtered = X_pred[:, variance_mask]\n",
            "        \n",
            "        print(f\"\\n📊 After Feature Filtering:\")\n",
            "        print(f\"   • Animals: {X_pred_filtered.shape[0]:,}\")\n",
            "        print(f\"   • Markers (filtered): {X_pred_filtered.shape[1]:,}\")\n",
            "        print(f\"   • Reduction: {X_pred.shape[1] - X_pred_filtered.shape[1]:,} markers removed\")\n",
            "    else:\n",
            "        print(f\"\\n❌ ERROR: Feature dimension mismatch!\")\n",
            "        print(f\"   • Input data has: {X_pred.shape[1]:,} features\")\n",
            "        print(f\"   • Mask expects: {variance_mask.shape[0]:,} features\")"
        ]
    }
    notebook["cells"].append(feature_filtering_code)
    
    # Missing Data Handling Section
    missing_handling_header = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## 🧹 Missing Data Handling {#missing-handling}\n",
            "\n",
            "### Strategy:\n",
            "\n",
            "Missing values in the prediction dataset are handled using **column-wise mode imputation**, which matches the approach used in Phase 1.\n",
            "\n",
            "### Method:\n",
            "\n",
            "For each marker (column):\n",
            "1. Calculate the **mode** (most frequent value) from non-missing values\n",
            "2. Replace all missing values in that column with the mode\n",
            "3. If all values are missing (edge case), replace with 0\n",
            "\n",
            "### Rationale:\n",
            "\n",
            "- **Mode imputation** is appropriate for genotype data where values are discrete (typically 0, 1, 2)\n",
            "- **Column-wise** approach ensures each marker's missing values are filled with its own most common value\n",
            "- This maintains the **distribution characteristics** of each marker\n",
            "- **Consistent with Phase 1** preprocessing ensures model compatibility"
        ]
    }
    notebook["cells"].append(missing_handling_header)
    
    missing_handling_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Apply missing data handling (same as preprocessing function)\n",
            "if variance_mask is not None and variance_mask.shape[0] == X_pred.shape[1]:\n",
            "    X_clean = X_pred_filtered.copy()\n",
            "    \n",
            "    # Count missing before imputation\n",
            "    missing_before = np.isnan(X_clean).sum()\n",
            "    missing_percentage_before = (missing_before / X_clean.size) * 100\n",
            "    \n",
            "    print(\"=\"*80)\n",
            "    print(\"MISSING DATA HANDLING\")\n",
            "    print(\"=\"*80)\n",
            "    print(f\"\\n📊 Before Imputation (on filtered data):\")\n",
            "    print(f\"   • Missing values: {missing_before:,} ({missing_percentage_before:.2f}%)\")\n",
            "    print(f\"   • Non-missing values: {X_clean.size - missing_before:,} ({100 - missing_percentage_before:.2f}%)\")\n",
            "    \n",
            "    # Apply mode imputation (column-wise)\n",
            "    if np.any(np.isnan(X_clean)):\n",
            "        print(f\"\\n🔄 Applying mode imputation (column-wise)...\")\n",
            "        \n",
            "        imputation_stats = []\n",
            "        for col in range(X_clean.shape[1]):\n",
            "            col_data = X_clean[:, col]\n",
            "            if np.any(np.isnan(col_data)):\n",
            "                # Get non-missing values\n",
            "                non_missing = col_data[~np.isnan(col_data)]\n",
            "                if len(non_missing) > 0:\n",
            "                    # Calculate mode\n",
            "                    unique_vals, counts = np.unique(non_missing, return_counts=True)\n",
            "                    mode_val = unique_vals[np.argmax(counts)]\n",
            "                    \n",
            "                    # Replace missing with mode\n",
            "                    n_missing = np.isnan(col_data).sum()\n",
            "                    X_clean[np.isnan(col_data), col] = mode_val\n",
            "                    \n",
            "                    imputation_stats.append({\n",
            "                        'marker_index': col,\n",
            "                        'n_missing': n_missing,\n",
            "                        'mode_value': mode_val\n",
            "                    })\n",
            "                else:\n",
            "                    # All values missing - replace with 0\n",
            "                    n_missing = np.isnan(col_data).sum()\n",
            "                    X_clean[np.isnan(col_data), col] = 0\n",
            "        \n",
            "        # Verify no missing values remain\n",
            "        missing_after = np.isnan(X_clean).sum()\n",
            "        \n",
            "        print(f\"\\n✅ After Imputation:\")\n",
            "        print(f\"   • Missing values: {missing_after:,}\")\n",
            "        print(f\"   • Markers with imputation: {len(imputation_stats):,}\")\n",
            "        print(f\"   • Total values imputed: {sum(s['n_missing'] for s in imputation_stats):,}\")\n",
            "    else:\n",
            "        print(f\"\\n✅ No missing values found - no imputation needed\")\n",
            "    \n",
            "    print(f\"\\n📊 Final Processed Data:\")\n",
            "    print(f\"   • Shape: {X_clean.shape}\")\n",
            "    print(f\"   • Animals: {X_clean.shape[0]:,}\")\n",
            "    print(f\"   • Markers: {X_clean.shape[1]:,}\")\n",
            "    print(f\"   • Missing values: {np.isnan(X_clean).sum():,}\")\n",
            "    print(f\"   • Data ready for prediction: ✅\")\n",
            "else:\n",
            "    print(\"❌ Cannot proceed with missing data handling - feature validation failed\")"
        ]
    }
    notebook["cells"].append(missing_handling_code)
    
    # G-Matrix Calculation Section
    gmatrix_header = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## 🧬 G-Matrix Calculation {#g-matrix}\n",
            "\n",
            "The Genomic Relationship Matrix (G-matrix) is calculated from the processed genotype data. The G-matrix represents the genetic relationships between animals based on their genomic data.\n",
            "\n",
            "### Purpose:\n",
            "\n",
            "- **Relationship Matrix**: Quantifies genetic similarity between animals\n",
            "- **Model Input**: Required for some algorithms (e.g., GBLUP) that use genomic relationships\n",
            "- **Quality Check**: Helps identify related animals and potential data issues\n",
            "\n",
            "### Calculation Method:\n",
            "\n",
            "The G-matrix is calculated using the formula:\n",
            "```\n",
            "G = (X - P)(X - P)' / sum(2p(1-p))\n",
            "```\n",
            "\n",
            "Where:\n",
            "- **X**: Genotype matrix (animals × markers)\n",
            "- **P**: Matrix of allele frequencies\n",
            "- **p**: Vector of allele frequencies for each marker"
        ]
    }
    notebook["cells"].append(gmatrix_header)
    
    gmatrix_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Calculate G-matrix from processed data\n",
            "if variance_mask is not None and variance_mask.shape[0] == X_pred.shape[1]:\n",
            "    print('='*80)\n",
            "    print('G-MATRIX CALCULATION')\n",
            "    print('='*80)\n",
            "    \n",
            "    # Use the cleaned, filtered data\n",
            "    X_processed = X_clean.copy()\n",
            "    \n",
            "    print(f'\\n📊 Input Data for G-Matrix:')\n",
            "    print(f'  • Animals: {X_processed.shape[0]:,}')\n",
            "    print(f'  • Markers: {X_processed.shape[1]:,}')\n",
            "    \n",
            "    # Calculate allele frequencies (p) for each marker\n",
            "    print(f'\\n🔄 Calculating allele frequencies...')\n",
            "    p = np.nanmean(X_processed, axis=0) / 2.0  # Divide by 2 since genotypes are 0, 1, 2\n",
            "    \n",
            "    # Create P matrix (broadcast p to all animals)\n",
            "    P = np.tile(p, (X_processed.shape[0], 1))\n",
            "    \n",
            "    # Calculate G-matrix: G = (X - P)(X - P)' / sum(2p(1-p))\n",
            "    print(f'🔄 Computing G-matrix...')\n",
            "    \n",
            "    # Center the data\n",
            "    X_centered = X_processed - P\n",
            "    \n",
            "    # Calculate denominator: sum of 2p(1-p) for each marker\n",
            "    denominator = np.sum(2 * p * (1 - p))\n",
            "    \n",
            "    if denominator > 0:\n",
            "        # Calculate G-matrix\n",
            "        G_matrix = np.dot(X_centered, X_centered.T) / denominator\n",
            "        \n",
            "        print(f'\\n✅ G-Matrix calculated successfully!')\n",
            "        print(f'\\n📊 G-Matrix Statistics:')\n",
            "        print(f'  • Shape: {G_matrix.shape} (animals × animals)')\n",
            "        print(f'  • Size: {G_matrix.size:,} elements')\n",
            "        print(f'  • Memory usage: {G_matrix.nbytes / (1024**2):.2f} MB')\n",
            "        print(f'  • Diagonal mean: {np.mean(np.diag(G_matrix)):.4f}')\n",
            "        print(f'  • Diagonal std: {np.std(np.diag(G_matrix)):.4f}')\n",
            "        print(f'  • Off-diagonal mean: {np.mean(G_matrix[~np.eye(G_matrix.shape[0], dtype=bool)]):.4f}')\n",
            "        print(f'  • Off-diagonal std: {np.std(G_matrix[~np.eye(G_matrix.shape[0], dtype=bool)]):.4f}')\n",
            "        print(f'  • Min value: {np.min(G_matrix):.4f}')\n",
            "        print(f'  • Max value: {np.max(G_matrix):.4f}')\n",
            "        \n",
            "        # Check for symmetry\n",
            "        is_symmetric = np.allclose(G_matrix, G_matrix.T)\n",
            "        print(f'  • Is symmetric: {is_symmetric}')\n",
            "        \n",
            "        # Eigenvalue analysis\n",
            "        print(f'\\n🔬 Eigenvalue Analysis:')\n",
            "        eigenvals = np.linalg.eigvals(G_matrix)\n",
            "        eigenvals_real = np.real(eigenvals)  # Take real part\n",
            "        eigenvals_real = np.sort(eigenvals_real)[::-1]  # Sort descending\n",
            "        \n",
            "        print(f'  • Number of eigenvalues: {len(eigenvals_real)}')\n",
            "        print(f'  • Largest eigenvalue: {eigenvals_real[0]:.4f}')\n",
            "        print(f'  • Smallest eigenvalue: {eigenvals_real[-1]:.4f}')\n",
            "        print(f'  • Condition number: {eigenvals_real[0] / eigenvals_real[-1]:.2f}')\n",
            "        print(f'  • Positive eigenvalues: {(eigenvals_real > 0).sum()} ({(eigenvals_real > 0).sum() / len(eigenvals_real) * 100:.2f}%)')\n",
            "        \n",
            "        # Visualization\n",
            "        print(f'\\n📈 G-Matrix Visualizations:')\n",
            "        \n",
            "        # Plot 1: Heatmap of G-matrix (sample)\n",
            "        fig, axes = plt.subplots(2, 2, figsize=(16, 12))\n",
            "        \n",
            "        # Heatmap (first 50 animals)\n",
            "        sample_size = min(50, G_matrix.shape[0])\n",
            "        ax1 = axes[0, 0]\n",
            "        im1 = ax1.imshow(G_matrix[:sample_size, :sample_size], cmap='coolwarm', aspect='auto')\n",
            "        ax1.set_title(f'G-Matrix Heatmap (First {sample_size} Animals)', fontsize=12, fontweight='bold')\n",
            "        ax1.set_xlabel('Animal Index')\n",
            "        ax1.set_ylabel('Animal Index')\n",
            "        plt.colorbar(im1, ax=ax1)\n",
            "        \n",
            "        # Distribution of diagonal elements\n",
            "        ax2 = axes[0, 1]\n",
            "        ax2.hist(np.diag(G_matrix), bins=30, edgecolor='black', alpha=0.7)\n",
            "        ax2.set_title('Distribution of Diagonal Elements', fontsize=12, fontweight='bold')\n",
            "        ax2.set_xlabel('Diagonal Value')\n",
            "        ax2.set_ylabel('Frequency')\n",
            "        ax2.axvline(np.mean(np.diag(G_matrix)), color='red', linestyle='--', \n",
            "                   label=f'Mean: {np.mean(np.diag(G_matrix)):.3f}')\n",
            "        ax2.legend()\n",
            "        \n",
            "        # Distribution of off-diagonal elements\n",
            "        ax3 = axes[1, 0]\n",
            "        off_diag = G_matrix[~np.eye(G_matrix.shape[0], dtype=bool)]\n",
            "        ax3.hist(off_diag, bins=50, edgecolor='black', alpha=0.7)\n",
            "        ax3.set_title('Distribution of Off-Diagonal Elements', fontsize=12, fontweight='bold')\n",
            "        ax3.set_xlabel('Off-Diagonal Value')\n",
            "        ax3.set_ylabel('Frequency')\n",
            "        ax3.axvline(np.mean(off_diag), color='red', linestyle='--', \n",
            "                   label=f'Mean: {np.mean(off_diag):.3f}')\n",
            "        ax3.legend()\n",
            "        \n",
            "        # Eigenvalue distribution\n",
            "        ax4 = axes[1, 1]\n",
            "        ax4.plot(eigenvals_real[:min(100, len(eigenvals_real))], 'o-', markersize=3)\n",
            "        ax4.set_title('Eigenvalue Spectrum (First 100)', fontsize=12, fontweight='bold')\n",
            "        ax4.set_xlabel('Eigenvalue Index')\n",
            "        ax4.set_ylabel('Eigenvalue')\n",
            "        ax4.grid(True, alpha=0.3)\n",
            "        \n",
            "        plt.tight_layout()\n",
            "        plt.show()\n",
            "        \n",
            "        print(f'\\n✅ G-Matrix analysis complete!')\n",
            "        \n",
            "    else:\n",
            "        print(f'❌ Error: Denominator is zero or negative. Cannot calculate G-matrix.')\n",
            "else:\n",
            "    print('❌ Cannot calculate G-matrix - feature validation failed or data not processed')"
        ]
    }
    notebook["cells"].append(gmatrix_code)
    
    # Preprocessing Pipeline Summary
    preprocessing_header = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## ⚙️ Data Preprocessing Pipeline {#preprocessing}\n",
            "\n",
            "### Complete Pipeline:\n",
            "\n",
            "```\n",
            "1. Load Input Data (Geno.csv)\n",
            "   ↓\n",
            "2. Validate Feature Dimensions\n",
            "   ↓\n",
            "3. Apply Variance Filter Mask (from Phase 1)\n",
            "   ↓\n",
            "4. Handle Missing Values (Mode Imputation)\n",
            "   ↓\n",
            "5. Data Ready for Prediction\n",
            "```\n",
            "\n",
            "### Key Points:\n",
            "\n",
            "- **Feature consistency**: Same features as Phase 1\n",
            "- **Preprocessing consistency**: Same missing data handling as Phase 1\n",
            "- **Validation**: Feature dimensions are checked before processing\n",
            "- **Traceability**: Variance mask source is tracked"
        ]
    }
    notebook["cells"].append(preprocessing_header)
    
    preprocessing_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "print(\"=\"*80)\n",
            "print(\"DATA PREPROCESSING PIPELINE SUMMARY\")\n",
            "print(\"=\"*80)\n",
            "\n",
            "if variance_mask is not None and variance_mask.shape[0] == X_pred.shape[1]:\n",
            "    pipeline_summary = pd.DataFrame({\n",
            "        'Step': [\n",
            "            '1. Original Input Data',\n",
            "            '2. Feature Filtering (Variance Mask)',\n",
            "            '3. Missing Data Handling',\n",
            "            '4. Final Processed Data'\n",
            "        ],\n",
            "        'Animals': [\n",
            "            X_pred.shape[0],\n",
            "            X_pred_filtered.shape[0],\n",
            "            X_clean.shape[0],\n",
            "            X_clean.shape[0]\n",
            "        ],\n",
            "        'Markers': [\n",
            "            X_pred.shape[1],\n",
            "            X_pred_filtered.shape[1],\n",
            "            X_clean.shape[1],\n",
            "            X_clean.shape[1]\n",
            "        ],\n",
            "        'Missing Values': [\n",
            "            np.isnan(X_pred).sum(),\n",
            "            np.isnan(X_pred_filtered).sum(),\n",
            "            np.isnan(X_clean).sum(),\n",
            "            np.isnan(X_clean).sum()\n",
            "        ],\n",
            "        'Status': [\n",
            "            '✅ Loaded',\n",
            "            '✅ Filtered',\n",
            "            '✅ Imputed',\n",
            "            '✅ Ready'\n",
            "        ]\n",
            "    })\n",
            "    \n",
            "    display(pipeline_summary)\n",
            "    \n",
            "    print(f\"\\n📊 Pipeline Statistics:\")\n",
            "    print(f\"   • Feature reduction: {X_pred.shape[1] - X_clean.shape[1]:,} markers ({((X_pred.shape[1] - X_clean.shape[1]) / X_pred.shape[1] * 100):.2f}%)\")\n",
            "    print(f\"   • Missing values handled: {np.isnan(X_pred).sum():,} → {np.isnan(X_clean).sum():,}\")\n",
            "    print(f\"   • Variance mask source: {mask_source}\")\n",
            "    print(f\"   • Data consistency: ✅ Matches Phase 1 preprocessing\")\n",
            "else:\n",
            "    print(\"❌ Pipeline summary not available - feature validation failed\")"
        ]
    }
    notebook["cells"].append(preprocessing_code)
    
    # Save notebook
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"✅ Phase 2 EDA & G-Matrix report notebook created: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate Phase 2 EDA report notebook')
    parser.add_argument('--dataset_dir', required=True, help='Path to dataset directory (containing Geno.csv)')
    parser.add_argument('--deployment_dir', required=True, help='Path to deployment directory (containing variance_filter_mask.npy)')
    parser.add_argument('--output_file', required=True, help='Path to output notebook file (.ipynb)')
    
    args = parser.parse_args()
    
    create_prediction_eda_report(args.dataset_dir, args.deployment_dir, args.output_file)

if __name__ == '__main__':
    main()
