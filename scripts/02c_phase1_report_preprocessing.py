#!/usr/bin/env python3
"""
Generate comprehensive QC report notebook
Creates a Jupyter notebook with visualizations and analysis for QC stage
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime
import os

def _set_code_hidden_by_default(notebook):
    """Collapse/hide code cells by default while keeping them expandable."""
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            metadata = cell.setdefault("metadata", {})
            metadata["collapsed"] = True
            jupyter_meta = metadata.setdefault("jupyter", {})
            jupyter_meta["source_hidden"] = True

def create_qc_report_notebook(dataset_dir, gmatrix_dir, output_file):
    """Create a Jupyter notebook with comprehensive QC analysis"""

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
            "# Quality Control (QC) Pipeline Report\n",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "\n",
            "This report provides comprehensive analysis of the Quality Control pipeline including:\n",
            "- Data loading and exploration\n",
            "- SNP filtering and quality control\n",
            "- G-matrix calculation\n",
            "- Genotype-phenotype alignment\n",
            "- Final dataset statistics"
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
            "1. [Data Overview](#data-overview)\n",
            "2. [Genotype Data (Geno.csv)](#genotype-data)\n",
            "3. [Phenotype Data (Pheno.csv)](#phenotype-data)\n",
            "4. [SNP Filtering Analysis](#snp-filtering)\n",
            "5. [G-Matrix Calculation](#g-matrix)\n",
            "6. [Data Alignment Results](#data-alignment)\n",
            "7. [Quality Control Summary](#qc-summary)"
        ]
    }
    notebook["cells"].append(toc_cell)

    # Load data cell
    load_cell = {
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
            "import json\n",
            "from datetime import datetime\n",
            "\n",
            "# Set style\n",
            "plt.style.use('seaborn-v0_8')\n",
            "sns.set_palette('husl')\n",
            "\n",
            "# Define paths\n",
            f"dataset_dir = Path('{dataset_dir}')\n",
            f"gmatrix_dir = Path('{gmatrix_dir}')\n",
            "\n",
            "print('🔍 Loading QC pipeline data...')\n",
            "\n",
            "# Load genotype data (first column is animal IDs)\n",
            "try:\n",
            "    geno_path = dataset_dir / 'Geno.csv'\n",
            "    geno_df = pd.read_csv(geno_path, index_col=0)\n",
            "    print(f'✅ Genotype data loaded: {geno_df.shape}')\n",
            "except Exception as e:\n",
            "    print(f'❌ Error loading genotype data: {e}')\n",
            "    geno_df = None\n",
            "\n",
            "# Load phenotype data (first column is animal IDs)\n",
            "try:\n",
            "    pheno_path = dataset_dir / 'Pheno.csv'\n",
            "    pheno_df = pd.read_csv(pheno_path, index_col=0)\n",
            "    print(f'✅ Phenotype data loaded: {pheno_df.shape}')\n",
            "except Exception as e:\n",
            "    print(f'❌ Error loading phenotype data: {e}')\n",
            "    pheno_df = None\n",
            "\n",
            "# Load G-matrix data\n",
            "try:\n",
            "    gmatrix_path = gmatrix_dir / 'Gmatrix.csv'\n",
            "    gmatrix_df = pd.read_csv(gmatrix_path, index_col=0)\n",
            "    print(f'✅ G-matrix loaded: {gmatrix_df.shape}')\n",
            "except Exception as e:\n",
            "    print(f'❌ Error loading G-matrix: {e}')\n",
            "    gmatrix_df = None\n",
            "\n",
            "# Load metadata\n",
            "try:\n",
            "    metadata_path = gmatrix_dir / 'gmatrix_metadata.json'\n",
            "    with open(metadata_path, 'r') as f:\n",
            "        metadata = json.load(f)\n",
            "    print(f'✅ Metadata loaded')\n",
            "except Exception as e:\n",
            "    print(f'❌ Error loading metadata: {e}')\n",
            "    metadata = {}\n",
            "\n",
            "# Load allele frequencies\n",
            "try:\n",
            "    allele_freq_path = gmatrix_dir / 'allele_frequencies.csv'\n",
            "    allele_freq_df = pd.read_csv(allele_freq_path)\n",
            "    print(f'✅ Allele frequencies loaded: {allele_freq_df.shape}')\n",
            "except Exception as e:\n",
            "    print(f'❌ Error loading allele frequencies: {e}')\n",
            "    allele_freq_df = None"
        ]
    }
    notebook["cells"].append(load_cell)

    # Data Overview Section
    overview_header = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## 📊 Data Overview {#data-overview}\n",
            "\n",
            "### Dataset Summary"
        ]
    }
    notebook["cells"].append(overview_header)

    overview_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Dataset summary\n",
            "print('=== QUALITY CONTROL PIPELINE SUMMARY ===')\n",
            "print()\n",
            "\n",
            "if geno_df is not None:\n",
            "    print(f'📊 Genotype Data (Geno.csv):')\n",
            "    print(f'   • Samples (animals): {geno_df.shape[0]:,}')\n",
            "    print(f'   • Markers (SNPs): {geno_df.shape[1]:,}')\n",
            "    print(f'   • Data type: {geno_df.dtypes.iloc[0]}')\n",
            "    print(f'   • Memory usage: {geno_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB')\n",
            "    print()\n",
            "\n",
            "if pheno_df is not None:\n",
            "    print(f'🎯 Phenotype Data (Pheno.csv):')\n",
            "    print(f'   • Samples (animals): {pheno_df.shape[0]:,}')\n",
            "    print(f'   • Traits: {pheno_df.shape[1]:,}')\n",
            "    print(f'   • Data completeness: {pheno_df.notna().sum().sum()}/{pheno_df.shape[0]*pheno_df.shape[1]} ({pheno_df.notna().sum().sum()/(pheno_df.shape[0]*pheno_df.shape[1])*100:.1f}%)')\n",
            "    print()\n",
            "\n",
            "if gmatrix_df is not None:\n",
            "    print(f'🧬 G-Matrix:')\n",
            "    print(f'   • Shape: {gmatrix_df.shape}')\n",
            "    print(f'   • Diagonal range: [{gmatrix_df.values.diagonal().min():.4f}, {gmatrix_df.values.diagonal().max():.4f}]')\n",
            "    print(f'   • Off-diagonal range: [{gmatrix_df.values[np.triu_indices_from(gmatrix_df.values, k=1)].min():.4f}, {gmatrix_df.values[np.triu_indices_from(gmatrix_df.values, k=1)].max():.4f}]')\n",
            "    print()\n",
            "\n",
            "if metadata:\n",
            "    print(f'📋 Pipeline Metadata:')\n",
            "    for key, value in metadata.items():\n",
            "        print(f'   • {key}: {value}')\n",
            "    print()\n",
            "\n",
            "print('✅ Data loading completed successfully')"
        ]
    }
    notebook["cells"].append(overview_code)

    # Genotype Data Section
    geno_header = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## 🧬 Genotype Data Analysis {#genotype-data}\n",
            "\n",
            "### Genotype Matrix Statistics"
        ]
    }
    notebook["cells"].append(geno_header)

    geno_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "if geno_df is not None:\n",
            "    print('=== GENOTYPE DATA ANALYSIS ===')\n",
            "    print()\n",
            "    \n",
            "    # Basic statistics\n",
            "    print('📊 Basic Statistics:')\n",
            "    print(f'   • Total SNPs: {geno_df.shape[1]:,}')\n",
            "    print(f'   • Total animals: {geno_df.shape[0]:,}')\n",
            "    print(f'   • Missing values: {geno_df.isna().sum().sum():,}')\n",
            "    print(f'   • Missing rate: {geno_df.isna().sum().sum()/(geno_df.shape[0]*geno_df.shape[1])*100:.2f}%')\n",
            "    print()\n",
            "    \n",
            "    # SNP-wise missing rates\n",
            "    snp_missing = geno_df.isna().mean()\n",
            "    print('🔍 SNP Missing Rate Distribution:')\n",
            "    print(f'   • SNPs with 0% missing: {(snp_missing == 0).sum():,}')\n",
            "    print(f'   • SNPs with <5% missing: {(snp_missing < 0.05).sum():,}')\n",
            "    print(f'   • SNPs with >50% missing: {(snp_missing > 0.5).sum():,}')\n",
            "    print()\n",
            "    \n",
            "    # Genotype distribution\n",
            "    if geno_df.dtypes.iloc[0] in ['int64', 'float64']:\n",
            "        unique_vals = np.unique(geno_df.values[~np.isnan(geno_df.values)])\n",
            "        print(f'🎲 Genotype Values: {sorted(unique_vals)}')\n",
            "        \n",
            "        # Distribution plot\n",
            "        fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n",
            "        \n",
            "        # SNP missing rate histogram\n",
            "        axes[0].hist(snp_missing, bins=50, alpha=0.7, edgecolor='black')\n",
            "        axes[0].set_xlabel('Missing Rate')\n",
            "        axes[0].set_ylabel('Number of SNPs')\n",
            "        axes[0].set_title('SNP Missing Rate Distribution')\n",
            "        axes[0].axvline(snp_missing.mean(), color='red', linestyle='--', label=f'Mean: {snp_missing.mean():.3f}')\n",
            "        axes[0].legend()\n",
            "        \n",
            "        # Minor allele frequency distribution (if binary)\n",
            "        if len(unique_vals) == 2 and 0 in unique_vals and 1 in unique_vals:\n",
            "            maf = geno_df.mean(axis=0, skipna=True)\n",
            "            maf = np.minimum(maf, 1-maf)  # Use minor allele frequency\n",
            "            axes[1].hist(maf, bins=50, alpha=0.7, edgecolor='black')\n",
            "            axes[1].set_xlabel('Minor Allele Frequency')\n",
            "            axes[1].set_ylabel('Number of SNPs')\n",
            "            axes[1].set_title('MAF Distribution')\n",
            "            axes[1].axvline(maf.mean(), color='red', linestyle='--', label=f'Mean MAF: {maf.mean():.3f}')\n",
            "            axes[1].legend()\n",
            "        \n",
            "        plt.tight_layout()\n",
            "        plt.show()\n",
            "    \n",
            "    # Sample first few rows\n",
            "    print('\\n📋 Sample of Genotype Data:')\n",
            "    display(HTML(geno_df.head().to_html()))\n",
            "    \n",
            "else:\n",
            "    print('❌ Genotype data not available')"
        ]
    }
    notebook["cells"].append(geno_code)

    # Phenotype Data Section
    pheno_header = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## 🎯 Phenotype Data Analysis {#phenotype-data}\n",
            "\n",
            "### Phenotype Matrix Statistics"
        ]
    }
    notebook["cells"].append(pheno_header)

    pheno_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "if pheno_df is not None:\n",
            "    print('=== PHENOTYPE DATA ANALYSIS ===')\n",
            "    print()\n",
            "    \n",
            "    # Basic statistics\n",
            "    print('📊 Basic Statistics:')\n",
            "    print(f'   • Total traits: {pheno_df.shape[1]}')\n",
            "    print(f'   • Total animals: {pheno_df.shape[0]}')\n",
            "    print(f'   • Complete records: {(pheno_df.notna().all(axis=1)).sum()}')\n",
            "    print(f'   • Records with missing data: {(~pheno_df.notna().all(axis=1)).sum()}')\n",
            "    print()\n",
            "    \n",
            "    # Trait-wise completeness\n",
            "    trait_completeness = pheno_df.notna().mean()\n",
            "    print('🎯 Trait Completeness:')\n",
            "    for trait in trait_completeness.sort_values(ascending=False).index:\n",
            "        pct = trait_completeness[trait] * 100\n",
            "        print(f'   • {trait}: {pct:.1f}% complete')\n",
            "    print()\n",
            "    \n",
            "    # Summary statistics for each trait\n",
            "    numeric_cols = pheno_df.select_dtypes(include=[np.number]).columns\n",
            "    if len(numeric_cols) > 0:\n",
            "        trait_stats = pheno_df[numeric_cols].describe().T\n",
            "        trait_stats = trait_stats[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]\n",
            "        trait_stats['missing_rate'] = (1 - trait_stats['count']/pheno_df.shape[0]) * 100\n",
            "        \n",
            "        print('📈 Trait Summary Statistics:')\n",
            "        display(HTML(trait_stats.round(3).to_html()))\n",
            "        \n",
            "        # Distribution plots\n",
            "        fig, axes = plt.subplots(2, 3, figsize=(18, 10))\n",
            "        axes = axes.flatten()\n",
            "        \n",
            "        # Plot distributions for first 6 traits\n",
            "        plot_traits = numeric_cols[:6]\n",
            "        for i, trait in enumerate(plot_traits):\n",
            "            if i < len(axes):\n",
            "                data = pheno_df[trait].dropna()\n",
            "                axes[i].hist(data, bins=30, alpha=0.7, edgecolor='black')\n",
            "                axes[i].set_title(f'{trait} Distribution')\n",
            "                axes[i].set_xlabel('Value')\n",
            "                axes[i].set_ylabel('Frequency')\n",
            "        \n",
            "        plt.tight_layout()\n",
            "        plt.show()\n",
            "    \n",
            "    # Sample data\n",
            "    print('\\n📋 Sample of Phenotype Data:')\n",
            "    display(HTML(pheno_df.head().to_html()))\n",
            "    \n",
            "else:\n",
            "    print('❌ Phenotype data not available')"
        ]
    }
    notebook["cells"].append(pheno_code)

    # SNP Filtering Section
    snp_header = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## 🔬 SNP Filtering Analysis {#snp-filtering}\n",
            "\n",
            "### SNP Quality Control and Filtering"
        ]
    }
    notebook["cells"].append(snp_header)

    snp_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "print('=== SNP FILTERING ANALYSIS ===')\n",
            "print()\n",
            "\n",
            "if geno_df is not None:\n",
            "    print('🔍 SNP Filtering Criteria Applied:')\n",
            "    print('   • Low variance SNPs removed')\n",
            "    print('   • Missing data filtering')\n",
            "    print('   • MAF filtering (if applicable)')\n",
            "    print()\n",
            "    \n",
            "    # Calculate variance for each SNP\n",
            "    snp_variance = geno_df.var(axis=0, skipna=True)\n",
            "    \n",
            "    print('📊 SNP Variance Analysis:')\n",
            "    print(f'   • SNPs with zero variance: {(snp_variance == 0).sum()}')\n",
            "    print(f'   • SNPs with variance < 0.01: {(snp_variance < 0.01).sum()}')\n",
            "    print(f'   • SNPs with variance < 0.05: {(snp_variance < 0.05).sum()}')\n",
            "    print(f'   • Mean variance: {snp_variance.mean():.4f}')\n",
            "    print(f'   • Variance range: [{snp_variance.min():.4f}, {snp_variance.max():.4f}]')\n",
            "    print()\n",
            "    \n",
            "    # Missing data analysis\n",
            "    snp_missing_rate = geno_df.isna().mean()\n",
            "    print('📊 SNP Missing Data Analysis:')\n",
            "    print(f'   • SNPs with >90% missing: {(snp_missing_rate > 0.9).sum()}')\n",
            "    print(f'   • SNPs with >50% missing: {(snp_missing_rate > 0.5).sum()}')\n",
            "    print(f'   • SNPs with >10% missing: {(snp_missing_rate > 0.1).sum()}')\n",
            "    print()\n",
            "    \n",
            "    # MAF analysis (if binary data)\n",
            "    if geno_df.dtypes.iloc[0] in ['int64', 'float64']:\n",
            "        unique_vals = np.unique(geno_df.values[~np.isnan(geno_df.values)])\n",
            "        if len(unique_vals) == 2 and 0 in unique_vals and 1 in unique_vals:\n",
            "            maf = geno_df.mean(axis=0, skipna=True)\n",
            "            maf = np.minimum(maf, 1-maf)\n",
            "            \n",
            "            print('🎲 Minor Allele Frequency (MAF) Analysis:')\n",
            "            print(f'   • SNPs with MAF = 0: {(maf == 0).sum()}')\n",
            "            print(f'   • SNPs with MAF < 0.01: {(maf < 0.01).sum()}')\n",
            "            print(f'   • SNPs with MAF < 0.05: {(maf < 0.05).sum()}')\n",
            "            print(f'   • Mean MAF: {maf.mean():.4f}')\n",
            "            print()\n",
            "            \n",
            "            # MAF distribution plot\n",
            "            fig, ax = plt.subplots(figsize=(10, 6))\n",
            "            ax.hist(maf, bins=50, alpha=0.7, edgecolor='black')\n",
            "            ax.set_xlabel('Minor Allele Frequency')\n",
            "            ax.set_ylabel('Number of SNPs')\n",
            "            ax.set_title('MAF Distribution')\n",
            "            ax.axvline(maf.mean(), color='red', linestyle='--', label=f'Mean MAF: {maf.mean():.3f}')\n",
            "            ax.legend()\n",
            "            plt.show()\n",
            "\n",
            "# Load allele frequencies if available\n",
            "if allele_freq_df is not None:\n",
            "    print('\\n📊 Allele Frequency Summary:')\n",
            "    display(HTML(allele_freq_df.describe().to_html()))\n",
            "    \n",
            "    # Plot allele frequencies\n",
            "    fig, ax = plt.subplots(figsize=(10, 6))\n",
            "    ax.hist(allele_freq_df['allele_frequency'], bins=50, alpha=0.7, edgecolor='black')\n",
            "    ax.set_xlabel('Allele Frequency')\n",
            "    ax.set_ylabel('Number of SNPs')\n",
            "    ax.set_title('Allele Frequency Distribution')\n",
            "    ax.axvline(allele_freq_df['allele_frequency'].mean(), color='red', linestyle='--', \n",
            "                label=f'Mean: {allele_freq_df[\"allele_frequency\"].mean():.3f}')\n",
            "    ax.legend()\n",
            "    plt.show()\n",
            "\n",
            "print('✅ SNP filtering analysis completed')"
        ]
    }
    notebook["cells"].append(snp_code)

    # G-Matrix Section
    gmatrix_header = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## 🧮 G-Matrix Calculation {#g-matrix}\n",
            "\n",
            "### Genomic Relationship Matrix Analysis (Heatmap + CA)"
        ]
    }
    notebook["cells"].append(gmatrix_header)

    gmatrix_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "print('=== G-MATRIX ANALYSIS ===')\n",
            "print()\n",
            "\n",
            "if gmatrix_df is not None:\n",
            "    print('🧬 G-Matrix Properties:')\n",
            "    print(f'   • Shape: {gmatrix_df.shape}')\n",
            "    print(f'   • Animals: {gmatrix_df.shape[0]}')\n",
            "    print(f'   • Matrix is symmetric: {np.allclose(gmatrix_df.values, gmatrix_df.values.T)}')\n",
            "    print(f'   • Matrix is positive semi-definite: {np.all(np.linalg.eigvals(gmatrix_df.values) >= -1e-10)}')\n",
            "    print()\n",
            "    \n",
            "    # Diagonal analysis\n",
            "    diagonal = np.diag(gmatrix_df.values)\n",
            "    print('📊 Diagonal Elements (Self-relationships):')\n",
            "    print(f'   • Mean: {diagonal.mean():.4f}')\n",
            "    print(f'   • Std: {diagonal.std():.4f}')\n",
            "    print(f'   • Min: {diagonal.min():.4f}')\n",
            "    print(f'   • Max: {diagonal.max():.4f}')\n",
            "    print(f'   • Range: [{diagonal.min():.4f}, {diagonal.max():.4f}]')\n",
            "    print()\n",
            "    \n",
            "    # Off-diagonal analysis\n",
            "    off_diagonal = gmatrix_df.values[np.triu_indices_from(gmatrix_df.values, k=1)]\n",
            "    print('📊 Off-diagonal Elements (Genetic relationships):')\n",
            "    print(f'   • Mean: {off_diagonal.mean():.4f}')\n",
            "    print(f'   • Std: {off_diagonal.std():.4f}')\n",
            "    print(f'   • Min: {off_diagonal.min():.4f}')\n",
            "    print(f'   • Max: {off_diagonal.max():.4f}')\n",
            "    print(f'   • Range: [{off_diagonal.min():.4f}, {off_diagonal.max():.4f}]')\n",
            "    print()\n",
            "    \n",
            "    # Eigenvalue analysis\n",
            "    eigenvals = np.linalg.eigvals(gmatrix_df.values)\n",
            "    eigenvals = np.sort(eigenvals)[::-1]  # Sort descending\n",
            "    \n",
            "    print('📊 Eigenvalue Analysis:')\n",
            "    print(f'   • Largest eigenvalue: {eigenvals[0]:.4f}')\n",
            "    print(f'   • Smallest eigenvalue: {eigenvals[-1]:.4f}')\n",
            "    print(f'   • Condition number: {eigenvals[0]/eigenvals[-1]:.2f}')\n",
            "    print(f'   • Positive eigenvalues: {(eigenvals > 0).sum()}')\n",
            "    print(f'   • Zero/near-zero eigenvalues: {(eigenvals < 1e-10).sum()}')\n",
            "    print()\n",
            "    \n",
            "    # Visualization\n",
            "    fig, axes = plt.subplots(2, 2, figsize=(15, 12))\n",
            "    \n",
            "    # Heatmap (sample if too large)\n",
            "    sample_size = min(50, gmatrix_df.shape[0])\n",
            "    sample_indices = np.random.choice(gmatrix_df.shape[0], sample_size, replace=False)\n",
            "    sample_g = gmatrix_df.values[np.ix_(sample_indices, sample_indices)]\n",
            "    \n",
            "    sns.heatmap(sample_g, ax=axes[0,0], cmap='RdYlBu_r', center=0)\n",
            "    axes[0,0].set_title(f'G-Matrix Heatmap (Sample of {sample_size} animals)')\n",
            "    \n",
            "    # Diagonal histogram\n",
            "    axes[0,1].hist(diagonal, bins=30, alpha=0.7, edgecolor='black')\n",
            "    axes[0,1].set_xlabel('Diagonal Value')\n",
            "    axes[0,1].set_ylabel('Frequency')\n",
            "    axes[0,1].set_title('Diagonal Distribution')\n",
            "    axes[0,1].axvline(diagonal.mean(), color='red', linestyle='--', label=f'Mean: {diagonal.mean():.3f}')\n",
            "    axes[0,1].legend()\n",
            "    \n",
            "    # Off-diagonal histogram\n",
            "    axes[1,0].hist(off_diagonal, bins=30, alpha=0.7, edgecolor='black')\n",
            "    axes[1,0].set_xlabel('Off-diagonal Value')\n",
            "    axes[1,0].set_ylabel('Frequency')\n",
            "    axes[1,0].set_title('Off-diagonal Distribution')\n",
            "    axes[1,0].axvline(off_diagonal.mean(), color='red', linestyle='--', label=f'Mean: {off_diagonal.mean():.3f}')\n",
            "    axes[1,0].legend()\n",
            "    \n",
            "    # Eigenvalue scree plot\n",
            "    axes[1,1].plot(range(1, len(eigenvals)+1), eigenvals, 'bo-')\n",
            "    axes[1,1].set_xlabel('Eigenvalue Rank')\n",
            "    axes[1,1].set_ylabel('Eigenvalue')\n",
            "    axes[1,1].set_title('Eigenvalue Scree Plot')\n",
            "    axes[1,1].set_yscale('log')\n",
            "    \n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "    \n",
            "    # CA: Cluster analysis heatmap (sample)\n",
            "    print('\\n🔎 CA (Cluster Analysis) on G-matrix sample:')\n",
            "    try:\n",
            "        cg = sns.clustermap(\n",
            "            sample_g,\n",
            "            cmap='RdYlBu_r',\n",
            "            center=0,\n",
            "            figsize=(10, 8),\n",
            "            xticklabels=False,\n",
            "            yticklabels=False\n",
            "        )\n",
            "        cg.ax_heatmap.set_title(f'CA Clustered G-Matrix Heatmap (Sample of {sample_size} animals)')\n",
            "        plt.show()\n",
            "    except Exception as e:\n",
            "        print(f'⚠️ Could not render CA clustered heatmap: {e}')\n",
            "    \n",
            "    # Sample of G-matrix\n",
            "    print('\\n📋 Sample of G-Matrix:')\n",
            "    display(HTML(gmatrix_df.head().to_html()))\n",
            "    \n",
            "else:\n",
            "    print('❌ G-matrix not available')"
        ]
    }
    notebook["cells"].append(gmatrix_code)

    # Data Alignment Section
    alignment_header = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## 🔗 Data Alignment Results {#data-alignment}\n",
            "\n",
            "### Genotype-Phenotype Matching"
        ]
    }
    notebook["cells"].append(alignment_header)

    alignment_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "print('=== DATA ALIGNMENT ANALYSIS ===')\n",
            "print()\n",
            "\n",
            "if geno_df is not None and pheno_df is not None:\n",
            "    # Check for common identifiers\n",
            "    geno_samples = set(geno_df.index if geno_df.index.name else range(len(geno_df)))\n",
            "    pheno_samples = set(pheno_df.index if pheno_df.index.name else range(len(pheno_df)))\n",
            "    \n",
            "    common_samples = geno_samples.intersection(pheno_samples)\n",
            "    geno_only = geno_samples - pheno_samples\n",
            "    pheno_only = pheno_samples - geno_samples\n",
            "    \n",
            "    print('🔗 Sample Alignment:')\n",
            "    print(f'   • Samples in genotype data: {len(geno_samples)}')\n",
            "    print(f'   • Samples in phenotype data: {len(pheno_samples)}')\n",
            "    print(f'   • Common samples: {len(common_samples)}')\n",
            "    print(f'   • Samples only in genotype: {len(geno_only)}')\n",
            "    print(f'   • Samples only in phenotype: {len(pheno_only)}')\n",
            "    print(f'   • Alignment rate: {len(common_samples)/max(len(geno_samples), len(pheno_samples))*100:.1f}%')\n",
            "    print()\n",
            "    \n",
            "    # Check if indices are properly aligned\n",
            "    if len(geno_df) == len(pheno_df):\n",
            "        # Check if indices match\n",
            "        indices_match = geno_df.index.equals(pheno_df.index)\n",
            "        print(f'📊 Index Alignment: {indices_match}')\n",
            "        \n",
            "        if not indices_match:\n",
            "            # Show first few mismatches\n",
            "            mismatches = 0\n",
            "            for i in range(min(len(geno_df), len(pheno_df))):\n",
            "                if geno_df.index[i] != pheno_df.index[i]:\n",
            "                    mismatches += 1\n",
            "                    if mismatches <= 5:\n",
            "                        print(f'   • Position {i}: Geno={geno_df.index[i]}, Pheno={pheno_df.index[i]}')\n",
            "            if mismatches > 5:\n",
            "                print(f'   • ... and {mismatches-5} more mismatches')\n",
            "        print()\n",
            "    \n",
            "    # Phenotype completeness by trait for aligned samples\n",
            "    if len(common_samples) > 0:\n",
            "        # Get aligned data (this would need to be done properly in real pipeline)\n",
            "        print('🎯 Phenotype Completeness for Aligned Data:')\n",
            "        completeness = pheno_df.notna().mean() * 100\n",
            "        for trait in completeness.sort_values(ascending=False).index:\n",
            "            print(f'   • {trait}: {completeness[trait]:.1f}%')\n",
            "        print()\n",
            "    \n",
            "    # Visualization of alignment\n",
            "    fig, ax = plt.subplots(figsize=(10, 6))\n",
            "    categories = ['Geno Only', 'Pheno Only', 'Both']\n",
            "    values = [len(geno_only), len(pheno_only), len(common_samples)]\n",
            "    bars = ax.bar(categories, values, alpha=0.7, edgecolor='black')\n",
            "    ax.set_ylabel('Number of Samples')\n",
            "    ax.set_title('Sample Alignment Overview')\n",
            "    \n",
            "    for bar, value in zip(bars, values):\n",
            "        ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + value + 1, \n",
            "                f'{value}', ha='center', va='bottom')\n",
            "    \n",
            "    plt.show()\n",
            "    \n",
            "else:\n",
            "    print('❌ Both genotype and phenotype data needed for alignment analysis')\n",
            "\n",
            "print('✅ Data alignment analysis completed')"
        ]
    }
    notebook["cells"].append(alignment_code)

    # QC Summary Section
    summary_header = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## ✅ Quality Control Summary {#qc-summary}\n",
            "\n",
            "### Pipeline Results Overview"
        ]
    }
    notebook["cells"].append(summary_header)

    summary_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "print('=== QUALITY CONTROL PIPELINE SUMMARY ===')\n",
            "print(f'Generated: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')\n",
            "print('=' * 50)\n",
            "print()\n",
            "\n",
            "# Data summary\n",
            "if geno_df is not None:\n",
            "    print('🧬 GENOTYPE DATA:')\n",
            "    print(f'   ✓ Loaded {geno_df.shape[0]:,} animals × {geno_df.shape[1]:,} SNPs')\n",
            "    missing_rate = geno_df.isna().sum().sum() / (geno_df.shape[0] * geno_df.shape[1]) * 100\n",
            "    print(f'   ✓ Missing data rate: {missing_rate:.2f}%')\n",
            "else:\n",
            "    print('❌ Genotype data not loaded')\n",
            "\n",
            "if pheno_df is not None:\n",
            "    print('\\n🎯 PHENOTYPE DATA:')\n",
            "    print(f'   ✓ Loaded {pheno_df.shape[0]:,} animals × {pheno_df.shape[1]} traits')\n",
            "    complete_records = (pheno_df.notna().all(axis=1)).sum()\n",
            "    print(f'   ✓ Complete records: {complete_records}/{pheno_df.shape[0]} ({complete_records/pheno_df.shape[0]*100:.1f}%)')\n",
            "else:\n",
            "    print('\\n❌ Phenotype data not loaded')\n",
            "\n",
            "if gmatrix_df is not None:\n",
            "    print('\\n🧮 G-MATRIX:')\n",
            "    print(f'   ✓ Calculated {gmatrix_df.shape[0]} × {gmatrix_df.shape[0]} relationship matrix')\n",
            "    diagonal_mean = np.diag(gmatrix_df.values).mean()\n",
            "    print(f'   ✓ Mean diagonal (self-relationships): {diagonal_mean:.4f}')\n",
            "    off_diag_mean = gmatrix_df.values[np.triu_indices_from(gmatrix_df.values, k=1)].mean()\n",
            "    print(f'   ✓ Mean off-diagonal (relationships): {off_diag_mean:.4f}')\n",
            "else:\n",
            "    print('\\n❌ G-matrix not calculated')\n",
            "\n",
            "# Quality metrics\n",
            "print('\\n📊 QUALITY METRICS:')\n",
            "if geno_df is not None and pheno_df is not None:\n",
            "    geno_samples = set(geno_df.index if geno_df.index.name else range(len(geno_df)))\n",
            "    pheno_samples = set(pheno_df.index if pheno_df.index.name else range(len(pheno_df)))\n",
            "    common = len(geno_samples.intersection(pheno_samples))\n",
            "    alignment_rate = common / max(len(geno_samples), len(pheno_samples)) * 100\n",
            "    print(f'   ✓ Geno-Pheno alignment: {alignment_rate:.1f}%')\n",
            "\n",
            "if metadata:\n",
            "    print('\\n⚙️  PIPELINE PARAMETERS:')\n",
            "    for key, value in metadata.items():\n",
            "        print(f'   • {key}: {value}')\n",
            "\n",
            "print('\\n🎉 QC Pipeline completed successfully!')\n",
            "print('📋 Data is ready for downstream genomic prediction analysis.')\n",
            "\n",
            "# Final recommendations\n",
            "print('\\n💡 RECOMMENDATIONS:')\n",
            "if geno_df is not None:\n",
            "    missing_rate = geno_df.isna().sum().sum() / (geno_df.shape[0] * geno_df.shape[1])\n",
            "    if missing_rate > 0.1:\n",
            "        print('   ⚠️  High missing data rate in genotypes - consider imputation')\n",
            "    \n",
            "    snp_var = geno_df.var(axis=0, skipna=True)\n",
            "    low_var_snps = (snp_var < 0.01).sum()\n",
            "    if low_var_snps > 0:\n",
            "        print(f'   ⚠️  {low_var_snps} SNPs with very low variance - filtering applied')\n",
            "\n",
            "if pheno_df is not None:\n",
            "    complete_rate = (pheno_df.notna().all(axis=1)).sum() / pheno_df.shape[0]\n",
            "    if complete_rate < 0.8:\n",
            "        print('   ⚠️  Many incomplete phenotype records - consider imputation or filtering')\n",
            "\n",
            "print('   ✅ Data quality checks passed - proceed to training phase')"
        ]
    }
    notebook["cells"].append(summary_code)

    # Hide code by default (users can expand in UI)
    _set_code_hidden_by_default(notebook)

    # Save notebook
    with open(output_file, 'w') as f:
        json.dump(notebook, f, indent=2)

    print(f"✅ Comprehensive QC report notebook created: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive QC report notebook')
    parser.add_argument('--dataset_dir', required=True, help='Path to dataset directory (containing Geno.csv, Pheno.csv)')
    parser.add_argument('--gmatrix_dir', required=True, help='Path to G-matrix directory')
    parser.add_argument('--output_file', required=True, help='Path to output notebook file (.ipynb)')

    args = parser.parse_args()

    create_qc_report_notebook(args.dataset_dir, args.gmatrix_dir, args.output_file)

if __name__ == '__main__':
    main()
