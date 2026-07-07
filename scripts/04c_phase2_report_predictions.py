#!/usr/bin/env python3
"""
Generate Phase 2 Prediction Results Report
Creates a comprehensive Jupyter notebook showing all algorithm and ensemble predictions
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

def create_prediction_results_report(results_dir, output_file):
    """Create a Jupyter notebook with comprehensive prediction results analysis"""
    
    results_path = Path(results_dir)
    
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
            "# Phase 2: Prediction Results Report\n",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "**Phase:** Deployment & Prediction\n",
            "\n",
            "This report provides comprehensive analysis of prediction results from all algorithms and ensemble methods.\n",
            "\n",
            "### Report Contents:\n",
            "\n",
            "1. **Prediction Process**: Explanation of how predictions were generated\n",
            "2. **Data Loading & Summary**: Load results and overall statistics\n",
            "3. **Animal-wise Predictions**: Animals on rows — Weighted Average, all algorithms + ensembles, ensembles only\n",
            "4. **Export Files**: List generated files and export report tables to CSV"
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
            "1. [Prediction Process Overview](#prediction-process)\n",
            "2. [Data Loading](#data-loading)\n",
            "3. [Summary Statistics](#summary)\n",
            "4. [Animal-wise Predictions](#animal-predictions)\n",
            "   - [4.1 Weighted Average (all traits)](#animal-weighted-average)\n",
            "   - [4.2 All algorithms + 3 ensembles](#animal-all-algorithms-ensembles)\n",
            "   - [4.3 Ensembles only](#animal-ensembles-only)\n",
            "5. [Export Files](#export-files)"
        ]
    }
    notebook["cells"].append(toc_cell)
    
    # Prediction Process Overview
    process_header = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## 🔮 Prediction Process Overview {#prediction-process}\n",
            "\n",
            "### How Predictions Were Generated:\n",
            "\n",
            "1. **Input Data Loading**: Load the prediction input file (Geno.csv)\n",
            "2. **Feature Validation**: Verify input file has same number of features as training phase\n",
            "3. **Variance Filtering**: Apply the same variance filter mask from Phase 1/Deployment\n",
            "4. **Missing Data Handling**: Apply mode imputation (same as training phase)\n",
            "5. **G-Matrix Calculation**: Calculate genomic relationship matrix (if needed)\n",
            "6. **Model Loading**: Load deployed models for each trait-algorithm combination\n",
            "7. **Prediction**: Generate predictions using all algorithms for all traits\n",
            "8. **Ensemble Creation**: Create ensemble predictions (Simple Average, Median, Weighted Average)\n",
            "9. **Results Export**: Save predictions to CSV files\n",
            "\n",
            "### Key Characteristics:\n",
            "\n",
            "- **Feature Consistency**: Same features as training/deployment phase (validated via variance mask)\n",
            "- **Preprocessing Consistency**: Same missing data handling as training phase\n",
            "- **Batch Processing**: Input data is divided into batches for efficient processing\n",
            "- **All Algorithms**: Predictions from all algorithms deployed in Phase 2\n",
            "- **Ensemble Methods**: Three ensemble methods are created:\n",
            "  - **Simple Average**: Mean of all algorithm predictions\n",
            "  - **Median**: Median of all algorithm predictions\n",
            "  - **Weighted Average**: Weighted mean (Bayesian-style weighting based on algorithm types)\n",
            "\n",
            "### Output Files:\n",
            "\n",
            "1. **Individual trait CSVs**: `{TraitName}_predictions.csv` - One file per trait with all algorithms\n",
            "2. **Combined all-algorithms CSV**: `all_predictions_all_algorithms_*.csv` - All algorithms for all traits (one row per animal)\n",
            "3. **Ensemble-only CSV**: `ensemble_weighted_average_predictions_*.csv` - Weighted Average ensemble for all traits\n",
            "4. **Excel file**: `all_predictions_*.xlsx` - All predictions in Excel format (one sheet per trait)"
        ]
    }
    notebook["cells"].append(process_header)
    
    # Data Loading
    load_header = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## 📥 Data Loading {#data-loading}"
        ]
    }
    notebook["cells"].append(load_header)
    
    load_code = {
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
            "plt.rcParams['figure.figsize'] = (14, 8)\n",
            "\n",
            f"# Define results directory\n",
            f"RESULTS_DIR = Path('{results_dir}')\n",
            "\n",
            "print('🔍 Loading prediction results...')\n",
            "\n",
            "# Find all prediction CSV files\n",
            "prediction_files = list(RESULTS_DIR.glob('*_predictions.csv'))\n",
            "\n",
            "# Exclude combined files\n",
            "prediction_files = [f for f in prediction_files if 'all_predictions' not in f.name and 'ensemble' not in f.name]\n",
            "\n",
            "print(f'✅ Found {len(prediction_files)} trait prediction files')\n",
            "\n",
            "# Load all prediction files\n",
            "all_predictions = {}\n",
            "for pred_file in prediction_files:\n",
            "    trait_name = pred_file.stem.replace('_predictions', '').replace('_', ' ')\n",
            "    df = pd.read_csv(pred_file, index_col=0)\n",
            "    all_predictions[trait_name] = df\n",
            "    print(f'  • {trait_name}: {df.shape[0]} animals, {df.shape[1]} algorithms')\n",
            "\n",
            "# Load combined files if available\n",
            "combined_all_file = list(RESULTS_DIR.glob('all_predictions_all_algorithms_*.csv'))\n",
            "if combined_all_file:\n",
            "    combined_all_df = pd.read_csv(combined_all_file[0], index_col=0)\n",
            "    print(f'\\n✅ Loaded combined file: {combined_all_file[0].name}')\n",
            "    print(f'   Shape: {combined_all_df.shape}')\n",
            "else:\n",
            "    combined_all_df = None\n",
            "    print('\\n⚠️  Combined all-algorithms file not found')\n",
            "\n",
            "# Load ensemble-only file if available\n",
            "ensemble_file = list(RESULTS_DIR.glob('ensemble_weighted_average_predictions_*.csv'))\n",
            "if ensemble_file:\n",
            "    ensemble_df = pd.read_csv(ensemble_file[0], index_col=0)\n",
            "    print(f'✅ Loaded ensemble file: {ensemble_file[0].name}')\n",
            "    print(f'   Shape: {ensemble_df.shape}')\n",
            "else:\n",
            "    ensemble_df = None\n",
            "    print('⚠️  Ensemble-only file not found')"
        ]
    }
    notebook["cells"].append(load_code)
    
    # Summary Statistics
    summary_header = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## 📊 Summary Statistics {#summary}"
        ]
    }
    notebook["cells"].append(summary_header)
    
    summary_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "print('='*80)\n",
            "print('PREDICTION RESULTS SUMMARY')\n",
            "print('='*80)\n",
            "print(f'\\n📊 Overall Statistics:')\n",
            "print(f'  • Total traits predicted: {len(all_predictions)}')\n",
            "print(f'  • Total animals: {len(set().union(*[df.index.tolist() for df in all_predictions.values()]))}')\n",
            "\n",
            "# Count algorithms\n",
            "all_algorithms = set()\n",
            "all_ensembles = set()\n",
            "for df in all_predictions.values():\n",
            "    for col in df.columns:\n",
            "        if col.startswith('Ensemble_'):\n",
            "            all_ensembles.add(col)\n",
            "        else:\n",
            "            all_algorithms.add(col)\n",
            "\n",
            "print(f'  • Individual algorithms: {len(all_algorithms)}')\n",
            "print(f'    {sorted(all_algorithms)}')\n",
            "print(f'  • Ensemble methods: {len(all_ensembles)}')\n",
            "print(f'    {sorted(all_ensembles)}')\n",
            "\n",
            "# Check missing algorithms from audit file (if available)\n",
            "audit_file = RESULTS_DIR / 'prediction_models_audit.json'\n",
            "if audit_file.exists():\n",
            "    try:\n",
            "        import json\n",
            "        with open(audit_file, 'r') as f:\n",
            "            audit = json.load(f)\n",
            "        expected = set(audit.get('expected_algorithms', []))\n",
            "        missing_overall = sorted(list(expected - all_algorithms))\n",
            "        if missing_overall:\n",
            "            print('\\n⚠️  Missing expected algorithms (overall):')\n",
            "            print(f'  {missing_overall}')\n",
            "            # Explain common causes\n",
            "            bglr = [a for a in missing_overall if a in ('BayesA', 'BayesB', 'BayesCpi')]\n",
            "            gp = [a for a in missing_overall if a in ('GP_RBF', 'GP_Matern52')]\n",
            "            if bglr or gp:\n",
            "                print('\\n  Why these are often missing:')\n",
            "                if bglr:\n",
            "                    print('  • BGLR (BayesA/BayesB/BayesCpi): R-backed models; joblib save/load can fail.')\n",
            "                    print('    Re-run Phase 2 deployment so they are saved as .npz (custom format).')\n",
            "                if gp:\n",
            "                    print('  • GP_RBF / GP_Matern52: GPflow/TensorFlow models; joblib often fails to serialize.')\n",
            "                    print('    No custom save/load yet; these are skipped if serialization fails.')\n",
            "        else:\n",
            "            print('\\n✅ All expected algorithms are present')\n",
            "    except Exception as e:\n",
            "        print(f'⚠️  Could not read audit file: {e}')\n",
            "else:\n",
            "    print('\\nℹ️  No prediction_models_audit.json found in results directory')\n",
            "\n",
            "# Statistics per trait\n",
            "print(f'\\n📋 Per-Trait Statistics:')\n",
            "for trait_name, df in sorted(all_predictions.items()):\n",
            "    print(f'  • {trait_name}:')\n",
            "    print(f'    - Animals: {df.shape[0]}')\n",
            "    print(f'    - Algorithms: {df.shape[1]}')\n",
            "    print(f'    - Mean prediction range: [{df.values.min():.4f}, {df.values.max():.4f}]')"
        ]
    }
    notebook["cells"].append(summary_code)
    
    # Animal-wise Predictions Section
    animal_header = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## 🐄 Animal-wise Predictions {#animal-predictions}\n",
            "\n",
            "Animals on **rows**; each subsection shows a different view of predictions."
        ]
    }
    notebook["cells"].append(animal_header)

    # 4.1 Weighted Average (all traits)
    wa_header = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4.1 Animal prediction — Weighted Average (all traits) {#animal-weighted-average}\n",
            "\n",
            "Animals on rows; columns = **Weighted Average** ensemble for each trait."
        ]
    }
    notebook["cells"].append(wa_header)

    wa_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "if ensemble_df is not None:\n",
            "    print('📊 Weighted Average ensemble — all traits (animals × traits)')\n",
            "    print(f'   Shape: {ensemble_df.shape[0]} animals × {ensemble_df.shape[1]} traits\\n')\n",
            "    display(ensemble_df.head(15))\n",
            "    print(f'\\n... ({ensemble_df.shape[0]} rows total)')\n",
            "else:\n",
            "    # Build from combined file: columns containing Weighted_Average\n",
            "    if combined_all_df is not None:\n",
            "        wa_cols = [c for c in combined_all_df.columns if 'Weighted_Average' in c or 'Weighted Average' in c]\n",
            "        if wa_cols:\n",
            "            wa_df = combined_all_df[wa_cols].copy()\n",
            "            wa_df.columns = [c.replace('_Ensemble_Weighted_Average', '').replace('_Weighted_Average', '') for c in wa_df.columns]\n",
            "            print('📊 Weighted Average ensemble — all traits (animals × traits)')\n",
            "            print(f'   Shape: {wa_df.shape[0]} animals × {wa_df.shape[1]} traits\\n')\n",
            "            display(wa_df.head(15))\n",
            "            print(f'\\n... ({wa_df.shape[0]} rows total)')\n",
            "        else:\n",
            "            print('⚠️  No Weighted Average columns found in combined file')\n",
            "    else:\n",
            "        print('⚠️  Ensemble file and combined file not available')"
        ]
    }
    notebook["cells"].append(wa_code)

    # 4.2 All algorithms + 3 ensembles
    all_algo_header = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4.2 All algorithms + 3 ensembles {#animal-all-algorithms-ensembles}\n",
            "\n",
            "Animals on rows; columns = all **individual algorithms** plus **Simple Average**, **Median**, and **Weighted Average** for each trait."
        ]
    }
    notebook["cells"].append(all_algo_header)

    all_algo_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "if combined_all_df is not None:\n",
            "    print('📊 All algorithms + 3 ensembles (animals × trait-algorithm combinations)')\n",
            "    print(f'   Shape: {combined_all_df.shape[0]} animals × {combined_all_df.shape[1]} columns\\n')\n",
            "    display(combined_all_df.head(10).iloc[:, :20])\n",
            "    print(f'   ... (first 20 of {combined_all_df.shape[1]} columns; {combined_all_df.shape[0]} rows total)')\n",
            "else:\n",
            "    print('⚠️  Combined all-algorithms file not available')"
        ]
    }
    notebook["cells"].append(all_algo_code)

    # 4.3 Ensembles only
    ens_only_header = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4.3 Ensembles only {#animal-ensembles-only}\n",
            "\n",
            "Animals on rows; columns = **Simple Average**, **Median**, and **Weighted Average** for each trait (no individual algorithms)."
        ]
    }
    notebook["cells"].append(ens_only_header)

    ens_only_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "ensemble_cols = []\n",
            "if combined_all_df is not None:\n",
            "    ensemble_cols = [c for c in combined_all_df.columns if 'Ensemble_' in c]\n",
            "if ensemble_cols:\n",
            "    ens_only_df = combined_all_df[ensemble_cols].copy()\n",
            "    print('📊 Ensembles only — Simple Average, Median, Weighted Average (animals × trait-ensemble)')\n",
            "    print(f'   Shape: {ens_only_df.shape[0]} animals × {ens_only_df.shape[1]} columns\\n')\n",
            "    display(ens_only_df.head(15))\n",
            "    print(f'\\n... ({ens_only_df.shape[0]} rows total)')\n",
            "else:\n",
            "    print('⚠️  No ensemble columns found. Build from trait files if needed.')"
        ]
    }
    notebook["cells"].append(ens_only_code)

    # Export Files
    export_header = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## 📁 Export Files {#export-files}\n",
            "\n",
            "### Available pipeline outputs:\n",
            "\n",
            "1. **Individual trait files**: `*_predictions.csv` — one per trait, all algorithms\n",
            "2. **Combined all-algorithms**: `all_predictions_all_algorithms_*.csv` — all algorithms + ensembles (one row per animal)\n",
            "3. **Ensemble-only**: `ensemble_weighted_average_predictions_*.csv` — Weighted Average for all traits\n",
            "4. **Excel**: `all_predictions_*.xlsx` — all predictions (one sheet per trait)\n",
            "\n",
            "Below: list these files, then **export** the report tables (Weighted Average, all+ensembles, ensembles only) to CSV."
        ]
    }
    notebook["cells"].append(export_header)

    export_list_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "print('='*80)\n",
            "print('EXPORT FILES SUMMARY')\n",
            "print('='*80)\n",
            "\n",
            "print(f'\\n📁 Results directory: {RESULTS_DIR}')\n",
            "\n",
            "all_files = list(RESULTS_DIR.glob('*.csv')) + list(RESULTS_DIR.glob('*.xlsx'))\n",
            "print(f'\\n📊 Found {len(all_files)} result files:')\n",
            "for f in sorted(all_files):\n",
            "    size_mb = f.stat().st_size / (1024**2)\n",
            "    print(f'  • {f.name} ({size_mb:.2f} MB)')\n",
            "\n",
            "if combined_all_df is not None:\n",
            "    print(f'\\n✅ Combined all-algorithms file available')\n",
            "    print(f'   Columns: {combined_all_df.shape[1]}; Rows: {combined_all_df.shape[0]} (animals)')\n",
            "if ensemble_df is not None:\n",
            "    print(f'\\n✅ Ensemble-only file available')\n",
            "    print(f'   Columns: {ensemble_df.shape[1]} (traits); Rows: {ensemble_df.shape[0]} (animals)')"
        ]
    }
    notebook["cells"].append(export_list_code)

    export_save_header = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "#### Export report tables to CSV\n",
            "\n",
            "Run the cell below to save the report tables into the results directory."
        ]
    }
    notebook["cells"].append(export_save_header)

    export_save_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "from datetime import datetime\n",
            "ts = datetime.now().strftime('%Y%m%d_%H%M%S')\n",
            "export_dir = RESULTS_DIR / 'report_exports'\n",
            "export_dir.mkdir(parents=True, exist_ok=True)\n",
            "saved = []\n",
            "\n",
            "if ensemble_df is not None:\n",
            "    path = export_dir / f'report_weighted_average_{ts}.csv'\n",
            "    ensemble_df.to_csv(path)\n",
            "    saved.append(path.name)\n",
            "elif combined_all_df is not None:\n",
            "    wa_cols = [c for c in combined_all_df.columns if 'Weighted_Average' in c or 'Weighted Average' in c]\n",
            "    if wa_cols:\n",
            "        path = export_dir / f'report_weighted_average_{ts}.csv'\n",
            "        combined_all_df[wa_cols].to_csv(path)\n",
            "        saved.append(path.name)\n",
            "\n",
            "if combined_all_df is not None:\n",
            "    path = export_dir / f'report_all_algorithms_ensembles_{ts}.csv'\n",
            "    combined_all_df.to_csv(path)\n",
            "    saved.append(path.name)\n",
            "    enc = [c for c in combined_all_df.columns if 'Ensemble_' in c]\n",
            "    if enc:\n",
            "        path = export_dir / f'report_ensembles_only_{ts}.csv'\n",
            "        combined_all_df[enc].to_csv(path)\n",
            "        saved.append(path.name)\n",
            "\n",
            "if saved:\n",
            "    print(f'✅ Exported to {export_dir}:')\n",
            "    for n in saved:\n",
            "        print(f'   • {n}')\n",
            "else:\n",
            "    print('⚠️  No report tables available to export')"
        ]
    }
    notebook["cells"].append(export_save_code)
    
    # Save notebook
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"✅ Phase 2 Prediction Results report notebook created: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate Phase 2 Prediction Results report notebook')
    parser.add_argument('--results_dir', required=True, help='Path to prediction results directory')
    parser.add_argument('--output_file', required=True, help='Path to output notebook file (.ipynb)')
    
    args = parser.parse_args()
    
    create_prediction_results_report(args.results_dir, args.output_file)

if __name__ == '__main__':
    main()
