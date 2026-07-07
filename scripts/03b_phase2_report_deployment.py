#!/usr/bin/env python3
"""
Generate Phase 2 Deployment Report with EDA
Creates a Jupyter notebook documenting EDA on prediction input file and deployment process
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime

def create_deployment_report_notebook(models_dir, deployment_dir, output_file):
    """Create a Jupyter notebook with deployment summary"""
    
    models_path = Path(models_dir)
    deployment_path = Path(deployment_dir)
    
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
            "# Phase 2: Deployment Summary\n",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "**Phase:** Deployment & Prediction\n",
            "\n",
            "This notebook summarizes the deployment stage, including model deployment information, deployed models, and deployment artifacts.\n",
            "\n",
            "### Report Contents:\n",
            "\n",
            "1. **Deployment Overview**: Summary of the deployment process\n",
            "2. **Deployed Models**: Information about models deployed from Phase 1\n",
            "3. **Deployment Artifacts**: Files and resources available for prediction\n",
            "4. **Model Statistics**: Summary statistics of deployed models"
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
            "1. [Deployment Overview](#deployment-overview)\n",
            "2. [Deployment Directory Structure](#directory-structure)\n",
            "3. [Deployed Models](#deployed-models)\n",
            "4. [Variance Filter Mask](#variance-mask)\n",
            "5. [Training Summary](#training-summary)\n",
            "6. [Deployment Performance](#deployment-performance)"
        ]
    }
    notebook["cells"].append(toc_cell)
    
    # Deployment Overview
    overview_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## 🚀 Deployment Overview {#deployment-overview}\n",
            "\n",
            "The deployment stage prepares models trained in Phase 1 for use in Phase 2 (Prediction). During deployment:\n",
            "\n",
            "1. **Model Serialization**: Trained models from Phase 1 are saved as `.pkl` files\n",
            "2. **Feature Mask Preservation**: The variance filter mask from Phase 1 is saved for consistent feature selection\n",
            "3. **Model Organization**: Models are organized by trait and algorithm\n",
            "4. **Deployment Artifacts**: All necessary files for prediction are stored in the deployment directory\n",
            "\n",
            "### Key Components:\n",
            "\n",
            "- **Deployed Models**: Serialized models ready for prediction\n",
            "- **Variance Filter Mask**: Ensures prediction uses same features as training\n",
            "- **Training Summary**: Metadata about the training process\n",
            "- **Deployment Scripts**: Scripts used for model deployment"
        ]
    }
    notebook["cells"].append(overview_cell)
    
    # Deployment Directory Structure
    directory_header = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## 📁 Deployment Directory Structure {#directory-structure}"
        ]
    }
    notebook["cells"].append(directory_header)
    
    # Directory Structure Code
    directory_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import json\n",
            "from pathlib import Path\n",
            "from IPython.display import display, HTML\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "# Define paths\n",
            f"PROJECT_ROOT = Path('{Path(deployment_dir).parent.parent}')\n",
            f"DEPLOYMENT_DIR = Path('{deployment_dir}')\n",
            f"MODELS_DIR = Path('{models_dir}')\n",
            "\n",
            "print('='*80)\n",
            "print('DEPLOYMENT DIRECTORY STRUCTURE')\n",
            "print('='*80)\n",
            "print(f'\\n📁 Deployment directory: {DEPLOYMENT_DIR}')\n",
            "\n",
            "if not DEPLOYMENT_DIR.exists():\n",
            "    print(f'❌ Deployment directory not found at {DEPLOYMENT_DIR}')\n",
            "else:\n",
            "    print(f'✅ Deployment directory exists')\n",
            "    \n",
            "    # List all files and directories\n",
            "    print(f'\\n📊 Contents:')\n",
            "    \n",
            "    # Count files by type\n",
            "    pkl_files = list(DEPLOYMENT_DIR.glob('*.pkl'))\n",
            "    csv_files = list(DEPLOYMENT_DIR.glob('*.csv'))\n",
            "    json_files = list(DEPLOYMENT_DIR.glob('*.json'))\n",
            "    npy_files = list(DEPLOYMENT_DIR.glob('*.npy'))\n",
            "    sh_files = list(DEPLOYMENT_DIR.glob('*.sh'))\n",
            "    log_files = list(DEPLOYMENT_DIR.glob('*.log'))\n",
            "    \n",
            "    print(f'  • Model files (.pkl): {len(pkl_files)}')\n",
            "    print(f'  • CSV files: {len(csv_files)}')\n",
            "    print(f'  • JSON files: {len(json_files)}')\n",
            "    print(f'  • NumPy files (.npy): {len(npy_files)}')\n",
            "    print(f'  • Shell scripts (.sh): {len(sh_files)}')\n",
            "    print(f'  • Log files (.log): {len(log_files)}')\n",
            "    \n",
            "    # Check for models subdirectory\n",
            "    models_subdir = DEPLOYMENT_DIR / 'models'\n",
            "    if models_subdir.exists():\n",
            "        model_subdirs = [d for d in models_subdir.iterdir() if d.is_dir()]\n",
            "        print(f'  • Model subdirectories: {len(model_subdirs)}')\n",
            "        if model_subdirs:\n",
            "            print(f'    Sample subdirectories: {[d.name for d in model_subdirs[:10]]}')"
        ]
    }
    notebook["cells"].append(directory_code)
    
    # Deployed Models Section
    models_header = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## 🤖 Deployed Models {#deployed-models}\n",
            "\n",
            "Information about the models deployed from Phase 1."
        ]
    }
    notebook["cells"].append(models_header)
    
    # Models Code
    models_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "print('='*80)\n",
            "print('DEPLOYED MODELS')\n",
            "print('='*80)\n",
            "\n",
            "# Check models directory\n",
            "models_subdir = DEPLOYMENT_DIR / 'models'\n",
            "\n",
            "if models_subdir.exists():\n",
            "    print(f'\\n📁 Models directory: {models_subdir}')\n",
            "    \n",
            "    # Get all model files\n",
            "    all_model_files = list(models_subdir.rglob('*.pkl'))\n",
            "    print(f'\\n📊 Total model files: {len(all_model_files)}')\n",
            "    \n",
            "    # Organize by trait and algorithm\n",
            "    model_info = []\n",
            "    for model_file in all_model_files:\n",
            "        # Parse filename to extract trait and algorithm\n",
            "        parts = model_file.stem.split('_')\n",
            "        if len(parts) >= 2:\n",
            "            trait = '_'.join(parts[:-1])\n",
            "            algorithm = parts[-1]\n",
            "        else:\n",
            "            trait = 'Unknown'\n",
            "            algorithm = model_file.stem\n",
            "        \n",
            "        model_info.append({\n",
            "            'Trait': trait,\n",
            "            'Algorithm': algorithm,\n",
            "            'File': model_file.name,\n",
            "            'Size (MB)': model_file.stat().st_size / (1024**2)\n",
            "        })\n",
            "    \n",
            "    if model_info:\n",
            "        models_df = pd.DataFrame(model_info)\n",
            "        print(f'\\n📋 Model Summary:')\n",
            "        print(f'  • Unique traits: {models_df[\"Trait\"].nunique()}')\n",
            "        print(f'  • Unique algorithms: {models_df[\"Algorithm\"].nunique()}')\n",
            "        print(f'  • Total models: {len(models_df)}')\n",
            "        \n",
            "        print(f'\\n📊 Models by Trait:')\n",
            "        trait_counts = models_df.groupby('Trait').size().sort_values(ascending=False)\n",
            "        display(trait_counts.head(20))\n",
            "        \n",
            "        print(f'\\n📊 Models by Algorithm:')\n",
            "        algo_counts = models_df.groupby('Algorithm').size().sort_values(ascending=False)\n",
            "        display(algo_counts)\n",
            "        \n",
            "        print(f'\\n📋 Sample Models (first 10):')\n",
            "        display(models_df.head(10))\n",
            "    else:\n",
            "        print('⚠️  No model files found in models directory')\n",
            "else:\n",
            "    print(f'⚠️  Models directory not found: {models_subdir}')"
        ]
    }
    notebook["cells"].append(models_code)
    
    # Variance Filter Mask Section
    variance_header = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## 🔧 Variance Filter Mask {#variance-mask}\n",
            "\n",
            "The variance filter mask ensures that prediction uses the same features as training."
        ]
    }
    notebook["cells"].append(variance_header)
    
    variance_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "print('='*80)\n",
            "print('VARIANCE FILTER MASK')\n",
            "print('='*80)\n",
            "\n",
            "variance_mask_file = DEPLOYMENT_DIR / 'variance_filter_mask.npy'\n",
            "\n",
            "if variance_mask_file.exists():\n",
            "    variance_mask = np.load(variance_mask_file)\n",
            "    \n",
            "    print(f'\\n✅ Variance filter mask loaded: {variance_mask_file.name}')\n",
            "    print(f'\\n📊 Mask Statistics:')\n",
            "    print(f'  • Mask shape: {variance_mask.shape}')\n",
            "    print(f'  • Total features: {len(variance_mask):,}')\n",
            "    print(f'  • Features kept (high variance): {np.sum(variance_mask):,} ({np.sum(variance_mask)/len(variance_mask)*100:.2f}%)')\n",
            "    print(f'  • Features removed (low variance): {np.sum(~variance_mask):,} ({np.sum(~variance_mask)/len(variance_mask)*100:.2f}%)')\n",
            "    print(f'  • File size: {variance_mask_file.stat().st_size / 1024:.2f} KB')\n",
            "else:\n",
            "    print(f'❌ Variance filter mask not found: {variance_mask_file}')"
        ]
    }
    notebook["cells"].append(variance_code)
    
    # Training Summary Section
    training_header = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## 📊 Training Summary {#training-summary}\n",
            "\n",
            "Summary information from the training phase."
        ]
    }
    notebook["cells"].append(training_header)
    
    training_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "print('='*80)\n",
            "print('TRAINING SUMMARY')\n",
            "print('='*80)\n",
            "\n",
            "training_summary_file = DEPLOYMENT_DIR / 'training_summary.json'\n",
            "\n",
            "if training_summary_file.exists():\n",
            "    with open(training_summary_file, 'r') as f:\n",
            "        training_summary = json.load(f)\n",
            "    \n",
            "    print(f'\\n✅ Training summary loaded: {training_summary_file.name}')\n",
            "    print(f'\\n📊 Training Summary:')\n",
            "    \n",
            "    # Display summary information\n",
            "    for key, value in training_summary.items():\n",
            "        if isinstance(value, (dict, list)):\n",
            "            print(f'\\n  • {key}:')\n",
            "            if isinstance(value, dict):\n",
            "                for k, v in value.items():\n",
            "                    print(f'    - {k}: {v}')\n",
            "            else:\n",
            "                print(f'    {value[:10] if len(value) > 10 else value}...')\n",
            "        else:\n",
            "            print(f'  • {key}: {value}')\n",
            "else:\n",
            "    print(f'⚠️  Training summary file not found: {training_summary_file}')"
        ]
    }
    notebook["cells"].append(training_code)
    
    # Deployment Performance Section
    performance_header = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## 📈 Deployment Performance {#deployment-performance}\n",
            "\n",
            "Performance metrics and statistics from the deployment process."
        ]
    }
    notebook["cells"].append(performance_header)
    
    performance_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "print('='*80)\n",
            "print('DEPLOYMENT PERFORMANCE')\n",
            "print('='*80)\n",
            "\n",
            "performance_file = DEPLOYMENT_DIR / 'deployment_performance_summary.csv'\n",
            "\n",
            "if performance_file.exists():\n",
            "    perf_df = pd.read_csv(performance_file)\n",
            "    \n",
            "    print(f'\\n✅ Deployment performance summary loaded: {performance_file.name}')\n",
            "    print(f'\\n📊 Performance Summary:')\n",
            "    print(f'  • Shape: {perf_df.shape}')\n",
            "    print(f'  • Columns: {list(perf_df.columns)}')\n",
            "    \n",
            "    display(perf_df.head(20))\n",
            "    \n",
            "    if len(perf_df) > 0:\n",
            "        print(f'\\n📈 Summary Statistics:')\n",
            "        display(perf_df.describe())\n",
            "else:\n",
            "    print(f'⚠️  Deployment performance file not found: {performance_file}')"
        ]
    }
    notebook["cells"].append(performance_code)
    
    # Deployment Status
    status_header = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## ✅ Deployment Status\n",
            "\n",
            "### Deployment Artifacts Checklist:\n",
            "\n",
            "- ✅ Variance filter mask (`variance_filter_mask.npy`)\n",
            "- ✅ Deployed models (`.pkl` files in `models/` directory)\n",
            "- ✅ Training summary (`training_summary.json`)\n",
            "- ✅ Deployment scripts (`.sh` files)\n",
            "\n",
            "### Next Steps:\n",
            "\n",
            "1. **EDA & G-Matrix Analysis**: Run `2.1_Preprocessing_report.ipynb` to analyze input data\n",
            "2. **Prediction**: Use deployed models to generate predictions\n",
            "3. **Results Analysis**: Review predictions in `2.3_prediction_report.ipynb`"
        ]
    }
    notebook["cells"].append(status_header)
    
    # Save notebook
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"✅ Phase 2 Deployment Summary notebook created: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate Phase 2 Deployment Summary notebook')
    parser.add_argument('--models_dir', required=True, help='Path to models directory')
    parser.add_argument('--deployment_dir', required=True, help='Path to deployment directory')
    parser.add_argument('--output_file', required=True, help='Path to output notebook file (.ipynb)')
    
    args = parser.parse_args()
    
    create_deployment_report_notebook(
        args.models_dir,
        args.deployment_dir,
        args.output_file
    )

if __name__ == "__main__":
    main()
