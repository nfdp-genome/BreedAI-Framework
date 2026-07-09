#!/usr/bin/env python3
"""
Generate comprehensive report notebook for train-validate-test results
Creates a Jupyter notebook with visualizations and analysis
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime

def _set_code_hidden_by_default(notebook):
    """Collapse/hide code cells by default while keeping them expandable."""
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            metadata = cell.setdefault("metadata", {})
            metadata["collapsed"] = True
            jupyter_meta = metadata.setdefault("jupyter", {})
            jupyter_meta["source_hidden"] = True
            # nbformat v4 requires these on every code cell
            cell.setdefault("outputs", [])
            cell.setdefault("execution_count", None)

def create_report_notebook(results_file, output_file):
    """Create a Jupyter notebook with comprehensive analysis"""

    # Handle glob patterns for results files
    if '*' in str(results_file):
        import glob
        matching_files = glob.glob(str(results_file))
        if not matching_files:
            print(f"❌ No results files found matching: {results_file}")
            return
        # Use the most recent file
        results_file = max(matching_files, key=lambda f: Path(f).stat().st_mtime)
        print(f"📊 Using results file: {results_file}")

    # Load results
    results_df = pd.read_csv(results_file)
    
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
            "# Train-Validate-Test Performance Report\n",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"**Results File:** {Path(results_file).name}\n",
            "\n",
            "This report provides comprehensive analysis of algorithm performance across train, validation, and test sets."
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
            "2. [Train Set Results](#train-set-results)\n",
            "   - [Best Models by Trait](#train-best-models)\n",
            "   - [Algorithm Rankings](#train-algorithm-rankings)\n",
            "   - [Ensemble vs Best Model Comparison](#train-ensemble-comparison)\n",
            "   - [Performance Tables](#train-performance-tables)\n",
            "3. [Validation Set Results](#validation-set-results)\n",
            "   - [Best Models by Trait](#val-best-models)\n",
            "   - [Algorithm Rankings](#val-algorithm-rankings)\n",
            "   - [Ensemble vs Best Model Comparison](#val-ensemble-comparison)\n",
            "   - [Performance Tables](#val-performance-tables)\n",
            "4. [Test Set Results](#test-set-results)\n",
            "   - [Best Models by Trait](#test-best-models)\n",
            "   - [Algorithm Rankings](#test-algorithm-rankings)\n",
            "   - [Ensemble vs Best Model Comparison](#test-ensemble-comparison)\n",
            "   - [Performance Tables](#test-performance-tables)\n",
            "5. [Stacking Preferences](#stacking-preferences)\n",
            "   - [Per-Trait Preference Profile](#stacking-per-trait)\n",
            "   - [Stability Analysis](#stacking-stability)\n",
            "6. [Cross-Set Analysis](#cross-set-analysis)\n",
            "   - [Performance Across Sets](#performance-across-sets)\n",
            "   - [Overfitting Analysis](#overfitting-analysis)\n",
            "7. [Summary & Recommendations](#summary-recommendations)"
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
            "\n",
            "# Set style\n",
            "plt.style.use('seaborn-v0_8')\n",
            "sns.set_palette('husl')\n",
            "\n",
            "# Resolve the repo root from the working directory (portable; no placeholders)\n",
            "_repo = Path.cwd()\n",
            "while _repo != _repo.parent and not (_repo / 'environment.yml').exists():\n",
            "    _repo = _repo.parent\n",
            "DATA_DIR = _repo / 'Phase1_Learning_Benchmarking' / 'training_validation'\n",
            "print(f\"📁 Data directory: {DATA_DIR}\")\n",
            "\n",
            "# Load results\n",
            "results_file = DATA_DIR / 'combined_train_validate_results.csv'\n",
            "results_df = pd.read_csv(results_file)\n",
            "\n",
            "# Separate individual algorithms and ensembles\n",
            "individual_df = results_df[~results_df['algorithm'].str.startswith('Ensemble_')].copy()\n",
            "ensemble_df = results_df[results_df['algorithm'].str.startswith('Ensemble_')].copy()\n",
            "\n",
            "print(f'✅ Loaded {len(results_df)} results')\n",
            "print(f'📊 Traits: {results_df[\"trait\"].nunique()}')\n",
            "print(f'🔧 Individual Algorithms: {individual_df[\"algorithm\"].nunique()}')\n",
            "print(f'🎯 Ensemble Methods: {ensemble_df[\"algorithm\"].nunique() if not ensemble_df.empty else 0}')"
        ]
    }
    notebook["cells"].append(load_cell)
    
    # Data Overview
    overview_cell = {
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
    notebook["cells"].append(overview_cell)
    
    overview_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Dataset summary\n",
            "print('=== Dataset Summary ===')\n",
            "print(f'Total Results: {len(results_df)}')\n",
            "print(f'Traits Analyzed: {results_df[\"trait\"].nunique()}')\n",
            "print(f'\\nTraits:')\n",
            "for trait in sorted(results_df['trait'].unique()):\n",
            "    trait_count = len(results_df[results_df['trait'] == trait])\n",
            "    print(f'  - {trait}: {trait_count} algorithm results')\n",
            "\n",
            "print(f'\\nAlgorithms Evaluated:')\n",
            "for alg in sorted(results_df['algorithm'].unique()):\n",
            "    alg_count = len(results_df[results_df['algorithm'] == alg])\n",
            "    print(f'  - {alg}: {alg_count} trait results')\n",
            "\n",
            "# Data split summary\n",
            "if 'n_train' in results_df.columns:\n",
            "    sample_row = results_df.iloc[0]\n",
            "    print(f'\\n=== Data Split ===')\n",
            "    print(f'Train: {int(sample_row[\"n_train\"])} samples ({sample_row[\"n_train\"]/(sample_row[\"n_train\"]+sample_row[\"n_val\"]+sample_row[\"n_test\"])*100:.1f}%)')\n",
            "    print(f'Validation: {int(sample_row[\"n_val\"])} samples ({sample_row[\"n_val\"]/(sample_row[\"n_train\"]+sample_row[\"n_val\"]+sample_row[\"n_test\"])*100:.1f}%)')\n",
            "    print(f'Test: {int(sample_row[\"n_test\"])} samples ({sample_row[\"n_test\"]/(sample_row[\"n_train\"]+sample_row[\"n_val\"]+sample_row[\"n_test\"])*100:.1f}%)')"
        ]
    }
    notebook["cells"].append(overview_code)

    stacking_header = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## 🎯 Preference (Stacking Weights) {#stacking-preferences}\n",
            "\n",
            "Convert stacking weights into a publishable preference profile. This section appears when stacking artifacts are present."
        ]
    }
    notebook["cells"].append(stacking_header)

    stacking_profile_header = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Per-Trait Preference Profile {#stacking-per-trait}\n",
            "\n",
            "Per trait, report model and family preference as **mean ± SD** across outer folds (and replicates if present), plus a top-k model summary."
        ]
    }
    notebook["cells"].append(stacking_profile_header)

    stacking_profile_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "stack_dirs = sorted(DATA_DIR.glob('stacking_*'))\n",
            "if not stack_dirs:\n",
            "    print('No stacking artifacts found.')\n",
            "else:\n",
            "    fold_frames = []\n",
            "    model_summary_frames = []\n",
            "    for d in stack_dirs:\n",
            "        trait = d.name.replace('stacking_', '')\n",
            "        fold_file = d / 'stacking_weights_by_fold.csv'\n",
            "        model_file = d / 'stacking_weights_summary.csv'\n",
            "        family_file = d / 'stacking_family_weights_summary.csv'\n",
            "        if fold_file.exists():\n",
            "            fdf = pd.read_csv(fold_file)\n",
            "            fdf['trait'] = trait\n",
            "            if 'replicate' not in fdf.columns:\n",
            "                fdf['replicate'] = 0\n",
            "            fold_frames.append(fdf)\n",
            "        if model_file.exists():\n",
            "            mdf = pd.read_csv(model_file)\n",
            "            mdf['trait'] = trait\n",
            "            model_summary_frames.append(mdf)\n",
            "\n",
            "    if not fold_frames:\n",
            "        print('No fold-level stacking files found.')\n",
            "    else:\n",
            "        all_folds = pd.concat(fold_frames, ignore_index=True)\n",
            "        all_folds['weight'] = pd.to_numeric(all_folds['weight'], errors='coerce')\n",
            "        all_folds = all_folds.dropna(subset=['weight'])\n",
            "\n",
            "        model_family_map = {}\n",
            "        if model_summary_frames:\n",
            "            all_models_summary = pd.concat(model_summary_frames, ignore_index=True)\n",
            "            for _, row in all_models_summary.iterrows():\n",
            "                model_family_map[row['model']] = row.get('family', 'other')\n",
            "        all_folds['family'] = all_folds['model'].map(model_family_map).fillna('other')\n",
            "\n",
            "        model_pref = all_folds.groupby(['trait', 'model', 'family'], as_index=False)['weight'].agg(['mean', 'std', 'count']).reset_index()\n",
            "        model_pref = model_pref.rename(columns={'mean': 'mean_weight', 'std': 'sd_weight', 'count': 'n'})\n",
            "        model_pref['sd_weight'] = model_pref['sd_weight'].fillna(0.0)\n",
            "        model_pref['mean±SD'] = model_pref['mean_weight'].map(lambda x: f'{x:.4f}') + ' ± ' + model_pref['sd_weight'].map(lambda x: f'{x:.4f}')\n",
            "\n",
            "        family_pref = all_folds.groupby(['trait', 'family'], as_index=False)['weight'].agg(['mean', 'std', 'count']).reset_index()\n",
            "        family_pref = family_pref.rename(columns={'mean': 'mean_weight', 'std': 'sd_weight', 'count': 'n'})\n",
            "        family_pref['sd_weight'] = family_pref['sd_weight'].fillna(0.0)\n",
            "        family_pref['mean±SD'] = family_pref['mean_weight'].map(lambda x: f'{x:.4f}') + ' ± ' + family_pref['sd_weight'].map(lambda x: f'{x:.4f}')\n",
            "\n",
            "        top_k = 10\n",
            "        for trait in sorted(model_pref['trait'].unique()):\n",
            "            print(f'\\n=== {trait}: Model Preference (mean ± SD) ===')\n",
            "            trait_models = model_pref[model_pref['trait'] == trait].sort_values('mean_weight', ascending=False)\n",
            "            display(trait_models[['model', 'family', 'mean±SD', 'n']].reset_index(drop=True))\n",
            "\n",
            "            print(f'\\n=== {trait}: Family Preference (mean ± SD) ===')\n",
            "            trait_families = family_pref[family_pref['trait'] == trait].sort_values('mean_weight', ascending=False)\n",
            "            display(trait_families[['family', 'mean±SD', 'n']].reset_index(drop=True))\n",
            "\n",
            "            print(f'\\n=== {trait}: Top-{top_k} Models ===')\n",
            "            display(trait_models.head(top_k)[['model', 'family', 'mean_weight', 'sd_weight']].round(4).reset_index(drop=True))"
        ]
    }
    notebook["cells"].append(stacking_profile_code)

    stacking_stability_header = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Stability Analysis {#stacking-stability}\n",
            "\n",
            "Summarize stability using coefficient of variation (CV) and visualize fold-to-fold variability."
        ]
    }
    notebook["cells"].append(stacking_stability_header)

    stacking_stability_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "stack_dirs = sorted(DATA_DIR.glob('stacking_*'))\n",
            "if not stack_dirs:\n",
            "    print('No stacking artifacts found.')\n",
            "else:\n",
            "    fold_frames = []\n",
            "    model_summary_frames = []\n",
            "    for d in stack_dirs:\n",
            "        trait = d.name.replace('stacking_', '')\n",
            "        fold_file = d / 'stacking_weights_by_fold.csv'\n",
            "        model_file = d / 'stacking_weights_summary.csv'\n",
            "        if fold_file.exists():\n",
            "            fdf = pd.read_csv(fold_file)\n",
            "            fdf['trait'] = trait\n",
            "            if 'replicate' not in fdf.columns:\n",
            "                fdf['replicate'] = 0\n",
            "            fold_frames.append(fdf)\n",
            "        if model_file.exists():\n",
            "            mdf = pd.read_csv(model_file)\n",
            "            mdf['trait'] = trait\n",
            "            model_summary_frames.append(mdf)\n",
            "\n",
            "    if not fold_frames:\n",
            "        print('No fold-level stacking files found.')\n",
            "    else:\n",
            "        all_folds = pd.concat(fold_frames, ignore_index=True)\n",
            "        all_folds['weight'] = pd.to_numeric(all_folds['weight'], errors='coerce')\n",
            "        all_folds = all_folds.dropna(subset=['weight'])\n",
            "\n",
            "        model_family_map = {}\n",
            "        if model_summary_frames:\n",
            "            all_models_summary = pd.concat(model_summary_frames, ignore_index=True)\n",
            "            for _, row in all_models_summary.iterrows():\n",
            "                model_family_map[row['model']] = row.get('family', 'other')\n",
            "        all_folds['family'] = all_folds['model'].map(model_family_map).fillna('other')\n",
            "\n",
            "        stability = all_folds.groupby(['trait', 'model', 'family'], as_index=False)['weight'].agg(['mean', 'std']).reset_index()\n",
            "        stability = stability.rename(columns={'mean': 'mean_weight', 'std': 'sd_weight'})\n",
            "        stability['sd_weight'] = stability['sd_weight'].fillna(0.0)\n",
            "        eps = 1e-12\n",
            "        stability['cv'] = stability['sd_weight'] / np.maximum(np.abs(stability['mean_weight']), eps)\n",
            "\n",
            "        print('=== Stability Metrics (Coefficient of Variation) ===')\n",
            "        display(stability.sort_values(['trait', 'cv'], ascending=[True, False]).round(4))\n",
            "\n",
            "        for trait in sorted(all_folds['trait'].unique()):\n",
            "            trait_df = all_folds[all_folds['trait'] == trait].copy()\n",
            "            trait_mean = trait_df.groupby('model', as_index=False)['weight'].mean().sort_values('weight', ascending=False)\n",
            "            top_models = trait_mean.head(10)['model'].tolist()\n",
            "            plot_df = trait_df[trait_df['model'].isin(top_models)].copy()\n",
            "            if plot_df.empty:\n",
            "                continue\n",
            "            plot_df = plot_df.groupby(['replicate', 'fold', 'model'], as_index=False)['weight'].mean()\n",
            "\n",
            "            plt.figure(figsize=(12, 5))\n",
            "            sns.lineplot(data=plot_df, x='fold', y='weight', hue='model', marker='o')\n",
            "            plt.title(f'{trait}: Fold-to-Fold Variability (Top 10 Models)')\n",
            "            plt.xlabel('Outer Fold')\n",
            "            plt.ylabel('Weight')\n",
            "            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')\n",
            "            plt.tight_layout()\n",
            "            plt.show()\n",
            "\n",
            "            heat_df = plot_df.pivot_table(index='model', columns='fold', values='weight', aggfunc='mean')\n",
            "            plt.figure(figsize=(10, max(4, 0.4 * len(heat_df))))\n",
            "            sns.heatmap(heat_df, annot=True, fmt='.3f', cmap='viridis')\n",
            "            plt.title(f'{trait}: Outer-Fold Weight Heatmap (Top 10 Models)')\n",
            "            plt.xlabel('Outer Fold')\n",
            "            plt.ylabel('Model')\n",
            "            plt.tight_layout()\n",
            "            plt.show()"
        ]
    }
    notebook["cells"].append(stacking_stability_code)
    
    # Helper function to create section
    def create_section(section_name, section_id, metric_prefix, metric_label):
        """Create a complete section for train/validate/test results"""

        # Calculate statistics for this section
        best_col = f"{metric_prefix}_r2"
        best_per_trait = results_df.loc[results_df.groupby('trait')[best_col].idxmax()].copy()
        best_per_trait = best_per_trait.sort_values(best_col, ascending=False)

        perf_cols = ['trait', 'algorithm', f'{metric_prefix}_r2', f'{metric_prefix}_pearson_r',
                     f'{metric_prefix}_rmse', f'{metric_prefix}_mae', f'{metric_prefix}_bias']
        perf_table = results_df[perf_cols].copy()
        perf_table = perf_table.sort_values(['trait', f'{metric_prefix}_r2'], ascending=[True, False])
        perf_table = perf_table.round(4)

        cells = []

        # Section header
        header = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"---\n",
                f"\n",
                f"## {section_name} Results {{#{section_id}}}\n",
                f"\n",
                f"This section presents all results for the **{section_name.lower()}** set."
            ]
        }
        cells.append(header)

        # Best models by trait
        best_header = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"### Best Models by Trait {{#{section_id.replace('set', 'best-models')}}}\n",
                f"\n",
                f"Best performing algorithm for each trait on the {section_name.lower()} set (ranked by {metric_label})."
            ]
        }
        cells.append(best_header)

        best_code = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                f"# Best model per trait on {section_name.lower()} set\n",
                f"best_col = '{metric_prefix}_r2'\n",
                f"best_per_trait = results_df.loc[results_df.groupby('trait')[best_col].idxmax()].copy()\n",
                f"best_per_trait = best_per_trait.sort_values(best_col, ascending=False)\n",
                f"\n",
                f"# Create comprehensive table\n",
                f"best_table = best_per_trait[[\n",
                f"    'trait', 'algorithm',\n",
                f"    '{metric_prefix}_r2', '{metric_prefix}_pearson_r',\n",
                f"    '{metric_prefix}_rmse', '{metric_prefix}_mae', '{metric_prefix}_bias'\n",
                f"]].round(4)\n",
                f"\n",
                f"# Rename columns for display\n",
                f"best_table.columns = ['Trait', 'Best Algorithm', 'R²', 'Pearson r', 'RMSE', 'MAE', 'Bias']\n",
                f"\n",
                f"print('=== Best Algorithm per Trait ({section_name} Set) ===')\n",
                f"print('Ranked by {metric_label}')\n",
                f"print()\n",
                f"display(HTML(best_table.to_html(index=False, classes='table table-striped')))\n",
                f"\n",
                f"# Summary statistics\n",
                f"print(f'\\n=== Summary Statistics ===')\n",
                f"print(f'Mean R²: {best_per_trait[best_col].mean():.4f}')\n",
                f"print(f'Median R²: {best_per_trait[best_col].median():.4f}')\n",
                f"print(f'Best R²: {best_per_trait[best_col].max():.4f}')\n",
                f"print(f'Worst R²: {best_per_trait[best_col].min():.4f}')"
            ]
        }
        cells.append(best_code)
        
        # Algorithm rankings
        rankings_header = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"### Algorithm Rankings {{#{section_id.replace('set', 'algorithm-rankings')}}}\n",
                f"\n",
                f"Average performance across all traits for each algorithm on the {section_name.lower()} set."
            ]
        }
        cells.append(rankings_header)
        
        rankings_code = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                f"# Algorithm rankings on {section_name.lower()} set\n",
                f"r2_col = '{metric_prefix}_r2'\n",
                f"pearson_col = '{metric_prefix}_pearson_r'\n",
                f"rmse_col = '{metric_prefix}_rmse'\n",
                f"mae_col = '{metric_prefix}_mae'\n",
                f"alg_rankings = individual_df.groupby('algorithm').agg({{\n",
                f"    r2_col: ['mean', 'std', 'min', 'max'],\n",
                f"    pearson_col: ['mean', 'std'],\n",
                f"    rmse_col: 'mean',\n",
                f"    mae_col: 'mean'\n",
                f"}}).round(4)\n",
                f"\n",
                f"# Flatten column names\n",
                f"alg_rankings.columns = ['_'.join(col).strip() for col in alg_rankings.columns.values]\n",
                f"r2_mean_col = f'{{r2_col}}_mean'\n",
                f"r2_std_col = f'{{r2_col}}_std'\n",
                f"alg_rankings = alg_rankings.sort_values(r2_mean_col, ascending=False)\n",
                f"\n",
                f"# Rename for display\n",
                f"pearson_mean_col = f'{{pearson_col}}_mean'\n",
                f"rmse_mean_col = f'{{rmse_col}}_mean'\n",
                f"mae_mean_col = f'{{mae_col}}_mean'\n",
                f"display_cols = [r2_mean_col, r2_std_col, pearson_mean_col, rmse_mean_col, mae_mean_col]\n",
                f"display_table = alg_rankings[display_cols].copy()\n",
                f"display_table.columns = ['R² (Mean)', 'R² (Std)', 'Pearson r (Mean)', 'RMSE (Mean)', 'MAE (Mean)']\n",
                f"\n",
                f"print('=== Algorithm Rankings ({section_name} Set) ===')\n",
                f"display(HTML(display_table.to_html(classes='table table-striped')))\n",
                f"\n",
                f"# Visualization\n",
                f"fig, ax = plt.subplots(figsize=(12, 6))\n",
                f"x = np.arange(len(alg_rankings))\n",
                f"ax.barh(x, alg_rankings[r2_mean_col], \n",
                f"        xerr=alg_rankings[r2_std_col], capsize=5)\n",
                f"ax.set_yticks(x)\n",
                f"ax.set_yticklabels(alg_rankings.index, rotation=0)\n",
                f"ax.set_xlabel('R² (Mean ± Std)')\n",
                f"ax.set_title(f'Algorithm Performance Ranking ({section_name} Set)')\n",
                f"ax.invert_yaxis()\n",
                f"plt.tight_layout()\n",
                f"plt.show()"
            ]
        }
        cells.append(rankings_code)
        
        # Ensemble vs Best Model Comparison
        ensemble_header = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"### Ensemble vs Best Model Comparison {{#{section_id.replace('set', 'ensemble-comparison')}}}\n",
                f"\n",
                f"Comparison of ensemble methods against the best individual algorithm for each trait on the {section_name.lower()} set."
            ]
        }
        cells.append(ensemble_header)
        
        ensemble_code = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                f"# Ensemble vs Best Model Comparison on {section_name.lower()} set\n",
                f"if not ensemble_df.empty:\n",
                f"    comparison_data = []\n",
                f"    \n",
                f"    for trait in sorted(results_df['trait'].unique()):\n",
                f"        trait_individual = individual_df[individual_df['trait'] == trait]\n",
                f"        trait_ensemble = ensemble_df[ensemble_df['trait'] == trait]\n",
                f"        \n",
                f"        if len(trait_individual) > 0 and len(trait_ensemble) > 0:\n",
                f"            # Best individual algorithm\n",
                f"            r2_col = '{metric_prefix}_r2'\n",
                f"            best_individual = trait_individual.loc[trait_individual[r2_col].idxmax()]\n",
                f"            \n",
                f"            # Best ensemble\n",
                f"            best_ensemble = trait_ensemble.loc[trait_ensemble[r2_col].idxmax()]\n",
                f"            \n",
                f"            # Calculate improvement\n",
                f"            individual_r2 = best_individual[r2_col]\n",
                f"            ensemble_r2 = best_ensemble[r2_col]\n",
                f"            improvement = ensemble_r2 - individual_r2\n",
                f"            improvement_pct = (improvement / abs(individual_r2) * 100) if individual_r2 != 0 else 0\n",
                f"            \n",
                f"            comparison_data.append({{\n",
                f"                'Trait': trait,\n",
                f"                'Best Individual': best_individual['algorithm'],\n",
                f"                'Best Individual R²': individual_r2,\n",
                f"                'Best Ensemble': best_ensemble['algorithm'],\n",
                f"                'Best Ensemble R²': ensemble_r2,\n",
                f"                'Improvement': improvement,\n",
                f"                'Improvement %': improvement_pct\n",
                f"            }})\n",
                f"    \n",
                f"    comparison_df = pd.DataFrame(comparison_data)\n",
                f"    comparison_df = comparison_df.sort_values('Improvement', ascending=False)\n",
                f"    comparison_df = comparison_df.round(4)\n",
                f"    \n",
                f"    print('=== Ensemble vs Best Individual Algorithm ({section_name} Set) ===')\n",
                f"    display(HTML(comparison_df.to_html(index=False, classes='table table-striped')))\n",
                f"    \n",
                f"    # Summary\n",
                f"    print('\\n=== Summary ===')\n",
                f"    better_count = (comparison_df['Improvement'] > 0).sum()\n",
                f"    worse_count = (comparison_df['Improvement'] < 0).sum()\n",
                f"    mean_improvement = comparison_df['Improvement'].mean()\n",
                f"    median_improvement = comparison_df['Improvement'].median()\n",
                f"    print(f'Traits where ensemble is better: {{better_count}}')\n",
                f"    print(f'Traits where individual is better: {{worse_count}}')\n",
                f"    print(f'Mean improvement: {{mean_improvement:.4f}}')\n",
                f"    print(f'Median improvement: {{median_improvement:.4f}}')\n",
                f"    \n",
                f"    # Visualization\n",
                f"    fig, ax = plt.subplots(figsize=(14, 8))\n",
                f"    x = np.arange(len(comparison_df))\n",
                f"    width = 0.35\n",
                f"    ax.bar(x - width/2, comparison_df['Best Individual R²'], width, label='Best Individual', alpha=0.8)\n",
                f"    ax.bar(x + width/2, comparison_df['Best Ensemble R²'], width, label='Best Ensemble', alpha=0.8)\n",
                f"    ax.set_xlabel('Trait')\n",
                f"    ax.set_ylabel('R²')\n",
                f"    ax.set_title(f'Ensemble vs Best Individual Algorithm ({section_name} Set)')\n",
                f"    ax.set_xticks(x)\n",
                f"    ax.set_xticklabels(comparison_df['Trait'], rotation=45, ha='right')\n",
                f"    ax.legend()\n",
                f"    ax.grid(axis='y', alpha=0.3)\n",
                f"    plt.tight_layout()\n",
                f"    plt.show()\n",
                f"else:\n",
                f"    print('No ensemble results available')"
            ]
        }
        cells.append(ensemble_code)
        
        # Performance tables
        tables_header = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"### Complete Performance Tables {{#{section_id.replace('set', 'performance-tables')}}}\n",
                f"\n",
                f"Detailed performance metrics for all algorithms and traits on the {section_name.lower()} set."
            ]
        }
        cells.append(tables_header)
        
        tables_code = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                f"# Complete performance table for {section_name.lower()} set\n",
                f"perf_cols = ['trait', 'algorithm',\n",
                f"            f'{metric_prefix}_r2', f'{metric_prefix}_pearson_r',\n",
                f"            f'{metric_prefix}_rmse', f'{metric_prefix}_mae', f'{metric_prefix}_bias']\n",
                f"\n",
                f"perf_table = results_df[perf_cols].copy()\n",
                f"perf_table = perf_table.sort_values(['trait', f'{metric_prefix}_r2'], ascending=[True, False])\n",
                f"perf_table = perf_table.round(4)\n",
                f"\n",
                f"# Rename columns\n",
                f"perf_table.columns = ['Trait', 'Algorithm', 'R²', 'Pearson r', 'RMSE', 'MAE', 'Bias']\n",
                f"\n",
                f"print(f'=== Complete Performance Table ({section_name} Set) ===')\n",
                f"print(f'Total entries: {{len(perf_table)}}')\n",
                f"print()\n",
                f"\n",
                f"# Display by trait for better readability\n",
                f"for trait_name in sorted(perf_table['Trait'].unique()):\n",
                f"    trait_data = perf_table[perf_table['Trait'] == trait_name].copy()\n",
                f"    print(f'\\n**{{trait_name}}**')\n",
                f"    display(HTML(trait_data.to_html(index=False, classes='table table-striped table-sm')))"
            ]
        }
        cells.append(tables_code)
        
        return cells
    
    # Add Train section
    notebook["cells"].extend(create_section("Train", "train-set", "train", "Train R²"))
    
    # Add Validation section
    notebook["cells"].extend(create_section("Validation", "validation-set", "val", "Validation R²"))
    
    # Add Test section
    notebook["cells"].extend(create_section("Test", "test-set", "test", "Test R²"))
    
    # Cross-set analysis
    cross_header = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## 📈 Cross-Set Analysis {#cross-set-analysis}\n",
            "\n",
            "### Performance Across Sets"
        ]
    }
    notebook["cells"].append(cross_header)
    
    cross_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Performance across sets\n",
            "fig, axes = plt.subplots(1, 3, figsize=(18, 6))\n",
            "\n",
            "# R² comparison\n",
            "individual_df = results_df[~results_df['algorithm'].str.startswith('Ensemble_')]\n",
            "\n",
            "for idx, (set_name, col) in enumerate([('Train', 'train_r2'), ('Validation', 'val_r2'), ('Test', 'test_r2')]):\n",
            "    data = [individual_df[individual_df['trait'] == trait][col].values \n",
            "            for trait in sorted(individual_df['trait'].unique())]\n",
            "    axes[idx].boxplot(data, labels=sorted(individual_df['trait'].unique()))\n",
            "    axes[idx].set_title(f'{set_name} R² by Trait')\n",
            "    axes[idx].set_ylabel('R²')\n",
            "    axes[idx].tick_params(axis='x', rotation=45)\n",
            "    axes[idx].grid(axis='y', alpha=0.3)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "# Algorithm performance across sets\n",
            "alg_cross = individual_df.groupby('algorithm').agg({\n",
            "    'train_r2': 'mean',\n",
            "    'val_r2': 'mean',\n",
            "    'test_r2': 'mean'\n",
            "}).round(4).sort_values('test_r2', ascending=False)\n",
            "\n",
            "print('=== Algorithm Performance Across Sets (Mean R²) ===')\n",
            "display(HTML(alg_cross.to_html(classes='table table-striped')))\n",
            "\n",
            "# Visualization\n",
            "fig, ax = plt.subplots(figsize=(14, 6))\n",
            "x = np.arange(len(alg_cross))\n",
            "width = 0.25\n",
            "ax.bar(x - width, alg_cross['train_r2'], width, label='Train R²', alpha=0.8)\n",
            "ax.bar(x, alg_cross['val_r2'], width, label='Validation R²', alpha=0.8)\n",
            "ax.bar(x + width, alg_cross['test_r2'], width, label='Test R²', alpha=0.8)\n",
            "ax.set_xlabel('Algorithm')\n",
            "ax.set_ylabel('R²')\n",
            "ax.set_title('Algorithm Performance Comparison Across Sets')\n",
            "ax.set_xticks(x)\n",
            "ax.set_xticklabels(alg_cross.index, rotation=45, ha='right')\n",
            "ax.legend()\n",
            "ax.grid(axis='y', alpha=0.3)\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    }
    notebook["cells"].append(cross_code)
    
    # Overfitting analysis
    overfit_header = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Overfitting Analysis {#overfitting-analysis}\n",
            "\n",
            "Overfitting indicator = Train R² - Test R²"
        ]
    }
    notebook["cells"].append(overfit_header)
    
    overfit_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Overfitting analysis\n",
            "results_df['overfitting'] = results_df['train_r2'] - results_df['test_r2']\n",
            "\n",
            "overfit_by_alg = individual_df.groupby('algorithm')['overfitting_indicator'].agg(['mean', 'std']).round(4)\n",
            "overfit_by_alg = overfit_by_alg.sort_values('mean')\n",
            "\n",
            "print('=== Overfitting Analysis by Algorithm ===')\n",
            "display(HTML(overfit_by_alg.to_html(classes='table table-striped')))\n",
            "\n",
            "# Visualization\n",
            "fig, ax = plt.subplots(figsize=(12, 6))\n",
            "x = np.arange(len(overfit_by_alg))\n",
            "ax.barh(x, overfit_by_alg['mean'], xerr=overfit_by_alg['std'], capsize=5)\n",
            "ax.set_yticks(x)\n",
            "ax.set_yticklabels(overfit_by_alg.index)\n",
            "ax.set_xlabel('Overfitting Indicator (Train R² - Test R²)')\n",
            "ax.set_title('Overfitting Analysis by Algorithm')\n",
            "ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='No Overfitting')\n",
            "ax.legend()\n",
            "ax.grid(axis='x', alpha=0.3)\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    }
    notebook["cells"].append(overfit_code)
    
    # Summary and recommendations
    summary_header = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## 📊 Summary & Recommendations {#summary-recommendations}\n",
            "\n",
            "### Overall Best Performers"
        ]
    }
    notebook["cells"].append(summary_header)
    
    summary_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Overall summary\n",
            "print('=== Overall Summary ===')\n",
            "print()\n",
            "\n",
            "# Best algorithm overall (by test R²)\n",
            "best_overall = individual_df.loc[individual_df['test_r2'].idxmax()]\n",
            "print(f'Best Overall Algorithm (Test Set):')\n",
            "print(f'  Algorithm: {best_overall[\"algorithm\"]}')\n",
            "print(f'  Trait: {best_overall[\"trait\"]}')\n",
            "print(f'  Test R²: {best_overall[\"test_r2\"]:.4f}')\n",
            "print(f'  Test Pearson r: {best_overall[\"test_pearson_r\"]:.4f}')\n",
            "print()\n",
            "\n",
            "# Best algorithm on average\n",
            "best_avg = individual_df.groupby('algorithm')['test_r2'].mean().sort_values(ascending=False)\n",
            "print(f'Best Algorithm on Average (across all traits):')\n",
            "print(f'  Algorithm: {best_avg.index[0]}')\n",
            "print(f'  Mean Test R²: {best_avg.iloc[0]:.4f}')\n",
            "print()\n",
            "\n",
            "# Ensemble summary\n",
            "if not ensemble_df.empty:\n",
            "    best_ensemble_avg = ensemble_df.groupby('algorithm')['test_r2'].mean().sort_values(ascending=False)\n",
            "    print(f'Best Ensemble on Average:')\n",
            "    print(f'  Ensemble: {best_ensemble_avg.index[0]}')\n",
            "    print(f'  Mean Test R²: {best_ensemble_avg.iloc[0]:.4f}')\n",
            "    print()\n",
            "    \n",
            "    # Compare ensemble to best individual\n",
            "    ensemble_improvement = best_ensemble_avg.iloc[0] - best_avg.iloc[0]\n",
            "    print(f'Ensemble vs Best Individual:')\n",
            "    print(f'  Improvement: {ensemble_improvement:.4f}')\n",
            "    if ensemble_improvement > 0:\n",
            "        print(f'  ✅ Ensemble performs better on average')\n",
            "    else:\n",
            "        print(f'  ⚠️  Individual algorithm performs better on average')\n",
            "\n",
            "print()\n",
            "print('=== Recommendations ===')\n",
            "print('1. Review the detailed tables above to find the best algorithm for each trait')\n",
            "print('2. Consider ensemble methods if they show consistent improvement')\n",
            "print('3. Check overfitting indicators to ensure model generalization')\n",
            "print('4. Use validation set performance for model selection')\n",
            "print('5. Test set performance provides final unbiased evaluation')"
        ]
    }
    notebook["cells"].append(summary_code)
    
    # Hide code by default (users can expand in UI)
    _set_code_hidden_by_default(notebook)

    # Save notebook
    with open(output_file, 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"✅ Comprehensive report notebook created: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive report notebook for train-validate-test results')
    parser.add_argument('--results_file', required=True, help='Path to combined results CSV file')
    parser.add_argument('--output_file', required=True, help='Path to output notebook file (.ipynb)')
    
    args = parser.parse_args()
    
    create_report_notebook(args.results_file, args.output_file)

if __name__ == '__main__':
    main()
