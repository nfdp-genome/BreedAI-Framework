#!/usr/bin/env python3
"""
Execute train-validate-test analysis and generate HTML report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import base64
from io import BytesIO

def create_html_report(results_file, output_file):
    """Create an HTML report with train-validate-test analysis"""

    # Set style for plots
    plt.style.use('seaborn-v0_8')
    sns.set_palette('husl')

    print("🔍 Loading train-validate-test results...")

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
    try:
        results_df = pd.read_csv(results_file)
        print(f'✅ Results loaded: {results_df.shape}')
    except Exception as e:
        print(f'❌ Error loading results: {e}')
        return

    # Separate individual algorithms and ensembles
    individual_df = results_df[~results_df['algorithm'].str.startswith('Ensemble_')].copy()
    ensemble_df = results_df[results_df['algorithm'].str.startswith('Ensemble_')].copy()

    print(f'📊 Individual Algorithms: {individual_df["algorithm"].nunique()}')
    print(f'🎯 Ensemble Methods: {ensemble_df["algorithm"].nunique() if not ensemble_df.empty else 0}')

    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Train-Validate-Test Performance Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .section {{ margin-bottom: 30px; }}
            .metric {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0; }}
            .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }}
            .stat-item {{ background: #e9ecef; padding: 10px; border-radius: 3px; text-align: center; }}
            table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .plot-container {{ margin: 20px 0; text-align: center; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
            .highlight {{ background-color: #fff3cd; padding: 10px; border-left: 4px solid #ffc107; }}
            .success {{ background-color: #d4edda; padding: 10px; border-left: 4px solid #28a745; }}
            .warning {{ background-color: #fff3cd; padding: 10px; border-left: 4px solid #ffc107; }}
        </style>
    </head>
    <body>
        <h1>🎯 Train-Validate-Test Performance Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Results File:</strong> {Path(results_file).name}</p>

        <div class="section">
            <h2>📋 Table of Contents</h2>
            <ol>
                <li><a href="#overview">Data Overview</a></li>
                <li><a href="#train">Train Set Results</a></li>
                <li><a href="#validation">Validation Set Results</a></li>
                <li><a href="#test">Test Set Results</a></li>
                <li><a href="#cross">Cross-Set Analysis</a></li>
                <li><a href="#summary">Summary & Recommendations</a></li>
            </ol>
        </div>

        <div id="overview" class="section">
            <h2>📊 Data Overview</h2>
            <div class="stats">
                <div class="stat-item">
                    <strong>Total Results</strong><br>
                    {len(results_df):,}
                </div>
                <div class="stat-item">
                    <strong>Traits Analyzed</strong><br>
                    {results_df["trait"].nunique()}
                </div>
                <div class="stat-item">
                    <strong>Individual Algorithms</strong><br>
                    {individual_df["algorithm"].nunique()}
                </div>
                <div class="stat-item">
                    <strong>Ensemble Methods</strong><br>
                    {ensemble_df["algorithm"].nunique() if not ensemble_df.empty else 0}
                </div>
            </div>

            <h3>Traits Analyzed</h3>
            <div style="display: flex; flex-wrap: wrap; gap: 5px;">
    """

    # Add trait tags
    for trait in sorted(results_df['trait'].unique()):
        trait_count = len(results_df[results_df['trait'] == trait])
        html_content += f'<span style="background: #e9ecef; padding: 3px 8px; border-radius: 3px; font-size: 0.9em;">{trait} ({trait_count})</span>'

    html_content += """
            </div>
        </div>
    """

    # Function to create section content
    def create_section_html(section_name, section_id, metric_prefix, metric_label):
        section_html = f"""
        <div id="{section_id}" class="section">
            <h2>{section_name} Results</h2>
        """

        # Best models by trait
        best_col = f"{metric_prefix}_r2"
        best_per_trait = results_df.loc[results_df.groupby('trait')[best_col].idxmax()].copy()
        best_per_trait = best_per_trait.sort_values(best_col, ascending=False)

        section_html += f"""
            <h3>Best Models by Trait</h3>
            <p>Best performing algorithm for each trait on the {section_name.lower()} set (ranked by {metric_label}).</p>

            <table>
                <tr>
                    <th>Trait</th>
                    <th>Best Algorithm</th>
                    <th>R²</th>
                    <th>Pearson r</th>
                    <th>RMSE</th>
                    <th>MAE</th>
                    <th>Bias</th>
                </tr>
        """

        for _, row in best_per_trait.iterrows():
            section_html += f"""
                <tr>
                    <td>{row['trait']}</td>
                    <td>{row['algorithm']}</td>
                    <td>{row[f'{metric_prefix}_r2']:.4f}</td>
                    <td>{row[f'{metric_prefix}_pearson_r']:.4f}</td>
                    <td>{row[f'{metric_prefix}_rmse']:.4f}</td>
                    <td>{row[f'{metric_prefix}_mae']:.4f}</td>
                    <td>{row[f'{metric_prefix}_bias']:.4f}</td>
                </tr>
            """

        section_html += "</table>"

        # Algorithm rankings
        r2_col = f"{metric_prefix}_r2"
        alg_rankings = individual_df.groupby('algorithm').agg({
            r2_col: ['mean', 'std', 'min', 'max'],
            f'{metric_prefix}_pearson_r': ['mean', 'std'],
            f'{metric_prefix}_rmse': 'mean',
            f'{metric_prefix}_mae': 'mean'
        }).round(4)

        # Flatten column names
        alg_rankings.columns = ['_'.join(col).strip() for col in alg_rankings.columns.values]
        r2_mean_col = f'{r2_col}_mean'
        r2_std_col = f'{r2_col}_std'
        alg_rankings = alg_rankings.sort_values(r2_mean_col, ascending=False)

        section_html += f"""
            <h3>Algorithm Rankings</h3>
            <p>Average performance across all traits for each algorithm on the {section_name.lower()} set.</p>

            <table>
                <tr>
                    <th>Algorithm</th>
                    <th>R² (Mean)</th>
                    <th>R² (Std)</th>
                    <th>Pearson r (Mean)</th>
                    <th>RMSE (Mean)</th>
                    <th>MAE (Mean)</th>
                </tr>
        """

        for alg in alg_rankings.index:
            section_html += f"""
                <tr>
                    <td>{alg}</td>
                    <td>{alg_rankings.loc[alg, r2_mean_col]:.4f}</td>
                    <td>{alg_rankings.loc[alg, r2_std_col]:.4f}</td>
                    <td>{alg_rankings.loc[alg, f'{metric_prefix}_pearson_r_mean']:.4f}</td>
                    <td>{alg_rankings.loc[alg, f'{metric_prefix}_rmse_mean']:.4f}</td>
                    <td>{alg_rankings.loc[alg, f'{metric_prefix}_mae_mean']:.4f}</td>
                </tr>
            """

        section_html += "</table>"

        # Create ranking plot
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(alg_rankings))
        ax.barh(x, alg_rankings[r2_mean_col],
                xerr=alg_rankings[r2_std_col], capsize=5)
        ax.set_yticks(x)
        ax.set_yticklabels(alg_rankings.index, rotation=0)
        ax.set_xlabel(f'{metric_label} (Mean ± Std)')
        ax.set_title(f'Algorithm Performance Ranking ({section_name} Set)')
        ax.invert_yaxis()
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        section_html += f"""
            <div class="plot-container">
                <img src="data:image/png;base64,{plot_data}" alt="Algorithm Rankings {section_name}">
            </div>
        """

        section_html += "</div>"
        return section_html

    # Add all sections
    html_content += create_section_html("Train", "train", "train", "Train R²")
    html_content += create_section_html("Validation", "validation", "val", "Validation R²")
    html_content += create_section_html("Test", "test", "test", "Test R²")

    # Cross-set analysis
    html_content += """
        <div id="cross" class="section">
            <h2>📈 Cross-Set Analysis</h2>

            <h3>Performance Across Sets</h3>
    """

    # Create cross-set comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (set_name, col) in enumerate([('Train', 'train_r2'), ('Validation', 'val_r2'), ('Test', 'test_r2')]):
        data = [individual_df[individual_df['trait'] == trait][col].values
                for trait in sorted(individual_df['trait'].unique())]
        axes[idx].boxplot(data, labels=sorted(individual_df['trait'].unique()))
        axes[idx].set_title(f'{set_name} R² by Trait')
        axes[idx].set_ylabel('R²')
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    html_content += f"""
            <div class="plot-container">
                <img src="data:image/png;base64,{plot_data}" alt="Performance Across Sets">
            </div>
    """

    # Algorithm performance across sets
    alg_cross = individual_df.groupby('algorithm').agg({
        'train_r2': 'mean',
        'val_r2': 'mean',
        'test_r2': 'mean'
    }).round(4).sort_values('test_r2', ascending=False)

    html_content += """
            <h3>Algorithm Performance Across Sets (Mean R²)</h3>
            <table>
                <tr>
                    <th>Algorithm</th>
                    <th>Train R²</th>
                    <th>Validation R²</th>
                    <th>Test R²</th>
                </tr>
    """

    for alg in alg_cross.index:
        html_content += f"""
                <tr>
                    <td>{alg}</td>
                    <td>{alg_cross.loc[alg, 'train_r2']:.4f}</td>
                    <td>{alg_cross.loc[alg, 'val_r2']:.4f}</td>
                    <td>{alg_cross.loc[alg, 'test_r2']:.4f}</td>
                </tr>
        """

    html_content += "</table>"

    # Algorithm comparison plot
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(alg_cross))
    width = 0.25
    ax.bar(x - width, alg_cross['train_r2'], width, label='Train R²', alpha=0.8)
    ax.bar(x, alg_cross['val_r2'], width, label='Validation R²', alpha=0.8)
    ax.bar(x + width, alg_cross['test_r2'], width, label='Test R²', alpha=0.8)
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('R²')
    ax.set_title('Algorithm Performance Comparison Across Sets')
    ax.set_xticks(x)
    ax.set_xticklabels(alg_cross.index, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    html_content += f"""
            <div class="plot-container">
                <img src="data:image/png;base64,{plot_data}" alt="Algorithm Comparison Across Sets">
            </div>
    """

    # Overfitting analysis
    individual_df_copy = individual_df.copy()
    individual_df_copy['overfitting'] = individual_df_copy['train_r2'] - individual_df_copy['test_r2']
    overfit_by_alg = individual_df_copy.groupby('algorithm')['overfitting'].agg(['mean', 'std']).round(4)
    overfit_by_alg = overfit_by_alg.sort_values('mean')

    html_content += """
            <h3>Overfitting Analysis</h3>
            <p>Overfitting indicator = Train R² - Test R²</p>
            <table>
                <tr>
                    <th>Algorithm</th>
                    <th>Mean Overfitting</th>
                    <th>Std Overfitting</th>
                </tr>
    """

    for alg in overfit_by_alg.index:
        html_content += f"""
                <tr>
                    <td>{alg}</td>
                    <td>{overfit_by_alg.loc[alg, 'mean']:.4f}</td>
                    <td>{overfit_by_alg.loc[alg, 'std']:.4f}</td>
                </tr>
        """

    html_content += "</table>"

    # Overfitting plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(overfit_by_alg))
    ax.barh(x, overfit_by_alg['mean'], xerr=overfit_by_alg['std'], capsize=5)
    ax.set_yticks(x)
    ax.set_yticklabels(overfit_by_alg.index)
    ax.set_xlabel('Overfitting Indicator (Train R² - Test R²)')
    ax.set_title('Overfitting Analysis by Algorithm')
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='No Overfitting')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    html_content += f"""
            <div class="plot-container">
                <img src="data:image/png;base64,{plot_data}" alt="Overfitting Analysis">
            </div>
        </div>
    """

    # Summary section
    html_content += """
        <div id="summary" class="section">
            <h2>📊 Summary & Recommendations</h2>

            <h3>Overall Best Performers</h3>
    """

    # Best overall algorithm
    best_overall = individual_df.loc[individual_df['test_r2'].idxmax()]
    best_avg = individual_df.groupby('algorithm')['test_r2'].mean().sort_values(ascending=False)

    html_content += f"""
            <div class="highlight">
                <h4>Best Overall Algorithm (Test Set)</h4>
                <ul>
                    <li><strong>Algorithm:</strong> {best_overall['algorithm']}</li>
                    <li><strong>Trait:</strong> {best_overall['trait']}</li>
                    <li><strong>Test R²:</strong> {best_overall['test_r2']:.4f}</li>
                    <li><strong>Test Pearson r:</strong> {best_overall['test_pearson_r']:.4f}</li>
                </ul>
            </div>

            <div class="success">
                <h4>Best Algorithm on Average (across all traits)</h4>
                <ul>
                    <li><strong>Algorithm:</strong> {best_avg.index[0]}</li>
                    <li><strong>Mean Test R²:</strong> {best_avg.iloc[0]:.4f}</li>
                </ul>
            </div>
    """

    # Ensemble summary
    if not ensemble_df.empty:
        best_ensemble_avg = ensemble_df.groupby('algorithm')['test_r2'].mean().sort_values(ascending=False)
        ensemble_improvement = best_ensemble_avg.iloc[0] - best_avg.iloc[0]

        html_content += f"""
            <div class="metric">
                <h4>Ensemble vs Best Individual Algorithm</h4>
                <ul>
                    <li><strong>Best Ensemble:</strong> {best_ensemble_avg.index[0]}</li>
                    <li><strong>Mean Test R²:</strong> {best_ensemble_avg.iloc[0]:.4f}</li>
                    <li><strong>Improvement:</strong> {ensemble_improvement:.4f}</li>
        """

        if ensemble_improvement > 0:
            html_content += "<li><span style='color: green;'>✅ Ensemble performs better on average</span></li>"
        else:
            html_content += "<li><span style='color: orange;'>⚠️ Individual algorithm performs better on average</span></li>"

        html_content += "</ul></div>"

    # Recommendations
    html_content += """
            <div class="metric">
                <h3>💡 Recommendations</h3>
                <ul>
                    <li>Review the detailed tables above to find the best algorithm for each trait</li>
                    <li>Consider ensemble methods if they show consistent improvement</li>
                    <li>Check overfitting indicators to ensure model generalization</li>
                    <li>Use validation set performance for model selection</li>
                    <li>Test set performance provides final unbiased evaluation</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """

    # Save HTML report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✅ HTML train-validate-test report generated: {output_file}")
    return output_file

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Execute train-validate-test analysis and generate HTML report')
    parser.add_argument('--results_file', default='../train_validate_array/combined_train_validate_results_20251210_154526.csv', help='Path to results CSV file')
    parser.add_argument('--output_file', default='../reports/Learning_Benchmarking_report.html', help='Path to output HTML file')

    args = parser.parse_args()

    # Create reports directory if it doesn't exist
    Path(args.output_file).parent.mkdir(exist_ok=True)

    create_html_report(args.results_file, args.output_file)

if __name__ == '__main__':
    main()
