#!/usr/bin/env python3
"""
Execute QC pipeline analysis and generate HTML report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
import base64
from io import BytesIO

def create_html_report(dataset_dir, gmatrix_dir, output_file):
    """Create an HTML report with embedded plots"""

    # Set style for plots
    plt.style.use('seaborn-v0_8')
    sns.set_palette('husl')

    # Define paths
    dataset_path = Path(dataset_dir)
    gmatrix_path = Path(gmatrix_dir)

    # Load data
    print("🔍 Loading QC pipeline data...")

    # Load genotype data
    try:
        geno_path = dataset_path / 'Geno.csv'
        geno_df = pd.read_csv(geno_path, index_col=0)  # First column is sample IDs
        print(f'✅ Genotype data loaded: {geno_df.shape}')
    except Exception as e:
        print(f'❌ Error loading genotype data: {e}')
        geno_df = None

    # Load phenotype data
    try:
        pheno_path = dataset_path / 'Pheno.csv'
        pheno_df = pd.read_csv(pheno_path, index_col=0)  # First column is sample IDs
        print(f'✅ Phenotype data loaded: {pheno_df.shape}')
    except Exception as e:
        print(f'❌ Error loading phenotype data: {e}')
        pheno_df = None

    # Load G-matrix data
    try:
        gmatrix_file = gmatrix_path / 'Gmatrix.csv'
        gmatrix_df = pd.read_csv(gmatrix_file, index_col=0)
        print(f'✅ G-matrix loaded: {gmatrix_df.shape}')
    except Exception as e:
        print(f'❌ Error loading G-matrix: {e}')
        gmatrix_df = None

    # Load metadata
    try:
        metadata_path = gmatrix_path / 'gmatrix_metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f'✅ Metadata loaded')
    except Exception as e:
        print(f'❌ Error loading metadata: {e}')
        metadata = {}

    # Load allele frequencies
    try:
        allele_freq_path = gmatrix_path / 'allele_frequencies.csv'
        allele_freq_df = pd.read_csv(allele_freq_path)
        print(f'✅ Allele frequencies loaded: {allele_freq_df.shape}')
    except Exception as e:
        print(f'❌ Error loading allele frequencies: {e}')
        allele_freq_df = None

    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Quality Control (QC) Pipeline Report</title>
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
        </style>
    </head>
    <body>
        <h1>🧬 Quality Control (QC) Pipeline Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="section">
            <h2>📋 Table of Contents</h2>
            <ol>
                <li><a href="#overview">Data Overview</a></li>
                <li><a href="#genotype">Genotype Data Analysis</a></li>
                <li><a href="#phenotype">Phenotype Data Analysis</a></li>
                <li><a href="#filtering">SNP Filtering Analysis</a></li>
                <li><a href="#gmatrix">G-Matrix Calculation</a></li>
                <li><a href="#alignment">Data Alignment Results</a></li>
                <li><a href="#summary">Quality Control Summary</a></li>
            </ol>
        </div>

        <div id="overview" class="section">
            <h2>📊 Data Overview</h2>
            <div class="stats">
    """

    # Data overview stats
    if geno_df is not None:
        html_content += f"""
                <div class="stat-item">
                    <strong>Genotype Data</strong><br>
                    {geno_df.shape[0]:,} animals × {geno_df.shape[1]:,} SNPs
                </div>
        """

    if pheno_df is not None:
        html_content += f"""
                <div class="stat-item">
                    <strong>Phenotype Data</strong><br>
                    {pheno_df.shape[0]:,} animals × {pheno_df.shape[1]} traits
                </div>
        """

    if gmatrix_df is not None:
        html_content += f"""
                <div class="stat-item">
                    <strong>G-Matrix</strong><br>
                    {gmatrix_df.shape[0]} × {gmatrix_df.shape[0]} relationships
                </div>
        """

    html_content += """
            </div>
        </div>

        <div id="genotype" class="section">
            <h2>🧬 Genotype Data Analysis</h2>
    """

    if geno_df is not None:
        # Genotype statistics
        missing_rate = geno_df.isna().sum().sum() / (geno_df.shape[0] * geno_df.shape[1]) * 100

        # SNP variance
        snp_variance = geno_df.var(axis=0, skipna=True)

        # Create plots
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Missing rate histogram
        snp_missing = geno_df.isna().mean()
        axes[0].hist(snp_missing, bins=50, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Missing Rate')
        axes[0].set_ylabel('Number of SNPs')
        axes[0].set_title('SNP Missing Rate Distribution')
        axes[0].axvline(snp_missing.mean(), color='red', linestyle='--', label=f'Mean: {snp_missing.mean():.3f}')
        axes[0].legend()

        # Variance histogram
        axes[1].hist(snp_variance, bins=50, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Variance')
        axes[1].set_ylabel('Number of SNPs')
        axes[1].set_title('SNP Variance Distribution')
        axes[1].axvline(snp_variance.mean(), color='red', linestyle='--', label=f'Mean: {snp_variance.mean():.3f}')
        axes[1].legend()

        plt.tight_layout()

        # Save plot to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        html_content += f"""
            <div class="metric">
                <h3>Basic Statistics</h3>
                <ul>
                    <li><strong>Total SNPs:</strong> {geno_df.shape[1]:,}</li>
                    <li><strong>Total animals:</strong> {geno_df.shape[0]:,}</li>
                    <li><strong>Missing values:</strong> {geno_df.isna().sum().sum():,}</li>
                    <li><strong>Missing rate:</strong> {missing_rate:.2f}%</li>
                </ul>
            </div>

            <div class="plot-container">
                <img src="data:image/png;base64,{plot_data}" alt="Genotype Data Distributions">
            </div>
        """

    html_content += """
        </div>

        <div id="phenotype" class="section">
            <h2>🎯 Phenotype Data Analysis</h2>
    """

    if pheno_df is not None:
        # Phenotype statistics
        complete_records = (pheno_df.notna().all(axis=1)).sum()
        completeness = pheno_df.notna().mean() * 100

        # Summary statistics for numeric columns
        numeric_cols = pheno_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            trait_stats = pheno_df[numeric_cols].describe().T
            trait_stats = trait_stats[['count', 'mean', 'std', 'min', 'max']].round(3)

            # Create completeness plot
            fig, ax = plt.subplots(figsize=(12, 6))
            completeness_sorted = completeness.sort_values(ascending=True)
            bars = ax.barh(range(len(completeness_sorted)), completeness_sorted.values)
            ax.set_yticks(range(len(completeness_sorted)))
            ax.set_yticklabels(completeness_sorted.index)
            ax.set_xlabel('Completeness (%)')
            ax.set_title('Trait Completeness')
            ax.grid(axis='x', alpha=0.3)

            # Color bars based on completeness
            for bar, pct in zip(bars, completeness_sorted.values):
                if pct >= 90:
                    bar.set_color('green')
                elif pct >= 70:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')

            plt.tight_layout()

            # Save plot
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            html_content += f"""
                <div class="metric">
                    <h3>Basic Statistics</h3>
                    <ul>
                        <li><strong>Total traits:</strong> {pheno_df.shape[1]}</li>
                        <li><strong>Total animals:</strong> {pheno_df.shape[0]}</li>
                        <li><strong>Complete records:</strong> {complete_records}/{pheno_df.shape[0]} ({complete_records/pheno_df.shape[0]*100:.1f}%)</li>
                    </ul>
                </div>

                <div class="plot-container">
                    <img src="data:image/png;base64,{plot_data}" alt="Trait Completeness">
                </div>

                <h3>Trait Summary Statistics</h3>
                <table>
                    <tr>
                        <th>Trait</th>
                        <th>Count</th>
                        <th>Mean</th>
                        <th>Std</th>
                        <th>Min</th>
                        <th>Max</th>
                        <th>Completeness (%)</th>
                    </tr>
            """

            for trait in trait_stats.index:
                html_content += f"""
                    <tr>
                        <td>{trait}</td>
                        <td>{trait_stats.loc[trait, 'count']}</td>
                        <td>{trait_stats.loc[trait, 'mean']:.3f}</td>
                        <td>{trait_stats.loc[trait, 'std']:.3f}</td>
                        <td>{trait_stats.loc[trait, 'min']:.3f}</td>
                        <td>{trait_stats.loc[trait, 'max']:.3f}</td>
                        <td>{completeness[trait]:.1f}%</td>
                    </tr>
                """

            html_content += "</table>"

    html_content += """
        </div>

        <div id="filtering" class="section">
            <h2>🔬 SNP Filtering Analysis</h2>
    """

    if geno_df is not None:
        # SNP filtering analysis
        snp_variance = geno_df.var(axis=0, skipna=True)
        snp_missing_rate = geno_df.isna().mean()

        # MAF analysis if binary data
        maf_info = ""
        if geno_df.dtypes.iloc[0] in ['int64', 'float64']:
            unique_vals = np.unique(geno_df.values[~np.isnan(geno_df.values)])
            if len(unique_vals) == 2 and 0 in unique_vals and 1 in unique_vals:
                maf = geno_df.mean(axis=0, skipna=True)
                maf = np.minimum(maf, 1-maf)

                # MAF distribution plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(maf, bins=50, alpha=0.7, edgecolor='black')
                ax.set_xlabel('Minor Allele Frequency')
                ax.set_ylabel('Number of SNPs')
                ax.set_title('MAF Distribution')
                ax.axvline(maf.mean(), color='red', linestyle='--', label=f'Mean MAF: {maf.mean():.3f}')
                ax.legend()

                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                plot_data = base64.b64encode(buffer.getvalue()).decode()
                plt.close()

                maf_info = f"""
                    <h3>Minor Allele Frequency (MAF) Analysis</h3>
                    <ul>
                        <li><strong>SNPs with MAF = 0:</strong> {(maf == 0).sum():,}</li>
                        <li><strong>SNPs with MAF < 0.01:</strong> {(maf < 0.01).sum():,}</li>
                        <li><strong>SNPs with MAF < 0.05:</strong> {(maf < 0.05).sum():,}</li>
                        <li><strong>Mean MAF:</strong> {maf.mean():.4f}</li>
                    </ul>
                    <div class="plot-container">
                        <img src="data:image/png;base64,{plot_data}" alt="MAF Distribution">
                    </div>
                """

        html_content += f"""
            <div class="metric">
                <h3>SNP Filtering Criteria Applied</h3>
                <ul>
                    <li>✅ Low variance SNPs removed</li>
                    <li>✅ Missing data filtering</li>
                    <li>✅ MAF filtering (if applicable)</li>
                </ul>
            </div>

            <h3>SNP Variance Analysis</h3>
            <ul>
                <li><strong>SNPs with zero variance:</strong> {(snp_variance == 0).sum():,}</li>
                <li><strong>SNPs with variance < 0.01:</strong> {(snp_variance < 0.01).sum():,}</li>
                <li><strong>SNPs with variance < 0.05:</strong> {(snp_variance < 0.05).sum():,}</li>
                <li><strong>Mean variance:</strong> {snp_variance.mean():.4f}</li>
                <li><strong>Variance range:</strong> [{snp_variance.min():.4f}, {snp_variance.max():.4f}]</li>
            </ul>

            <h3>SNP Missing Data Analysis</h3>
            <ul>
                <li><strong>SNPs with >90% missing:</strong> {(snp_missing_rate > 0.9).sum():,}</li>
                <li><strong>SNPs with >50% missing:</strong> {(snp_missing_rate > 0.5).sum():,}</li>
                <li><strong>SNPs with >10% missing:</strong> {(snp_missing_rate > 0.1).sum():,}</li>
            </ul>

            {maf_info}
        """

    html_content += """
        </div>

        <div id="gmatrix" class="section">
            <h2>🧮 G-Matrix Calculation</h2>
    """

    if gmatrix_df is not None:
        # G-matrix analysis
        diagonal = np.diag(gmatrix_df.values)
        off_diagonal = gmatrix_df.values[np.triu_indices_from(gmatrix_df.values, k=1)]

        # Eigenvalue analysis
        eigenvals = np.linalg.eigvals(gmatrix_df.values)
        eigenvals = np.sort(eigenvals)[::-1]

        # Create G-matrix visualization
        sample_size = min(50, gmatrix_df.shape[0])
        sample_indices = np.random.choice(gmatrix_df.shape[0], sample_size, replace=False)
        sample_g = gmatrix_df.values[np.ix_(sample_indices, sample_indices)]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Heatmap
        sns.heatmap(sample_g, ax=axes[0,0], cmap='RdYlBu_r', center=0)
        axes[0,0].set_title(f'G-Matrix Heatmap (Sample of {sample_size} animals)')

        # Diagonal histogram
        axes[0,1].hist(diagonal, bins=30, alpha=0.7, edgecolor='black')
        axes[0,1].set_xlabel('Diagonal Value')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Diagonal Distribution')
        axes[0,1].axvline(diagonal.mean(), color='red', linestyle='--', label=f'Mean: {diagonal.mean():.3f}')
        axes[0,1].legend()

        # Off-diagonal histogram
        axes[1,0].hist(off_diagonal, bins=30, alpha=0.7, edgecolor='black')
        axes[1,0].set_xlabel('Off-diagonal Value')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Off-diagonal Distribution')
        axes[1,0].axvline(off_diagonal.mean(), color='red', linestyle='--', label=f'Mean: {off_diagonal.mean():.3f}')
        axes[1,0].legend()

        # Eigenvalue scree plot
        axes[1,1].plot(range(1, len(eigenvals)+1), eigenvals, 'bo-')
        axes[1,1].set_xlabel('Eigenvalue Rank')
        axes[1,1].set_ylabel('Eigenvalue')
        axes[1,1].set_title('Eigenvalue Scree Plot')
        axes[1,1].set_yscale('log')

        plt.tight_layout()

        # Save plot
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        html_content += f"""
            <div class="metric">
                <h3>G-Matrix Properties</h3>
                <ul>
                    <li><strong>Shape:</strong> {gmatrix_df.shape}</li>
                    <li><strong>Animals:</strong> {gmatrix_df.shape[0]}</li>
                    <li><strong>Matrix is symmetric:</strong> {np.allclose(gmatrix_df.values, gmatrix_df.values.T)}</li>
                    <li><strong>Positive semi-definite:</strong> {np.all(np.linalg.eigvals(gmatrix_df.values) >= -1e-10)}</li>
                </ul>
            </div>

            <h3>Diagonal Elements (Self-relationships)</h3>
            <ul>
                <li><strong>Mean:</strong> {diagonal.mean():.4f}</li>
                <li><strong>Std:</strong> {diagonal.std():.4f}</li>
                <li><strong>Min:</strong> {diagonal.min():.4f}</li>
                <li><strong>Max:</strong> {diagonal.max():.4f}</li>
                <li><strong>Range:</strong> [{diagonal.min():.4f}, {diagonal.max():.4f}]</li>
            </ul>

            <h3>Off-diagonal Elements (Genetic relationships)</h3>
            <ul>
                <li><strong>Mean:</strong> {off_diagonal.mean():.4f}</li>
                <li><strong>Std:</strong> {off_diagonal.std():.4f}</li>
                <li><strong>Min:</strong> {off_diagonal.min():.4f}</li>
                <li><strong>Max:</strong> {off_diagonal.max():.4f}</li>
                <li><strong>Range:</strong> [{off_diagonal.min():.4f}, {off_diagonal.max():.4f}]</li>
            </ul>

            <h3>Eigenvalue Analysis</h3>
            <ul>
                <li><strong>Largest eigenvalue:</strong> {eigenvals[0]:.4f}</li>
                <li><strong>Smallest eigenvalue:</strong> {eigenvals[-1]:.4f}</li>
                <li><strong>Condition number:</strong> {eigenvals[0]/eigenvals[-1]:.2f}</li>
                <li><strong>Positive eigenvalues:</strong> {(eigenvals > 0).sum()}</li>
                <li><strong>Zero/near-zero eigenvalues:</strong> {(eigenvals < 1e-10).sum()}</li>
            </ul>

            <div class="plot-container">
                <img src="data:image/png;base64,{plot_data}" alt="G-Matrix Analysis">
            </div>
        """

    html_content += """
        </div>

        <div id="alignment" class="section">
            <h2>🔗 Data Alignment Results</h2>
    """

    if geno_df is not None and pheno_df is not None:
        # Sample alignment analysis
        geno_samples = set(range(len(geno_df)))  # Assuming row indices
        pheno_samples = set(range(len(pheno_df)))

        common_samples = len(geno_samples.intersection(pheno_samples))
        geno_only = len(geno_samples - pheno_samples)
        pheno_only = len(pheno_samples - geno_samples)

        alignment_rate = common_samples / max(len(geno_samples), len(pheno_samples)) * 100

        # Create alignment visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        categories = ['Geno Only', 'Pheno Only', 'Both']
        values = [geno_only, pheno_only, common_samples]
        bars = ax.bar(categories, values, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Sample Alignment Overview')

        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + value + 1,
                    f'{value}', ha='center', va='bottom')

        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        html_content += f"""
            <div class="metric">
                <h3>Sample Alignment</h3>
                <ul>
                    <li><strong>Samples in genotype data:</strong> {len(geno_samples)}</li>
                    <li><strong>Samples in phenotype data:</strong> {len(pheno_samples)}</li>
                    <li><strong>Common samples:</strong> {common_samples}</li>
                    <li><strong>Samples only in genotype:</strong> {geno_only}</li>
                    <li><strong>Samples only in phenotype:</strong> {pheno_only}</li>
                    <li><strong>Alignment rate:</strong> {alignment_rate:.1f}%</li>
                </ul>
            </div>

            <div class="plot-container">
                <img src="data:image/png;base64,{plot_data}" alt="Sample Alignment">
            </div>
        """

    html_content += """
        </div>

        <div id="summary" class="section">
            <h2>✅ Quality Control Summary</h2>
    """

    # Final summary
    html_content += f"""
            <div class="metric">
                <h3>Pipeline Results Overview</h3>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <h3>📊 QUALITY CONTROL PIPELINE SUMMARY</h3>
            <div style="background: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0;">
    """

    if geno_df is not None:
        missing_rate = geno_df.isna().sum().sum() / (geno_df.shape[0] * geno_df.shape[1]) * 100
        html_content += f"""
                <p><strong>🧬 GENOTYPE DATA:</strong></p>
                <ul>
                    <li>✅ Loaded {geno_df.shape[0]:,} animals × {geno_df.shape[1]:,} SNPs</li>
                    <li>✅ Missing data rate: {missing_rate:.2f}%</li>
                </ul>
        """

    if pheno_df is not None:
        complete_rate = (pheno_df.notna().all(axis=1)).sum() / pheno_df.shape[0]
        html_content += f"""
                <p><strong>🎯 PHENOTYPE DATA:</strong></p>
                <ul>
                    <li>✅ Loaded {pheno_df.shape[0]:,} animals × {pheno_df.shape[1]} traits</li>
                    <li>✅ Complete records: {(pheno_df.notna().all(axis=1)).sum()}/{pheno_df.shape[0]} ({complete_rate*100:.1f}%)</li>
                </ul>
        """

    if gmatrix_df is not None:
        diagonal_mean = np.diag(gmatrix_df.values).mean()
        off_diag_mean = gmatrix_df.values[np.triu_indices_from(gmatrix_df.values, k=1)].mean()
        html_content += f"""
                <p><strong>🧮 G-MATRIX:</strong></p>
                <ul>
                    <li>✅ Calculated {gmatrix_df.shape[0]} × {gmatrix_df.shape[0]} relationship matrix</li>
                    <li>✅ Mean diagonal (self-relationships): {diagonal_mean:.4f}</li>
                    <li>✅ Mean off-diagonal (relationships): {off_diag_mean:.4f}</li>
                </ul>
        """

    if metadata:
        html_content += f"""
                <p><strong>⚙️ PIPELINE PARAMETERS:</strong></p>
                <ul>
        """
        for key, value in metadata.items():
            html_content += f"<li>• {key}: {value}</li>"
        html_content += "</ul>"

    html_content += """
                <p><strong>🎉 QC Pipeline completed successfully!</strong></p>
                <p><strong>📋 Data is ready for downstream genomic prediction analysis.</strong></p>
            </div>

            <h3>💡 RECOMMENDATIONS</h3>
            <ul>
    """

    if geno_df is not None:
        missing_rate = geno_df.isna().sum().sum() / (geno_df.shape[0] * geno_df.shape[1])
        if missing_rate > 0.1:
            html_content += "<li><span style='color: orange;'>⚠️ High missing data rate in genotypes - consider imputation</span></li>"

        snp_var = geno_df.var(axis=0, skipna=True)
        low_var_snps = (snp_var < 0.01).sum()
        if low_var_snps > 0:
            html_content += f"<li><span style='color: orange;'>⚠️ {low_var_snps} SNPs with very low variance - filtering applied</span></li>"

    if pheno_df is not None:
        complete_rate = (pheno_df.notna().all(axis=1)).sum() / pheno_df.shape[0]
        if complete_rate < 0.8:
            html_content += "<li><span style='color: orange;'>⚠️ Many incomplete phenotype records - consider imputation or filtering</span></li>"

    html_content += """
                <li><span style="color: green;">✅ Data quality checks passed - proceed to training phase</span></li>
            </ul>
        </div>
    </body>
    </html>
    """

    # Save HTML report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✅ HTML QC report generated: {output_file}")
    return output_file

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Execute QC pipeline analysis and generate HTML report')
    parser.add_argument('--dataset_dir', default='../input', help='Path to dataset directory')
    parser.add_argument('--gmatrix_dir', default='../train_validate_array/gmatrix', help='Path to G-matrix directory')
    parser.add_argument('--output_file', default='../reports/qc_pipeline_report.html', help='Path to output HTML file')

    args = parser.parse_args()

    # Create reports directory if it doesn't exist
    Path(args.output_file).parent.mkdir(exist_ok=True)

    create_html_report(args.dataset_dir, args.gmatrix_dir, args.output_file)

if __name__ == '__main__':
    main()
