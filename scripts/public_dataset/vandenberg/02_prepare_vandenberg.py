#!/usr/bin/env python3
"""
Convert Van den Berg et al. dataset to BreedAI format.

Input format: Tab-separated files
- Genotypes_26503SNPs.txt: AnimalID | SNP1 | SNP2 | ... | SNP26503
- Phenotypes files: AnimalID | Trait1 | Trait2 | ...

Output format: 
- Geno.csv: animals × SNPs (index: Animal IDs, columns: SNP_1, SNP_2, ...)
- Pheno.csv: animals × traits (index: Animal IDs, columns: Trait names)
"""

import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
import sys
import json

# Import QC functions from validation module
try:
    import sys
    from pathlib import Path
    # Add validation scripts to path
    validation_path = Path(__file__).parent.parent.parent / "validation"
    if str(validation_path) not in sys.path:
        sys.path.insert(0, str(validation_path))
    from quality_control import QualityControl
    QC_AVAILABLE = True
except ImportError:
    QC_AVAILABLE = False
    print("Warning: quality_control.py not found. QC will be skipped.")

try:
    import pyplink
    PYPLINK_AVAILABLE = True
except ImportError:
    PYPLINK_AVAILABLE = False

try:
    from pysnptools.snpreader import Bed
    PYSNPTOOLS_AVAILABLE = True
except ImportError:
    PYSNPTOOLS_AVAILABLE = False


class VandenbergPreparer:
    """Convert Van den Berg dataset to BreedAI format"""
    
    def __init__(self, input_dir, output_dir, run_qc=True):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_qc = run_qc and QC_AVAILABLE
        self.setup_logging()
        
        if self.run_qc:
            self.qc = QualityControl(logger=self.logger)
        else:
            self.qc = None
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'prepare_vandenberg.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_plink_data(self, plink_prefix):
        """Load PLINK format data (.bed, .bim, .fam)"""
        self.logger.info(f"Loading PLINK data from {plink_prefix}")
        
        if PYSNPTOOLS_AVAILABLE:
            # Use pysnptools (recommended)
            try:
                bed = Bed(plink_prefix, count_A1=True)
                geno = bed.read().val
                
                # Get sample IDs from .fam file
                fam_file = f"{plink_prefix}.fam"
                if Path(fam_file).exists():
                    fam_df = pd.read_csv(fam_file, sep='\s+', header=None,
                                        names=['FID', 'IID', 'PID', 'MID', 'SEX', 'PHENO'])
                    sample_ids = fam_df['IID'].values
                else:
                    sample_ids = [f"Sample_{i}" for i in range(geno.shape[0])]
                
                # Get SNP IDs from .bim file
                bim_file = f"{plink_prefix}.bim"
                if Path(bim_file).exists():
                    bim_df = pd.read_csv(bim_file, sep='\s+', header=None,
                                       names=['CHR', 'SNP', 'CM', 'BP', 'A1', 'A2'])
                    snp_ids = bim_df['SNP'].values
                else:
                    snp_ids = [f"SNP_{i}" for i in range(geno.shape[1])]
                
                self.logger.info(f"Loaded genotype matrix: {geno.shape[0]} samples × {geno.shape[1]} SNPs")
                return pd.DataFrame(geno, index=sample_ids, columns=snp_ids)
            
            except Exception as e:
                self.logger.error(f"Failed to load with pysnptools: {e}")
                return None
        
        elif PYPLINK_AVAILABLE:
            # Use pyplink
            try:
                with pyplink.PyPlink(plink_prefix) as bed:
                    geno_list = []
                    sample_ids = []
                    
                    for i, (bim, fam, geno_array) in enumerate(bed):
                        if i == 0:
                            sample_ids = fam['iid'].values
                            snp_ids = [bim['snp']]
                        else:
                            snp_ids.append(bim['snp'])
                        
                        geno_list.append(geno_array)
                    
                    geno = np.column_stack(geno_list)
                    self.logger.info(f"Loaded genotype matrix: {geno.shape[0]} samples × {geno.shape[1]} SNPs")
                    return pd.DataFrame(geno, index=sample_ids, columns=snp_ids)
            
            except Exception as e:
                self.logger.error(f"Failed to load with pyplink: {e}")
                return None
        
        else:
            self.logger.error("No PLINK reader available. Install pysnptools or pyplink.")
            return None
    
    def load_vandenberg_geno(self, geno_file):
        """
        Load Van den Berg genotype file (tab-separated, no header)
        Format: AnimalID | SNP_string (concatenated 0/1/2 values)
        The SNP_string needs to be split character by character
        """
        self.logger.info(f"Loading genotype data from: {geno_file}")
        
        try:
            # Read space-separated file, no header
            # Format: AnimalID SNP_string (space-separated)
            raw_df = pd.read_csv(geno_file, sep='\s+', header=None, engine='python')
            
            # First column is animal ID - convert to integer for proper numeric comparison
            animal_ids = pd.to_numeric(raw_df.iloc[:, 0], errors='coerce')
            
            # Second column contains concatenated SNP values as a string
            snp_strings = raw_df.iloc[:, 1].astype(str)
            
            # Split each string into individual characters (SNPs)
            # Each character is a genotype value (0, 1, or 2)
            snp_arrays = []
            for snp_str in snp_strings:
                # Convert string to list of integers
                snp_array = [int(char) for char in snp_str]
                snp_arrays.append(snp_array)
            
            # Convert to numpy array
            snp_matrix = np.array(snp_arrays)
            
            # Create SNP column names
            n_snps = snp_matrix.shape[1]
            snp_columns = [f'SNP_{i+1}' for i in range(n_snps)]
            
            # Create DataFrame with animal IDs as index (as integers for proper sorting)
            geno_df = pd.DataFrame(snp_matrix, index=animal_ids, columns=snp_columns)
            
            self.logger.info(f"Loaded genotype matrix: {geno_df.shape[0]} animals × {geno_df.shape[1]} SNPs")
            self.logger.info(f"Animal ID range: {int(geno_df.index.min())} to {int(geno_df.index.max())}")
            self.logger.info(f"Genotype value range: {geno_df.values.min()} to {geno_df.values.max()}")
            
            return geno_df
            
        except Exception as e:
            self.logger.error(f"Failed to load genotype data: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def load_vandenberg_pheno(self, pheno_file):
        """
        Load Van den Berg phenotype file (tab-separated, no header)
        Format: AnimalID | Trait1 | Trait2 | ... | TraitN
        """
        self.logger.info(f"Loading phenotype data from: {pheno_file}")
        
        try:
            # Read space-separated file, no header
            raw_df = pd.read_csv(pheno_file, sep='\s+', header=None, engine='python')
            
            # First column is animal ID - convert to integer for proper numeric comparison
            animal_ids = pd.to_numeric(raw_df.iloc[:, 0], errors='coerce')
            
            # Remaining columns are traits
            trait_data = raw_df.iloc[:, 1:].values.astype(float)
            
            # Create trait column names
            n_traits = trait_data.shape[1]
            if n_traits == 1:
                trait_columns = ['TBV']  # True Breeding Value
            else:
                trait_columns = [f'Trait_{i+1}' for i in range(n_traits)]
            
            # Create DataFrame with animal IDs as index
            pheno_df = pd.DataFrame(trait_data, index=animal_ids, columns=trait_columns)
            
            self.logger.info(f"Loaded phenotype matrix: {pheno_df.shape[0]} animals × {pheno_df.shape[1]} traits")
            self.logger.info(f"Traits: {list(pheno_df.columns)}")
            self.logger.info(f"Phenotype value range: {pheno_df.values.min():.4f} to {pheno_df.values.max():.4f}")
            
            return pheno_df
            
        except Exception as e:
            self.logger.error(f"Failed to load phenotype data: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def load_pheno_data(self, pheno_file, trait_columns=None):
        """Load phenotypic data"""
        self.logger.info(f"Loading phenotypic data from: {pheno_file}")
        
        try:
            pheno_df = pd.read_csv(pheno_file, index_col=0)
            
            # If trait_columns specified, select only those columns
            if trait_columns:
                available_traits = [col for col in trait_columns if col in pheno_df.columns]
                if len(available_traits) < len(trait_columns):
                    missing = set(trait_columns) - set(available_traits)
                    self.logger.warning(f"Missing traits: {missing}")
                pheno_df = pheno_df[available_traits]
            
            self.logger.info(f"Loaded phenotypic data: {pheno_df.shape[0]} samples × {pheno_df.shape[1]} traits")
            return pheno_df
        except Exception as e:
            self.logger.error(f"Failed to load phenotypic data: {e}")
            return None
    
    def align_samples(self, geno_df, pheno_df):
        """Align genotype and phenotype data by sample IDs"""
        self.logger.info("Aligning genotype and phenotype data")
        
        # Find common samples
        common_samples = geno_df.index.intersection(pheno_df.index)
        self.logger.info(f"Found {len(common_samples)} common samples")
        
        if len(common_samples) == 0:
            self.logger.error("No common samples found between genotype and phenotype data!")
            return None, None
        
        # Align data
        geno_aligned = geno_df.loc[common_samples]
        pheno_aligned = pheno_df.loc[common_samples]
        
        self.logger.info(f"Aligned data: {geno_aligned.shape[0]} samples")
        return geno_aligned, pheno_aligned
    
    def convert_to_breedai_format(self, geno_df, pheno_df, output_suffix=''):
        """
        Convert data to BreedAI format and save
        
        Args:
            geno_df: Genotype DataFrame
            pheno_df: Phenotype DataFrame
            output_suffix: Suffix for output filenames (e.g., '_QTL300_rg8')
        """
        self.logger.info("=" * 60)
        self.logger.info("Converting to BreedAI format")
        self.logger.info("=" * 60)
        
        # Align samples
        geno_aligned, pheno_aligned = self.align_samples(geno_df, pheno_df)
        
        if geno_aligned is None or pheno_aligned is None:
            return False
        
        # Run quality control
        if self.run_qc:
            self.logger.info("\nRunning quality control...")
            qc_report_path = self.output_dir / f'qc_report{output_suffix}.json'
            self.qc.generate_qc_report(geno_aligned, pheno_aligned, qc_report_path)
        
        # Save Geno.csv (animals × SNPs)
        geno_output = self.output_dir / f'Geno{output_suffix}.csv'
        geno_aligned.to_csv(geno_output)
        self.logger.info(f"\nSaved genotype data: {geno_output}")
        self.logger.info(f"  Shape: {geno_aligned.shape[0]} animals × {geno_aligned.shape[1]} SNPs")
        
        # Save Pheno.csv (animals × traits)
        pheno_output = self.output_dir / f'Pheno{output_suffix}.csv'
        pheno_aligned.to_csv(pheno_output)
        self.logger.info(f"Saved phenotype data: {pheno_output}")
        self.logger.info(f"  Shape: {pheno_aligned.shape[0]} animals × {pheno_aligned.shape[1]} traits")
        self.logger.info(f"  Traits: {list(pheno_aligned.columns)}")
        
        # Save metadata
        metadata = {
            'n_animals': int(geno_aligned.shape[0]),
            'n_snps': int(geno_aligned.shape[1]),
            'n_traits': int(pheno_aligned.shape[1]),
            'traits': list(pheno_aligned.columns),
            'source': 'Van den Berg et al. (Dryad DOI: 10.5061/dryad.rq80k)',
            'genotype_file': str(geno_output),
            'phenotype_file': str(pheno_output),
        }
        
        metadata_file = self.output_dir / f'metadata{output_suffix}.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f"Saved metadata: {metadata_file}")
        
        return True
    
    def prepare_vandenberg_file(self, geno_file, pheno_file, output_suffix=''):
        """
        Prepare dataset from Van den Berg format files
        
        Args:
            geno_file: Path to Genotypes_26503SNPs.txt
            pheno_file: Path to phenotype file
            output_suffix: Suffix for output files (e.g., '_QTL300_rg8')
        """
        # Load data
        geno_df = self.load_vandenberg_geno(geno_file)
        if geno_df is None:
            return False
        
        pheno_df = self.load_vandenberg_pheno(pheno_file)
        if pheno_df is None:
            return False
        
        # Convert and save
        return self.convert_to_breedai_format(geno_df, pheno_df, output_suffix)
    
    def prepare_from_plink(self, plink_prefix, pheno_file, trait_columns=None):
        """Prepare dataset from PLINK format"""
        geno_df = self.load_plink_data(plink_prefix)
        if geno_df is None:
            return False
        
        pheno_df = self.load_pheno_data(pheno_file, trait_columns)
        if pheno_df is None:
            return False
        
        return self.convert_to_breedai_format(geno_df, pheno_df)
    
    def prepare_from_csv(self, geno_file, pheno_file, trait_columns=None):
        """Prepare dataset from CSV format"""
        geno_df = self.load_csv_geno(geno_file)
        if geno_df is None:
            return False
        
        pheno_df = self.load_pheno_data(pheno_file, trait_columns)
        if pheno_df is None:
            return False
        
        return self.convert_to_breedai_format(geno_df, pheno_df)


def main():
    parser = argparse.ArgumentParser(
        description='Convert Van den Berg dataset to BreedAI format (Geno.csv, Pheno.csv)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single phenotype file
  python 02_prepare_vandenberg.py \\
      --geno-file cattle_dataset/raw/vandenberg/Genotypes_26503SNPs.txt \\
      --pheno-file cattle_dataset/raw/vandenberg/Phenotypes_GenCor_0.8/Phenotypes_replicate_1.txt \\
      --output-dir cattle_dataset/processed/vandenberg_QTL300_rg8 \\
      --output-suffix _QTL300_rg8

  # Convert with automatic QC
  python 02_prepare_vandenberg.py \\
      --geno-file cattle_dataset/raw/vandenberg/Genotypes_26503SNPs.txt \\
      --pheno-file cattle_dataset/raw/vandenberg/Phenotypes_GenCor_0.8/Phenotypes_replicate_1.txt \\
      --output-dir cattle_dataset/processed/vandenberg_QTL300_rg8
        """
    )
    parser.add_argument(
        '--geno-file',
        type=str,
        required=True,
        help='Genotype file (Genotypes_26503SNPs.txt)'
    )
    parser.add_argument(
        '--pheno-file',
        type=str,
        required=True,
        help='Phenotype file (e.g., Phenotypes_QTL300_cor0.8.txt)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='cattle_dataset/processed/vandenberg',
        help='Output directory for BreedAI format files'
    )
    parser.add_argument(
        '--output-suffix',
        type=str,
        default='',
        help='Suffix for output files (e.g., _QTL300_rg8)'
    )
    parser.add_argument(
        '--no-qc',
        action='store_true',
        help='Skip quality control checks'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    geno_path = Path(args.geno_file)
    pheno_path = Path(args.pheno_file)
    
    if not geno_path.exists():
        print(f"ERROR: Genotype file not found: {geno_path}")
        sys.exit(1)
    
    if not pheno_path.exists():
        print(f"ERROR: Phenotype file not found: {pheno_path}")
        sys.exit(1)
    
    # Create preparer
    preparer = VandenbergPreparer(
        input_dir=geno_path.parent,
        output_dir=args.output_dir,
        run_qc=not args.no_qc
    )
    
    # Prepare dataset
    success = preparer.prepare_vandenberg_file(
        geno_file=str(geno_path),
        pheno_file=str(pheno_path),
        output_suffix=args.output_suffix
    )
    
    if success:
        print("\n" + "=" * 60)
        print("✓ Dataset preparation completed successfully!")
        print("=" * 60)
        print(f"  Output directory: {args.output_dir}")
        print(f"  Geno file: Geno{args.output_suffix}.csv")
        print(f"  Pheno file: Pheno{args.output_suffix}.csv")
        if not args.no_qc:
            print(f"  QC report: qc_report{args.output_suffix}.json")
    else:
        print("\n✗ Dataset preparation failed. Check logs for details.")
        sys.exit(1)


if __name__ == '__main__':
    main()


