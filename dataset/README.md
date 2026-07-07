# Dataset Input Directory

This directory is where users should place their **Geno.csv** and **Pheno.csv** files to run the BreedAI pipeline.

---

## Usage

### Step 1: Place Your Data Files

Copy your genotype and phenotype files here:

```bash
cp /path/to/your/Geno.csv dataset/input/
cp /path/to/your/Pheno.csv dataset/input/
```

### Step 2: Run the Pipeline

The pipeline will automatically detect and use files from `dataset/input/`:

```bash
./scripts/start_menu.sh
# Select option 4 (Complete Pipeline)
```

---

## File Format Requirements

### Geno.csv
- **Format**: CSV file
- **Index**: Animal IDs (first column, no header)
- **Columns**: SNP_1, SNP_2, ..., SNP_N
- **Values**: 0, 1, 2 (genotype coding)
  - 0 = homozygous reference
  - 1 = heterozygous
  - 2 = homozygous alternate

**Example:**
```csv
AnimalID,SNP_1,SNP_2,SNP_3,...
1,0,1,2,...
2,1,1,0,...
3,2,0,1,...
```

### Pheno.csv
- **Format**: CSV file
- **Index**: Animal IDs (first column, must match Geno.csv)
- **Columns**: Trait names (one column per trait)
- **Values**: Phenotypic values (numeric)

**Example:**
```csv
AnimalID,Trait_1,Trait_2,Trait_3,...
1,45.2,12.3,8.9,...
2,42.1,11.8,9.2,...
3,48.5,13.1,8.5,...
```

---

## Data Location Priority

The pipeline checks for data files in this order:

1. **`dataset/input/`** ← **Primary location** (put your files here!)
2. `data/` (fallback for backward compatibility)

---

## Quick Start

```bash
# 1. Copy your data files
cp your_geno.csv dataset/input/Geno.csv
cp your_pheno.csv dataset/input/Pheno.csv

# 2. Verify files are in place
ls -lh dataset/input/

# 3. Run the pipeline
cd scripts
./start_menu.sh
# Select option 4
```

---

## Notes

- ✅ Files in `dataset/input/` take priority over `data/`
- ✅ Animal IDs must match between Geno.csv and Pheno.csv
- ✅ Missing values should be handled before running the pipeline
- ✅ The pipeline will automatically split your data (80% train+val, 20% test) if needed

---

## Troubleshooting

**Error: "Required data files not found"**
- Make sure `Geno.csv` and `Pheno.csv` are in `dataset/input/`
- Check file names are exactly `Geno.csv` and `Pheno.csv` (case-sensitive)

**Error: "Animal IDs don't match"**
- Ensure the first column (index) in both files contains the same animal IDs
- Check for extra spaces or formatting differences

**Error: "Invalid genotype values"**
- Genotype values must be 0, 1, or 2
- Check for missing values or non-numeric entries

