# G-Matrix (Genomic Relationship Matrix) Integration

The G-matrix (Genomic Relationship Matrix) is automatically calculated and reported in **Step 1** (Train-Validate) and **Step 3** (Prediction) phases.

---

## What is G-Matrix?

The Genomic Relationship Matrix (G-matrix) quantifies the genetic relatedness between animals based on their genomic data. It's a fundamental component in genomic prediction methods like GBLUP.

**Formula (VanRaden method):**
```
G = (X - P)(X - P)' / sum(2p(1-p))
```
Where:
- `X` = Genotype matrix (animals × markers)
- `P` = Expected genotype values (2p, where p = allele frequencies)
- `p` = Allele frequencies

---

## Automatic Calculation

### Step 1: Train-Validate Phase

**When:** Automatically during train-validate preparation

**Location:** `02_phase1_train_validate.sh` calls `02a_phase1_train_validate_array.py` with `--calculate_gmatrix` flag

**Output:**
- `train_validate_array/gmatrix/Gmatrix.csv` - Full G-matrix
- `train_validate_array/gmatrix/gmatrix_metadata.json` - Calculation metadata
- `train_validate_array/gmatrix/allele_frequencies.csv` - Allele frequencies

**Report:** Summary printed to console with:
- Method used (VanRaden)
- Number of animals and markers
- Computation time
- G-matrix statistics (diagonal/off-diagonal ranges)

### Step 3: Prediction Phase

**When:** Automatically during prediction preparation

**Location:** `04a_phase2_predict_array.py` → `prepare_prediction_jobs()` function

**Output:**
- `Phase2_Deployment_Prediction/prediction/gmatrix/Gmatrix.csv` - Full G-matrix for prediction animals
- `Phase2_Deployment_Prediction/prediction/gmatrix/gmatrix_metadata.json` - Calculation metadata
- `Phase2_Deployment_Prediction/prediction/gmatrix/allele_frequencies.csv` - Allele frequencies

**Report:** Summary printed to console with same statistics as Step 1

---

## G-Matrix Files

### Gmatrix.csv
- **Format:** CSV file (animals × animals)
- **Values:** Genomic relationship coefficients (typically -1 to 2)
- **Diagonal:** Self-relationships (typically ~1.0)
- **Off-diagonal:** Relationships between pairs of animals

### gmatrix_metadata.json
Contains calculation metadata:
```json
{
  "method": "vanRaden",
  "n_animals": 1285,
  "n_markers": 26503,
  "computation_time": 12.34,
  "standardized": true,
  "scaling_factor": 12345.67,
  "allele_frequencies": [...]
}
```

### allele_frequencies.csv
- **Format:** CSV with columns: `marker`, `allele_frequency`
- **Purpose:** Allele frequency for each marker (used in G-matrix calculation)

---

## Usage in Analysis

The G-matrix can be used for:

1. **Relatedness Analysis:** Identify closely related animals
2. **Population Structure:** Understand genetic relationships
3. **GBLUP Methods:** Required for genomic BLUP predictions
4. **Quality Control:** Check for unexpected relationships or errors

---

## Technical Details

**Method:** VanRaden (2008) - Standard method for genomic relationship matrices

**Standardization:** Yes (diagonal elements normalized to ~1.0)

**Missing Values:** Handled by replacing with column means

**Computation:** Efficient matrix operations using NumPy

---

## Example Output

```
====================================================================
G-MATRIX CALCULATION SUMMARY (Train-Validate Phase)
====================================================================
Method: vanRaden
Animals: 1285
Markers: 26503
Computation time: 12.34 seconds
Standardized: True
Scaling factor: 12345.678901
G-matrix shape: (1285, 1285)
Diagonal range: [0.9500, 1.0500]
G-matrix saved to: train_validate_array/gmatrix/Gmatrix.csv
====================================================================
```

---

## Notes

- G-matrix calculation is **automatic** - no user action required
- Calculated separately for train-validate and prediction phases
- Both use the same VanRaden method for consistency
- Files are saved for later analysis and visualization

