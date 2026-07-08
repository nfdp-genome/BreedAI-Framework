# Quick Start: Download Van den Berg Dataset

## Option 1: Manual URL Entry (Recommended)

Since Dryad loads file links dynamically, you need to manually extract the URLs:

### Step 1: Get File URLs

1. Open your browser and visit:
   ```
   https://datadryad.org/stash/dataset/doi:10.5061/dryad.rq80k
   ```

2. For each file listed below, right-click the download link and select "Copy link address":
   - Genotypes_26503SNPs.txt
   - Phenotypes_QTL3_cor0.4.txt
   - Phenotypes_QTL3_cor0.8.txt
   - Phenotypes_QTL3_cor1.0.txt
   - Phenotypes_QTL30_cor0.4.txt
   - Phenotypes_QTL30_cor0.8.txt
   - Phenotypes_QTL30_cor1.0.txt
   - Phenotypes_QTL300_cor0.4.txt
   - Phenotypes_QTL300_cor0.8.txt
   - Phenotypes_QTL300_cor1.0.txt
   - Phenotypes_QTL3000_cor0.4.txt
   - Phenotypes_QTL3000_cor0.8.txt
   - Phenotypes_QTL3000_cor1.0.txt

### Step 2: Create URL Configuration File

```bash
cd <PROJECT_DIR>
cp scripts/public_dataset/vandenberg/vandenberg_urls.json.template scripts/public_dataset/vandenberg/vandenberg_urls.json
```

Then edit `scripts/public_dataset/vandenberg/vandenberg_urls.json` and replace `REPLACE_WITH_ACTUAL_FILE_ID` with the actual file IDs from the URLs you copied.

### Step 3: Run Download

```bash
cd <PROJECT_DIR>
./scripts/public_dataset/vandenberg/01_run_download.sh
```

Or directly:

```bash
cd <PROJECT_DIR>
python3 scripts/public_dataset/vandenberg/01_download_vandenberg.py \
    --dataset vandenberg \
    --output-dir cattle_dataset/raw \
    --url-config scripts/public_dataset/vandenberg/vandenberg_urls.json
```

## Option 2: Direct Command (If URLs are known)

If you already have the URLs, you can run:

```bash
cd <PROJECT_DIR>
python3 scripts/public_dataset/vandenberg/01_download_vandenberg.py \
    --dataset vandenberg \
    --output-dir cattle_dataset/raw \
    --url-config scripts/public_dataset/vandenberg/vandenberg_urls.json
```

## Output Location

Files will be downloaded to:
```
cattle_dataset/raw/vandenberg/
```

## Features

- ✅ Progress bars with `tqdm`
- ✅ Automatic resume (skips existing files)
- ✅ Support for compressed archives
- ✅ Comprehensive logging to `cattle_dataset/raw/download.log`


