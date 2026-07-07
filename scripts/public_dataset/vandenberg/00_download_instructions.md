# Download Instructions for Van den Berg Dataset

## Overview

The download script has been created and is ready to use. However, Dryad's API doesn't automatically provide file download URLs, so you'll need to manually extract them from the dataset page.

## Quick Start

### Step 1: Get File URLs from Dryad

1. Visit the dataset page: https://datadryad.org/stash/dataset/doi:10.5061/dryad.rq80k

2. For each file you need, right-click the download link and "Copy link address"

3. The files you need are:
   - `Genotypes_26503SNPs.txt`
   - `Phenotypes_QTL3_cor0.4.txt`
   - `Phenotypes_QTL3_cor0.8.txt`
   - `Phenotypes_QTL3_cor1.0.txt`
   - `Phenotypes_QTL30_cor0.4.txt`
   - `Phenotypes_QTL30_cor0.8.txt`
   - `Phenotypes_QTL30_cor1.0.txt`
   - `Phenotypes_QTL300_cor0.4.txt`
   - `Phenotypes_QTL300_cor0.8.txt`
   - `Phenotypes_QTL300_cor1.0.txt`
   - `Phenotypes_QTL3000_cor0.4.txt`
   - `Phenotypes_QTL3000_cor0.8.txt`
   - `Phenotypes_QTL3000_cor1.0.txt`

### Step 2: Create URL Configuration File

Create a file `vandenberg_urls.json` with the following structure:

```json
{
  "Genotypes_26503SNPs.txt": "https://datadryad.org/stash/downloads/file_stream/XXXXX",
  "Phenotypes_QTL3_cor0.4.txt": "https://datadryad.org/stash/downloads/file_stream/XXXXX",
  "Phenotypes_QTL3_cor0.8.txt": "https://datadryad.org/stash/downloads/file_stream/XXXXX",
  "Phenotypes_QTL3_cor1.0.txt": "https://datadryad.org/stash/downloads/file_stream/XXXXX",
  "Phenotypes_QTL30_cor0.4.txt": "https://datadryad.org/stash/downloads/file_stream/XXXXX",
  "Phenotypes_QTL30_cor0.8.txt": "https://datadryad.org/stash/downloads/file_stream/XXXXX",
  "Phenotypes_QTL30_cor1.0.txt": "https://datadryad.org/stash/downloads/file_stream/XXXXX",
  "Phenotypes_QTL300_cor0.4.txt": "https://datadryad.org/stash/downloads/file_stream/XXXXX",
  "Phenotypes_QTL300_cor0.8.txt": "https://datadryad.org/stash/downloads/file_stream/XXXXX",
  "Phenotypes_QTL300_cor1.0.txt": "https://datadryad.org/stash/downloads/file_stream/XXXXX",
  "Phenotypes_QTL3000_cor0.4.txt": "https://datadryad.org/stash/downloads/file_stream/XXXXX",
  "Phenotypes_QTL3000_cor0.8.txt": "https://datadryad.org/stash/downloads/file_stream/XXXXX",
  "Phenotypes_QTL3000_cor1.0.txt": "https://datadryad.org/stash/downloads/file_stream/XXXXX"
}
```

Replace `XXXXX` with the actual file IDs from the Dryad download links.

### Step 3: Run the Download Script

```bash
cd <PROJECT_DIR>
python3 scripts/public_dataset/vandenberg/01_download_vandenberg.py \
    --dataset vandenberg \
    --output-dir data/public_datasets/raw \
    --url-config vandenberg_urls.json
```

## Alternative: Manual Download

If you prefer to download manually:

1. Visit: https://datadryad.org/stash/dataset/doi:10.5061/dryad.rq80k
2. Download all required files to `data/public_datasets/raw/vandenberg/`
3. The script will skip files that already exist

## Features

- ✅ Progress bars using `tqdm`
- ✅ Automatic checksum verification (if checksums are provided)
- ✅ Support for compressed archives (zip, tar.gz, 7z)
- ✅ Resume downloads (skips existing files)
- ✅ Comprehensive logging

## Notes

- The script uses `requests` library for downloads
- Progress bars require `tqdm` (already installed)
- For 7z archives, install `py7zr`: `pip install py7zr`
- Files are saved to `data/public_datasets/raw/vandenberg/`



