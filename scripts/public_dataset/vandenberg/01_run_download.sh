#!/bin/bash
# Command to download Van den Berg dataset from Dryad
# 
# NOTE: Dryad requires manual URL extraction. Follow these steps:
# 1. Visit: https://datadryad.org/stash/dataset/doi:10.5061/dryad.rq80k
# 2. Right-click each file download link and "Copy link address"
# 3. Create vandenberg_urls.json with the URLs (see vandenberg_urls.json.template)
# 4. Then run this script

cd <PROJECT_DIR>

# Check if URL config exists
if [ ! -f "scripts/public_dataset/vandenberg/vandenberg_urls.json" ]; then
    echo "=========================================="
    echo "ERROR: vandenberg_urls.json not found!"
    echo "=========================================="
    echo ""
    echo "Please create the URL configuration file first:"
    echo ""
    echo "1. Visit: https://datadryad.org/stash/dataset/doi:10.5061/dryad.rq80k"
    echo "2. For each file, right-click the download link and copy the URL"
    echo "3. Create scripts/public_dataset/vandenberg/vandenberg_urls.json with format:"
    echo '   {"Genotypes_26503SNPs.txt": "https://datadryad.org/stash/downloads/file_stream/XXXXX", ...}'
    echo ""
    echo "You can use the template:"
    echo "  cp scripts/public_dataset/vandenberg/vandenberg_urls.json.template scripts/public_dataset/vandenberg/vandenberg_urls.json"
    echo "  # Then edit it with the actual URLs"
    echo ""
    exit 1
fi

# Run the download script
echo "Starting download..."
python3 scripts/public_dataset/vandenberg/01_download_vandenberg.py \
    --dataset vandenberg \
    --output-dir data/public_datasets/raw \
    --url-config scripts/public_dataset/vandenberg/vandenberg_urls.json

echo ""
echo "Download complete! Files saved to: data/public_datasets/raw/vandenberg/"


