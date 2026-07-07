#!/usr/bin/env python3
"""
Helper script to extract file download URLs from Dryad dataset page.
This script attempts to find file download links on the Dryad dataset page.
"""

import requests
import re
import json
import sys

def extract_dryad_urls(doi):
    """Extract file URLs from Dryad dataset page"""
    url = f'https://datadryad.org/stash/dataset/doi:{doi.replace(":", "%3A")}'
    print(f"Fetching: {url}")
    
    try:
        response = requests.get(url, timeout=30, headers={
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        })
        
        if response.status_code != 200:
            print(f"Error: HTTP {response.status_code}")
            return {}
        
        text = response.text
        
        # Look for file_stream links
        file_stream_pattern = r'/stash/downloads/file_stream/([a-zA-Z0-9_-]+)'
        file_ids = set(re.findall(file_stream_pattern, text))
        
        print(f"Found {len(file_ids)} file_stream IDs")
        
        # Try to match filenames with file IDs by looking at context
        url_mappings = {}
        
        # Expected filenames
        expected_files = [
            'Genotypes_26503SNPs.txt',
            'Phenotypes_QTL3_cor0.4.txt',
            'Phenotypes_QTL3_cor0.8.txt',
            'Phenotypes_QTL3_cor1.0.txt',
            'Phenotypes_QTL30_cor0.4.txt',
            'Phenotypes_QTL30_cor0.8.txt',
            'Phenotypes_QTL30_cor1.0.txt',
            'Phenotypes_QTL300_cor0.4.txt',
            'Phenotypes_QTL300_cor0.8.txt',
            'Phenotypes_QTL300_cor1.0.txt',
            'Phenotypes_QTL3000_cor0.4.txt',
            'Phenotypes_QTL3000_cor0.8.txt',
            'Phenotypes_QTL3000_cor1.0.txt',
        ]
        
        # Look for filename patterns near file_stream links
        for filename in expected_files:
            # Remove .txt extension for matching
            name_part = filename.replace('.txt', '')
            # Create pattern to find filename near file_stream link
            pattern = rf'{re.escape(name_part)}[^<]*<[^>]*href=["\'](/stash/downloads/file_stream/[^"\']+)["\']'
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                full_url = f"https://datadryad.org{matches[0]}"
                url_mappings[filename] = full_url
                print(f"  ✓ {filename} -> {matches[0]}")
        
        # If we found file IDs but couldn't match them, list them
        if file_ids and not url_mappings:
            print("\nFound file_stream IDs but couldn't match to filenames:")
            for i, file_id in enumerate(list(file_ids)[:15], 1):
                print(f"  {i}. https://datadryad.org/stash/downloads/file_stream/{file_id}")
            print("\nYou may need to manually match these to filenames.")
        
        return url_mappings
        
    except Exception as e:
        print(f"Error: {e}")
        return {}

if __name__ == '__main__':
    doi = '10.5061/dryad.rq80k'
    print(f"Extracting URLs for DOI: {doi}\n")
    
    url_mappings = extract_dryad_urls(doi)
    
    if url_mappings:
        output_file = 'vandenberg_urls.json'
        with open(output_file, 'w') as f:
            json.dump(url_mappings, f, indent=2)
        print(f"\n✓ Saved {len(url_mappings)} URL mappings to {output_file}")
        print(f"\nYou can now run:")
        print(f"  python3 scripts/public_dataset/vandenberg/01_download_vandenberg.py --dataset vandenberg --url-config {output_file}")
    else:
        print("\n✗ Could not automatically extract URLs.")
        print("\nPlease visit the dataset page and manually create the URL mapping:")
        print(f"  https://datadryad.org/stash/dataset/doi:{doi.replace(':', '%3A')}")
        print("\nCreate a JSON file with the format:")
        print('  {"Genotypes_26503SNPs.txt": "https://datadryad.org/stash/downloads/file_stream/XXXXX", ...}')


