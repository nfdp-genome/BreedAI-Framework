#!/usr/bin/env python3
"""
Download public benchmark datasets for genomic prediction validation.

Supported datasets:
- Van den Berg et al. - Genomic prediction benchmark dataset (Dryad DOI: 10.5061/dryad.rq80k)
- Additional datasets can be added here
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import requests
import hashlib
import tarfile
import zipfile

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available. Install with: pip install tqdm")

class DatasetDownloader:
    """Download and verify public benchmark datasets"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()
        
        # Dataset URLs and metadata
        # Van den Berg et al. dataset from Dryad (DOI: 10.5061/dryad.rq80k)
        # Dataset: "Data from: Across population genomic prediction scenarios in which Bayesian variable selection outperforms GBLUP"
        # Dryad page: https://datadryad.org/stash/dataset/doi:10.5061/dryad.rq80k
        
        self.datasets = {
            'vandenberg': {
                'name': 'Van den Berg et al. Genomic Prediction Benchmark',
                'doi': '10.5061/dryad.rq80k',
                'description': 'Genomic prediction benchmark dataset with multiple QTL scenarios (3, 30, 300, 3000) and genetic correlations (0.4, 0.8, 1.0)',
                'files': [
                    {
                        'name': 'Genotypes_26503SNPs.txt',
                        'filename': 'Genotypes_26503SNPs.txt',
                        'url': None,  # Will be fetched from Dryad API
                        'checksum': None
                    },
                    # Phenotype files: QTL scenarios × Genetic correlations
                    # Pattern: Phenotypes_QTL[scenario]_cor[correlation].txt
                    {
                        'name': 'Phenotypes_QTL3_cor0.4.txt',
                        'filename': 'Phenotypes_QTL3_cor0.4.txt',
                        'url': None,
                        'checksum': None
                    },
                    {
                        'name': 'Phenotypes_QTL3_cor0.8.txt',
                        'filename': 'Phenotypes_QTL3_cor0.8.txt',
                        'url': None,
                        'checksum': None
                    },
                    {
                        'name': 'Phenotypes_QTL3_cor1.0.txt',
                        'filename': 'Phenotypes_QTL3_cor1.0.txt',
                        'url': None,
                        'checksum': None
                    },
                    {
                        'name': 'Phenotypes_QTL30_cor0.4.txt',
                        'filename': 'Phenotypes_QTL30_cor0.4.txt',
                        'url': None,
                        'checksum': None
                    },
                    {
                        'name': 'Phenotypes_QTL30_cor0.8.txt',
                        'filename': 'Phenotypes_QTL30_cor0.8.txt',
                        'url': None,
                        'checksum': None
                    },
                    {
                        'name': 'Phenotypes_QTL30_cor1.0.txt',
                        'filename': 'Phenotypes_QTL30_cor1.0.txt',
                        'url': None,
                        'checksum': None
                    },
                    {
                        'name': 'Phenotypes_QTL300_cor0.4.txt',
                        'filename': 'Phenotypes_QTL300_cor0.4.txt',
                        'url': None,
                        'checksum': None
                    },
                    {
                        'name': 'Phenotypes_QTL300_cor0.8.txt',
                        'filename': 'Phenotypes_QTL300_cor0.8.txt',
                        'url': None,
                        'checksum': None
                    },
                    {
                        'name': 'Phenotypes_QTL300_cor1.0.txt',
                        'filename': 'Phenotypes_QTL300_cor1.0.txt',
                        'url': None,
                        'checksum': None
                    },
                    {
                        'name': 'Phenotypes_QTL3000_cor0.4.txt',
                        'filename': 'Phenotypes_QTL3000_cor0.4.txt',
                        'url': None,
                        'checksum': None
                    },
                    {
                        'name': 'Phenotypes_QTL3000_cor0.8.txt',
                        'filename': 'Phenotypes_QTL3000_cor0.8.txt',
                        'url': None,
                        'checksum': None
                    },
                    {
                        'name': 'Phenotypes_QTL3000_cor1.0.txt',
                        'filename': 'Phenotypes_QTL3000_cor1.0.txt',
                        'url': None,
                        'checksum': None
                    },
                ]
            },
            # Add more datasets here
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'download.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def download_file(self, url, output_path, chunk_size=8192, filename=None):
        """Download a file with progress tracking using requests and tqdm"""
        if filename is None:
            filename = output_path.name
        
        self.logger.info(f"Downloading {filename} from {url}")
        self.logger.info(f"  Saving to: {output_path}")
        
        try:
            # Make request with stream=True for large files
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Get file size from headers
            total_size = int(response.headers.get('content-length', 0))
            
            # Create output directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download with progress bar
            if TQDM_AVAILABLE and total_size > 0:
                with open(output_path, 'wb') as f, tqdm(
                    desc=filename,
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                # Fallback without tqdm
                downloaded = 0
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                sys.stdout.write(f"\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)")
                                sys.stdout.flush()
                sys.stdout.write("\n")
            
            self.logger.info(f"Download complete: {output_path}")
            return True
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Download failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                self.logger.error(f"HTTP Status: {e.response.status_code}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during download: {e}")
            return False
    
    def verify_checksum(self, filepath, expected_checksum):
        """Verify file checksum if provided"""
        if expected_checksum is None:
            return True
        
        self.logger.info(f"Verifying checksum for {filepath}")
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        
        calculated = hash_md5.hexdigest()
        if calculated.lower() != expected_checksum.lower():
            self.logger.error(f"Checksum mismatch! Expected: {expected_checksum}, Got: {calculated}")
            return False
        
        self.logger.info("Checksum verified successfully")
        return True
    
    def extract_archive(self, archive_path, extract_to=None):
        """Extract tar.gz, zip, or 7z archive"""
        if extract_to is None:
            extract_to = archive_path.parent
        
        self.logger.info(f"Extracting {archive_path.name} to {extract_to}")
        
        try:
            if archive_path.suffix == '.gz' or '.tar.gz' in archive_path.name:
                with tarfile.open(archive_path, 'r:gz') as tar:
                    tar.extractall(extract_to)
            elif archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif archive_path.suffix == '.7z':
                # Try using py7zr if available
                try:
                    import py7zr
                    with py7zr.SevenZipFile(archive_path, mode='r') as archive:
                        archive.extractall(path=extract_to)
                except ImportError:
                    self.logger.warning("py7zr not available. Install with: pip install py7zr")
                    self.logger.info(f"File {archive_path.name} downloaded but not extracted.")
                    self.logger.info("You can extract it manually or install py7zr: pip install py7zr")
                    return False
            else:
                self.logger.error(f"Unsupported archive format: {archive_path.suffix}")
                return False
            
            self.logger.info(f"Extraction complete")
            return True
        except Exception as e:
            self.logger.error(f"Extraction failed: {e}")
            return False
    
    def get_dryad_file_urls(self, doi):
        """Get actual download URLs from Dryad dataset page"""
        self.logger.info(f"Fetching file list from Dryad for DOI: {doi}")
        
        # Try multiple API endpoints
        api_endpoints = [
            f"https://datadryad.org/api/v2/datasets/{doi.replace(':', '%3A')}/files",
            f"https://datadryad.org/api/v2/datasets/doi%3A{doi}/files",
        ]
        
        file_urls = {}
        
        for api_url in api_endpoints:
            try:
                self.logger.info(f"Trying API endpoint: {api_url}")
                response = requests.get(api_url, timeout=30, headers={'Accept': 'application/json'})
                
                if response.status_code == 200:
                    data = response.json()
                    self.logger.info(f"Successfully fetched file list from Dryad API")
                    
                    # Parse different possible API response structures
                    files_list = []
                    if '_embedded' in data and 'stash:files' in data['_embedded']:
                        files_list = data['_embedded']['stash:files']
                    elif isinstance(data, list):
                        files_list = data
                    elif 'files' in data:
                        files_list = data['files']
                    
                    for file_info in files_list:
                        filename = file_info.get('path', '') or file_info.get('name', '')
                        if not filename:
                            continue
                        
                        # Get download URL
                        download_url = None
                        if '_links' in file_info:
                            links = file_info['_links']
                            if 'stash:download' in links:
                                download_url = links['stash:download'].get('href', '')
                            elif 'download' in links:
                                download_url = links['download'].get('href', '')
                        
                        if download_url:
                            # Extract file ID from URL
                            if '/file_stream/' in download_url:
                                file_id = download_url.split('/file_stream/')[-1]
                            elif '/downloads/' in download_url:
                                file_id = download_url.split('/downloads/')[-1]
                            else:
                                file_id = download_url.split('/')[-1]
                            
                            direct_url = f"https://datadryad.org/stash/downloads/file_stream/{file_id}"
                            file_urls[filename] = direct_url
                            self.logger.info(f"  Found: {filename} -> {direct_url}")
                    
                    if file_urls:
                        return file_urls
                        
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"API request failed: {e}")
                continue
            except Exception as e:
                self.logger.warning(f"Error parsing API response: {e}")
                continue
        
        # If API fails, try scraping the dataset page
        self.logger.info("API methods failed, attempting to parse dataset page...")
        dataset_page_url = f"https://datadryad.org/stash/dataset/doi:{doi.replace(':', '%3A')}"
        
        try:
            response = requests.get(dataset_page_url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
            if response.status_code == 200:
                import re
                # Pattern 1: /stash/downloads/file_stream/[file_id] with filename context
                # Look for links with file_stream and try to extract filename
                pattern1 = r'href=["\'](/stash/downloads/file_stream/[^"\']+)["\']'
                file_stream_links = re.findall(pattern1, response.text)
                
                # Also look for data attributes or text that might contain filenames
                # Pattern: Look for file_stream links near filename patterns
                pattern2 = r'(Genotypes_26503SNPs|Phenotypes_QTL\d+_cor[\d.]+)[^<]*<[^>]*href=["\'](/stash/downloads/file_stream/[^"\']+)["\']'
                matches2 = re.findall(pattern2, response.text, re.IGNORECASE)
                
                if matches2:
                    self.logger.info(f"Found {len(matches2)} file links with filenames")
                    for filename_part, url_path in matches2:
                        full_url = f"https://datadryad.org{url_path}"
                        # Try to match to our expected filenames
                        for expected_name in ['Genotypes_26503SNPs.txt'] + [f'Phenotypes_QTL{n}_cor{c}.txt' 
                                                                           for n in [3, 30, 300, 3000] 
                                                                           for c in [0.4, 0.8, 1.0]]:
                            if filename_part.lower() in expected_name.lower() or expected_name.lower().replace('.txt', '') in filename_part.lower():
                                file_urls[expected_name] = full_url
                                self.logger.info(f"  Matched {expected_name} -> {full_url}")
                
                # Also collect all file_stream links (might be in archives)
                if file_stream_links:
                    self.logger.info(f"Found {len(set(file_stream_links))} unique file_stream links")
                    # Store them for potential archive extraction
                    for link in set(file_stream_links):
                        full_url = f"https://datadryad.org{link}"
                        # Try to infer filename from URL or page context
                        # For now, just log them
                        self.logger.debug(f"  Found link: {full_url}")
                        
        except Exception as e:
            self.logger.warning(f"Could not parse dataset page: {e}")
        
        self.logger.warning("Could not automatically fetch file URLs from Dryad")
        self.logger.info("You may need to manually specify file URLs or download from the dataset page:")
        self.logger.info(f"  {dataset_page_url}")
        
        return file_urls if file_urls else None
    
    def download_dataset(self, dataset_key, overwrite=False):
        """Download a specific dataset"""
        if dataset_key not in self.datasets:
            self.logger.error(f"Unknown dataset: {dataset_key}")
            self.logger.info(f"Available datasets: {list(self.datasets.keys())}")
            return False
        
        dataset = self.datasets[dataset_key]
        self.logger.info("=" * 60)
        self.logger.info(f"Downloading dataset: {dataset['name']}")
        if 'doi' in dataset:
            self.logger.info(f"DOI: {dataset['doi']}")
        self.logger.info("=" * 60)
        
        # For Dryad datasets, try to get actual file URLs
        if 'doi' in dataset and 'dryad' in dataset['doi']:
            self.logger.info("Attempting to fetch file URLs from Dryad...")
            dryad_urls = self.get_dryad_file_urls(dataset['doi'])
            if dryad_urls:
                self.logger.info(f"Found {len(dryad_urls)} files in Dryad dataset")
                # Update file URLs with actual Dryad URLs
                for file_info in dataset.get('files', []):
                    filename = file_info['filename']
                    if filename in dryad_urls:
                        file_info['url'] = dryad_urls[filename]
                        self.logger.info(f"  Updated URL for {filename}")
        
        # Create dataset-specific directory
        dataset_dir = self.output_dir / dataset_key
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Download all files
        if 'files' in dataset:
            # Multi-file dataset
            success_count = 0
            skipped_count = 0
            for file_info in dataset['files']:
                output_file = dataset_dir / file_info['filename']
                
                if output_file.exists() and not overwrite:
                    self.logger.warning(f"File already exists: {output_file}")
                    self.logger.info("Skipping (use --overwrite to re-download)")
                    success_count += 1
                    continue
                
                # Check if URL is available
                if not file_info.get('url'):
                    self.logger.warning(f"No URL available for {file_info['name']} - skipping")
                    skipped_count += 1
                    continue
                
                if not self.download_file(file_info['url'], output_file, filename=file_info['name']):
                    self.logger.error(f"Failed to download {file_info['name']}")
                    continue
                
                # Verify checksum if available
                if file_info.get('checksum'):
                    if not self.verify_checksum(output_file, file_info['checksum']):
                        self.logger.error(f"Checksum verification failed for {file_info['name']}")
                        continue
                
                # Extract archive if needed (7z, zip, tar.gz, etc.)
                if output_file.suffix in ['.gz', '.zip', '.tar', '.7z'] or '.tar.gz' in output_file.name:
                    if not self.extract_archive(output_file, dataset_dir):
                        self.logger.warning(f"Could not extract {output_file.name}, but file was downloaded")
                
                success_count += 1
            
            if success_count == len(dataset['files']):
                self.logger.info(f"✓ Dataset {dataset_key} downloaded successfully ({success_count}/{len(dataset['files'])} files)")
                return True
            else:
                self.logger.warning(f"Partially downloaded: {success_count}/{len(dataset['files'])} files")
                return False
        else:
            # Single file dataset (legacy format)
            output_file = dataset_dir / dataset['filename']
            
            if output_file.exists() and not overwrite:
                self.logger.warning(f"File already exists: {output_file}")
                return True
            
            if not self.download_file(dataset['url'], output_file):
                return False
            
            # Verify checksum if available
            if dataset.get('checksum'):
                if not self.verify_checksum(output_file, dataset['checksum']):
                    self.logger.error("Checksum verification failed")
                    return False
            
            # Extract archive if needed
            if output_file.suffix in ['.gz', '.zip', '.tar'] or '.tar.gz' in output_file.name:
                if not self.extract_archive(output_file, dataset_dir):
                    return False
            
            self.logger.info(f"✓ Dataset {dataset_key} downloaded successfully")
            return True
    
    def list_datasets(self):
        """List all available datasets"""
        self.logger.info("Available datasets:")
        for key, info in self.datasets.items():
            self.logger.info(f"  {key}: {info['name']}")
            self.logger.info(f"    Description: {info['description']}")
            if 'doi' in info:
                self.logger.info(f"    DOI: {info['doi']}")
            if 'files' in info:
                self.logger.info(f"    Files: {len(info['files'])} files")
                for file_info in info['files']:
                    self.logger.info(f"      - {file_info['filename']}")
            elif 'url' in info:
                self.logger.info(f"    URL: {info['url']}")
            self.logger.info("")


def main():
    parser = argparse.ArgumentParser(
        description='Download public benchmark datasets for genomic prediction validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available datasets
  python 01_download_vandenberg.py --list
  
  # Download van den Berg dataset (will attempt to auto-fetch URLs from Dryad)
  python 01_download_vandenberg.py --dataset vandenberg
  
  # Download with manual URL mapping (if auto-fetch fails)
  # First, visit: https://datadryad.org/stash/dataset/doi:10.5061/dryad.rq80k
  # Then create a JSON file with URL mappings and use --url-config option
        """
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/public_datasets/raw',
        help='Directory to save downloaded datasets'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['vandenberg', 'all'],
        default='vandenberg',
        help='Dataset to download'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available datasets'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing files'
    )
    parser.add_argument(
        '--url-config',
        type=str,
        help='JSON file with manual URL mappings: {"filename": "url", ...}'
    )
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.output_dir)
    
    # Load manual URL config if provided
    if args.url_config:
        import json
        try:
            with open(args.url_config, 'r') as f:
                url_mappings = json.load(f)
            downloader.logger.info(f"Loaded {len(url_mappings)} URL mappings from {args.url_config}")
            # Apply to vandenberg dataset if it exists
            if 'vandenberg' in downloader.datasets and 'files' in downloader.datasets['vandenberg']:
                for file_info in downloader.datasets['vandenberg']['files']:
                    if file_info['filename'] in url_mappings:
                        file_info['url'] = url_mappings[file_info['filename']]
                        downloader.logger.info(f"  Set URL for {file_info['filename']}")
        except Exception as e:
            downloader.logger.error(f"Failed to load URL config: {e}")
            return 1
    
    if args.list:
        downloader.list_datasets()
    elif args.dataset == 'all':
        for dataset_key in downloader.datasets.keys():
            downloader.download_dataset(dataset_key, overwrite=args.overwrite)
    else:
        success = downloader.download_dataset(args.dataset, overwrite=args.overwrite)
        if not success:
            downloader.logger.error("\n" + "="*60)
            downloader.logger.error("DOWNLOAD FAILED")
            downloader.logger.error("="*60)
            downloader.logger.error("To manually specify file URLs:")
            downloader.logger.error("1. Visit the dataset page and find download links")
            if 'vandenberg' in downloader.datasets:
                dataset = downloader.datasets['vandenberg']
                if 'doi' in dataset:
                    downloader.logger.error(f"   https://datadryad.org/stash/dataset/doi:{dataset['doi'].replace(':', '%3A')}")
            downloader.logger.error("2. Create a JSON file with URL mappings:")
            downloader.logger.error('   {"Genotypes_26503SNPs.txt": "https://datadryad.org/stash/downloads/file_stream/XXXXX", ...}')
            downloader.logger.error("3. Run with --url-config option:")
            downloader.logger.error("   python 01_download_vandenberg.py --dataset vandenberg --url-config urls.json")
            return 1
    
    return 0


if __name__ == '__main__':
    main()

