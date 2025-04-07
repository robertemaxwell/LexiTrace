#!/usr/bin/env python3
import os
import subprocess
import argparse
import boto3
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import requests
import warnings
import logging
import sys
import importlib

# Configure logging to filter out PDF warnings
class PDFWarningFilter(logging.Filter):
    def filter(self, record):
        # Skip common PDF warnings that don't affect functionality
        pdf_warnings = [
            "Ignoring (part of) ToUnicode map",
            "CropBox missing from /Page",
            "Invalid CMap",
            "Corrupt JPEG data"
        ]
        
        if any(warning in record.getMessage() for warning in pdf_warnings):
            return False
        return True

# Set up logging configuration
def configure_logging():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Add filter to console handler
    console_handler = logging.StreamHandler()
    console_handler.addFilter(PDFWarningFilter())
    root_logger.addHandler(console_handler)
    
    # Also use warnings module to filter common PDF warnings
    warnings.filterwarnings("ignore", message=".*ToUnicode map.*")
    warnings.filterwarnings("ignore", message=".*CropBox missing.*")
    warnings.filterwarnings("ignore", message=".*Invalid CMap.*")
    warnings.filterwarnings("ignore", message=".*Corrupt JPEG data.*")

def check_and_install_dependency(package_name, install_name=None):
    """Check if a package is installed, and install it if not."""
    if install_name is None:
        install_name = package_name
    
    try:
        importlib.import_module(package_name)
        print(f"✓ {package_name} is already installed")
        return True
    except ImportError:
        print(f"✗ {package_name} is not installed. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", install_name])
            print(f"✓ Successfully installed {package_name}")
            return True
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package_name}")
            return False

def check_dependencies():
    """Check and install all dependencies required for process_pdfs.py."""
    print("Checking and installing dependencies...")
    
    # Define the dependencies needed
    dependencies = [
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("tqdm", "tqdm"),
        ("docx", "python-docx"),
        ("pdfplumber", "pdfplumber"),
        ("fuzzywuzzy", "fuzzywuzzy[speedup]"),  # installs python-Levenshtein for speed
    ]
    
    all_installed = True
    for package_name, install_name in dependencies:
        success = check_and_install_dependency(package_name, install_name)
        all_installed = all_installed and success
    
    if all_installed:
        print("\nAll dependencies are installed!")
        return True
    else:
        print("\nSome dependencies could not be installed. Please check the errors above.")
        return False

def clear_directory(directory):
    """Clear all files in the specified directory."""
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            os.unlink(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

def download_single_file(args):
    """Download a single file from S3."""
    s3, bucket_name, file_key, local_file_path = args
    
    # Create directory structure if needed
    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
    
    print(f"Downloading {file_key} to {local_file_path}")
    s3.download_file(bucket_name, file_key, local_file_path)
    return file_key

def download_from_s3(bucket_name, local_dir, max_workers=None):
    """Download all files from the S3 bucket to the specified directory using parallel transfers."""
    if max_workers is None:
        max_workers = min(32, multiprocessing.cpu_count() * 4)  # AWS recommends 4x CPU count
        
    print(f"Downloading data from s3://{bucket_name} to {local_dir} using {max_workers} parallel workers")
    s3 = boto3.client('s3')
    
    # Clear the local directory first
    clear_directory(local_dir)
    
    # List objects in the bucket - use pagination to handle large buckets
    paginator = s3.get_paginator('list_objects_v2')
    
    download_tasks = []
    for page in paginator.paginate(Bucket=bucket_name):
        if 'Contents' in page:
            for obj in page['Contents']:
                file_key = obj['Key']
                # Preserve full path structure
                local_file_path = os.path.join(local_dir, file_key)
                download_tasks.append((s3, bucket_name, file_key, local_file_path))
    
    print(f"Found {len(download_tasks)} files to download")
    
    # Use ThreadPoolExecutor for parallel downloads (better for I/O bound operations)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_single_file, task) for task in download_tasks]
        
        # Wait for all downloads to complete
        for future in as_completed(futures):
            try:
                file_key = future.result()
                print(f"Successfully downloaded {file_key}")
            except Exception as e:
                print(f"Error downloading file: {e}")
    
    print(f"Download complete - {len(download_tasks)} files downloaded to {local_dir}")

def upload_single_file(args):
    """Upload a single file to S3."""
    s3, local_file_path, bucket_name, s3_key = args
    print(f"Uploading {local_file_path} to s3://{bucket_name}/{s3_key}")
    s3.upload_file(local_file_path, bucket_name, s3_key)
    return s3_key

def upload_to_s3(local_dir, bucket_name, max_workers=None):
    """Upload all files from the local directory to the S3 bucket using parallel transfers."""
    if max_workers is None:
        max_workers = min(32, multiprocessing.cpu_count() * 4)  # AWS recommends 4x CPU count
        
    print(f"Uploading data from {local_dir} to s3://{bucket_name} using {max_workers} parallel workers")
    s3 = boto3.client('s3')
    
    upload_tasks = []
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            # Create S3 key (path within the bucket) that preserves directory structure
            s3_key = os.path.relpath(local_file_path, start=os.path.dirname(local_dir))
            upload_tasks.append((s3, local_file_path, bucket_name, s3_key))
    
    print(f"Found {len(upload_tasks)} files to upload")
    
    # Use ThreadPoolExecutor for parallel uploads (better for I/O bound operations)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(upload_single_file, task) for task in upload_tasks]
        
        # Wait for all uploads to complete
        for future in as_completed(futures):
            try:
                s3_key = future.result()
                print(f"Successfully uploaded to {s3_key}")
            except Exception as e:
                print(f"Error uploading file: {e}")
    
    print(f"Upload complete - {len(upload_tasks)} files uploaded to s3://{bucket_name}")

def run_lexitrace(lexicon_file="lexicon.csv", num_workers=None):
    """Run the LexiTrace processing script with better error handling."""
    print("Running LexiTrace processing...")
    
    # Get default number of workers if not specified (use CPU count)
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    try:
        # Run in real-time output mode (no capture) for better visibility
        cmd = [
            sys.executable,  # Use the same Python interpreter
            "process_pdfs.py",
            "clinical_trials",  # pdf_folder - where we downloaded input files
            lexicon_file,       # lexicon_file - from parameter
            "output",           # output_folder - where results should go
            "--workers", str(num_workers),  # number of parallel workers
            "--verbose"         # enable detailed logging
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)
        
        return True
    except subprocess.CalledProcessError as e:
        print("Error running process_pdfs.py:")
        print(f"Exit code: {e.returncode}")
        return False

def main():
    # Configure logging to filter PDF warnings
    configure_logging()
    
    parser = argparse.ArgumentParser(description="Run LexiTrace with S3 integration")
    parser.add_argument("--input-bucket", help="S3 bucket for input data")
    parser.add_argument("--output-bucket", help="S3 bucket for output results") 
    parser.add_argument("--lexicon-file", default="lexicon.csv", help="Lexicon file for term matching (default: lexicon.csv)")
    parser.add_argument("--workers", type=int, help="Number of worker processes for parallel processing")
    parser.add_argument("--s3-workers", type=int, help="Number of worker threads for S3 transfers")
    parser.add_argument("--auto-shutdown", action="store_true", help="Automatically shutdown EC2 instance after completion")
    parser.add_argument("--skip-dependency-check", action="store_true", help="Skip checking and installing dependencies")
    parser.add_argument("--local", action="store_true", help="Run in local mode without S3 (uses local clinical_trials directory)")
    args = parser.parse_args()
    
    # Check dependencies first
    if not args.skip_dependency_check:
        if not check_dependencies():
            print("Failed to install required dependencies. Exiting.")
            return 1
    
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    
    # Ensure clinical_trials directory exists
    os.makedirs("clinical_trials", exist_ok=True)
    
    # Check if we're running in local mode
    if args.local:
        print("Running in local mode - skipping S3 download")
        # Count PDF files in clinical_trials directory
        pdf_count = 0
        for root, _, files in os.walk("clinical_trials"):
            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_count += 1
        
        if pdf_count == 0:
            print("No PDF files found in clinical_trials directory!")
            print("Please add PDF files to the clinical_trials directory before running.")
            return 1
            
        print(f"Found {pdf_count} PDF files in clinical_trials directory")
    else:
        # Require S3 buckets if not in local mode
        if not args.input_bucket or not args.output_bucket:
            print("Error: --input-bucket and --output-bucket are required unless running in --local mode")
            return 1
            
        # Download data from S3
        download_from_s3(args.input_bucket, "clinical_trials", args.s3_workers)
    
    # Run LexiTrace processing
    success = run_lexitrace(args.lexicon_file, args.workers)
    
    if success:
        # Upload results to S3 if not in local mode
        if not args.local and args.output_bucket:
            upload_to_s3("output", args.output_bucket, args.s3_workers)
        
        print("Processing complete!")
        
        # Auto-shutdown if requested
        if args.auto_shutdown:
            print("Auto-shutdown enabled - shutting down instance...")
            # Use boto3 to get instance ID and shut down
            try:
                # Get instance ID from EC2 metadata service
                response = requests.get('http://169.254.169.254/latest/meta-data/instance-id', timeout=2)
                instance_id = response.text
                
                # Create EC2 client and stop the instance
                ec2 = boto3.client('ec2')
                ec2.stop_instances(InstanceIds=[instance_id])
                print(f"Successfully initiated shutdown for instance {instance_id}")
            except Exception as e:
                print(f"Error during auto-shutdown: {e}")
                print("Please manually stop your instance to avoid unnecessary charges")
        
        return 0
    else:
        print("Processing failed. Not uploading to S3.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 