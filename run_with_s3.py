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
    """Run the LexiTrace processing script."""
    print("Running LexiTrace processing...")
    
    # Get default number of workers if not specified (use CPU count)
    if num_workers is None:
        import multiprocessing
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # Update to provide the required arguments: pdf_folder, lexicon_file, output_folder and workers
    subprocess.run([
        "python", 
        "process_pdfs.py",
        "clinical_trials",  # pdf_folder - where we downloaded input files
        lexicon_file,       # lexicon_file - from parameter
        "output",           # output_folder - where results should go
        "--workers", str(num_workers)  # number of parallel workers
    ], check=True)

def main():
    # Configure logging to filter PDF warnings
    configure_logging()
    
    parser = argparse.ArgumentParser(description="Run LexiTrace with S3 integration")
    parser.add_argument("--input-bucket", required=True, help="S3 bucket for input data")
    parser.add_argument("--output-bucket", required=True, help="S3 bucket for output results")
    parser.add_argument("--lexicon-file", default="lexicon.csv", help="Lexicon file for term matching (default: lexicon.csv)")
    parser.add_argument("--workers", type=int, help="Number of worker processes for parallel processing")
    parser.add_argument("--s3-workers", type=int, help="Number of worker threads for S3 transfers")
    parser.add_argument("--auto-shutdown", action="store_true", help="Automatically shutdown EC2 instance after completion")
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    
    # Ensure clinical_trials directory exists
    os.makedirs("clinical_trials", exist_ok=True)
    
    # Download data from S3
    download_from_s3(args.input_bucket, "clinical_trials", args.s3_workers)
    
    # Run LexiTrace processing
    run_lexitrace(args.lexicon_file, args.workers)
    
    # Upload results to S3
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

if __name__ == "__main__":
    main()