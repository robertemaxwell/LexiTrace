#!/usr/bin/env python3
import os
import subprocess
import argparse
import boto3
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

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
    
    # List objects in the bucket
    response = s3.list_objects_v2(Bucket=bucket_name)
    
    download_tasks = []
    if 'Contents' in response:
        for obj in response['Contents']:
            file_key = obj['Key']
            local_file_path = os.path.join(local_dir, os.path.basename(file_key))
            download_tasks.append((s3, bucket_name, file_key, local_file_path))
    
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
            # Create S3 key (path within the bucket)
            s3_key = os.path.join(os.path.relpath(root, start=os.path.dirname(local_dir)), file)
            upload_tasks.append((s3, local_file_path, bucket_name, s3_key))
    
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
    parser = argparse.ArgumentParser(description="Run LexiTrace with S3 integration")
    parser.add_argument("--input-bucket", required=True, help="S3 bucket for input data")
    parser.add_argument("--output-bucket", required=True, help="S3 bucket for output results")
    parser.add_argument("--lexicon-file", default="lexicon.csv", help="Lexicon file for term matching (default: lexicon.csv)")
    parser.add_argument("--workers", type=int, help="Number of worker processes for parallel processing")
    parser.add_argument("--s3-workers", type=int, help="Number of worker threads for S3 transfers")
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

if __name__ == "__main__":
    main()