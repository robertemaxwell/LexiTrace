#!/usr/bin/env python3
import os
import subprocess
import argparse
import boto3
import shutil

def clear_directory(directory):
    """Clear all files in the specified directory."""
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            os.unlink(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

def download_from_s3(bucket_name, local_dir):
    """Download all files from the S3 bucket to the specified directory."""
    print(f"Downloading data from s3://{bucket_name} to {local_dir}")
    s3 = boto3.client('s3')
    
    # Clear the local directory first
    clear_directory(local_dir)
    
    # List objects in the bucket
    response = s3.list_objects_v2(Bucket=bucket_name)
    if 'Contents' in response:
        for obj in response['Contents']:
            file_key = obj['Key']
            local_file_path = os.path.join(local_dir, os.path.basename(file_key))
            print(f"Downloading {file_key} to {local_file_path}")
            s3.download_file(bucket_name, file_key, local_file_path)

def upload_to_s3(local_dir, bucket_name):
    """Upload all files from the local directory to the S3 bucket."""
    print(f"Uploading data from {local_dir} to s3://{bucket_name}")
    s3 = boto3.client('s3')
    
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            # Create S3 key (path within the bucket)
            s3_key = os.path.join(os.path.relpath(root, start=os.path.dirname(local_dir)), file)
            print(f"Uploading {local_file_path} to s3://{bucket_name}/{s3_key}")
            s3.upload_file(local_file_path, bucket_name, s3_key)

def run_lexitrace(lexicon_file="lexicon.csv"):
    """Run the LexiTrace processing script."""
    print("Running LexiTrace processing...")
    # Update to provide the required arguments: pdf_folder, lexicon_file, output_folder
    subprocess.run([
        "python", 
        "process_pdfs.py",
        "clinical_trials",  # pdf_folder - where we downloaded input files
        lexicon_file,       # lexicon_file - from parameter
        "output"            # output_folder - where results should go
    ], check=True)

def main():
    parser = argparse.ArgumentParser(description="Run LexiTrace with S3 integration")
    parser.add_argument("--input-bucket", required=True, help="S3 bucket for input data")
    parser.add_argument("--output-bucket", required=True, help="S3 bucket for output results")
    parser.add_argument("--lexicon-file", default="lexicon.csv", help="Lexicon file for term matching (default: lexicon.csv)")
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    
    # Ensure clinical_trials directory exists
    os.makedirs("clinical_trials", exist_ok=True)
    
    # Download data from S3
    download_from_s3(args.input_bucket, "clinical_trials")
    
    # Run LexiTrace processing
    run_lexitrace(args.lexicon_file)
    
    # Upload results to S3
    upload_to_s3("output", args.output_bucket)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()