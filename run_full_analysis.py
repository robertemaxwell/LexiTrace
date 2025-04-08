import os
import multiprocessing
import argparse
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full temporal analysis on clinical trials documents")
    parser.add_argument("--sample", action="store_true", help="Run in sample mode (limit files per year)")
    parser.add_argument("--sample-size", type=int, default=10, help="Files per year in sample mode")
    args = parser.parse_args()
    
    # Determine optimal worker count (use all but one CPU core)
    worker_count = max(1, multiprocessing.cpu_count() - 1)
    
    # Set up command
    base_cmd = ["python", "temporal_analysis.py", "clinical_trials", "lexicon.csv", 
                "output/temporal_analysis", "--workers", str(worker_count)]
    
    # Add sample mode if requested
    if args.sample:
        base_cmd.extend(["--sample", "--sample-size", str(args.sample_size)])
    
    print(f"Starting temporal analysis with {worker_count} worker processes...")
    print(f"Command: {' '.join(base_cmd)}")
    
    # Run the command with filtered output
    process = subprocess.Popen(
        base_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Process and filter output in real-time
    for line in process.stderr:
        if "WARNING:pdfminer.cmapdb:Ignoring (part of) ToUnicode map" not in line:
            print(line, end='', flush=True)
    
    # Print stdout
    for line in process.stdout:
        print(line, end='', flush=True)
    
    # Wait for process to complete
    process.wait() 