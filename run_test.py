import os
import sys
from temporal_analysis import process_pdfs_by_year

if __name__ == "__main__":
    # Use just a few files for testing
    base_dir = "clinical_trials"
    lexicon_file = "lexicon.csv"
    output_folder = "output/test_run"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    print("Running temporal analysis test...")
    # Use sample mode to process only a few files per year
    process_pdfs_by_year(
        base_dir, 
        lexicon_file, 
        output_folder, 
        threshold=85, 
        workers=2, 
        sample_mode=True, 
        sample_size=3  # Only process 3 files per year
    )
    print("Test completed.") 