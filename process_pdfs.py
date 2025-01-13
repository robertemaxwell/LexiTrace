import os
import pandas as pd
import time
from datetime import timedelta
from term_counter import load_lexicon, process_pdf_for_terms
from pdf_highlighter import highlight_terms_in_pdf

def format_time(seconds):
    """Format time duration in a human-readable format."""
    return str(timedelta(seconds=round(seconds)))

def process_all_pdfs(pdf_folder, lexicon_file, output_folder, threshold=85):
    """Process all PDFs for term counting, context, and highlighting."""
    start_time = time.time()
    
    # Create timestamped subdirectory
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    timestamped_output_dir = os.path.join(output_folder, timestamp)
    os.makedirs(timestamped_output_dir, exist_ok=True)

    print(f"\nLoading lexicon from {lexicon_file}...")
    lexicon_load_start = time.time()
    lexicon_terms = load_lexicon(lexicon_file)
    print(f"Loaded {len(lexicon_terms)} lexicon terms in {format_time(time.time() - lexicon_load_start)}")
    
    results = []
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    total_files = len(pdf_files)
    
    print(f"\nProcessing {total_files} PDF files...")
    
    for file_num, filename in enumerate(pdf_files, 1):
        file_start_time = time.time()
        pdf_path = os.path.join(pdf_folder, filename)
        print(f"\nProcessing file {file_num}/{total_files}: {filename}")

        # Count terms and track context
        matches = process_pdf_for_terms(pdf_path, lexicon_terms, threshold)
        
        # Only process and extend results if there are matches
        if matches:
            results.extend([{
                "filename": filename,
                "category": match["category"],
                "primary_term": match["primary_term"],
                "matched_term": match["matched_term"],
                "page": match["page"],
                "line_number": match["line_number"],
                "context": match["context"]
            } for match in matches])

            # Only create highlighted PDF if there are matches
            highlighted_pdf_path = os.path.join(timestamped_output_dir, f"highlighted_{filename}")
            highlight_terms_in_pdf(pdf_path, matches, highlighted_pdf_path)
            print(f"Created highlighted PDF with {len(matches)} matches")
        else:
            print(f"No matches found in {filename}")
        
        file_duration = time.time() - file_start_time
        print(f"File processing time: {format_time(file_duration)}")
    
    # Only create output files if there are any results
    if results:
        print("\nGenerating reports...")
        report_start_time = time.time()
        
        # Save detailed results to CSV
        csv_output_path = os.path.join(timestamped_output_dir, "term_locations_with_context.csv")
        results_df = pd.DataFrame(results)
        results_df.to_csv(csv_output_path, index=False)
        print(f"Term location results saved to: {csv_output_path}")

        # Generate summary statistics by category
        summary_stats = results_df.groupby('category').agg({
            'filename': 'nunique',
            'matched_term': 'count'
        }).rename(columns={
            'filename': 'unique_files',
            'matched_term': 'total_matches'
        })
        
        # Calculate percentage of files containing each category
        total_files = len(pdf_files)  # Use total PDF files, not just ones with matches
        summary_stats['percentage_of_files'] = (summary_stats['unique_files'] / total_files * 100).round(2)
        
        # Save summary statistics
        summary_path = os.path.join(timestamped_output_dir, "category_summary.csv")
        summary_stats.to_csv(summary_path)
        print(f"Category summary statistics saved to: {summary_path}")
        print(f"Report generation time: {format_time(time.time() - report_start_time)}")
    else:
        print("\nNo matches found in any files")
    
    total_duration = time.time() - start_time
    print(f"\nTotal processing time: {format_time(total_duration)}")
    print(f"Average time per file: {format_time(total_duration/total_files)}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python process_pdfs.py <pdf_folder> <lexicon_file> <output_folder>")
        sys.exit(1)

    pdf_folder = sys.argv[1]
    lexicon_file = sys.argv[2]
    output_folder = sys.argv[3]

    process_all_pdfs(pdf_folder, lexicon_file, output_folder)