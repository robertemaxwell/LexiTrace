import os
import pandas as pd
from term_counter import load_lexicon, process_pdf_for_terms
from pdf_highlighter import highlight_terms_in_pdf

def process_all_pdfs(pdf_folder, lexicon_file, output_folder, threshold=85):
    """Process all PDFs for term counting, context, and highlighting."""
    # Create timestamped subdirectory
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    timestamped_output_dir = os.path.join(output_folder, timestamp)
    os.makedirs(timestamped_output_dir, exist_ok=True)

    lexicon_terms = load_lexicon(lexicon_file)
    results = []

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            print(f"\nProcessing: {pdf_path}")

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
    
    # Only create output files if there are any results
    if results:
        # Save detailed results to CSV
        csv_output_path = os.path.join(timestamped_output_dir, "term_locations_with_context.csv")
        results_df = pd.DataFrame(results)
        results_df.to_csv(csv_output_path, index=False)
        print(f"\nTerm location results saved to: {csv_output_path}")

        # Generate summary statistics by category
        summary_stats = results_df.groupby('category').agg({
            'filename': 'nunique',
            'matched_term': 'count'
        }).rename(columns={
            'filename': 'unique_files',
            'matched_term': 'total_matches'
        })
        
        # Calculate percentage of files containing each category
        total_files = len(set(results_df['filename']))
        summary_stats['percentage_of_files'] = (summary_stats['unique_files'] / total_files * 100).round(2)
        
        # Save summary statistics
        summary_path = os.path.join(timestamped_output_dir, "category_summary.csv")
        summary_stats.to_csv(summary_path)
        print(f"Category summary statistics saved to: {summary_path}")
    else:
        print("\nNo matches found in any files")

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python process_pdfs.py <pdf_folder> <lexicon_file> <output_folder>")
        sys.exit(1)

    pdf_folder = sys.argv[1]
    lexicon_file = sys.argv[2]
    output_folder = sys.argv[3]

    process_all_pdfs(pdf_folder, lexicon_file, output_folder)