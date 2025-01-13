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

    lexicon = load_lexicon(lexicon_file)
    results = []

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            print(f"\nProcessing: {pdf_path}")

            # Count terms and track context
            matches = process_pdf_for_terms(pdf_path, lexicon, threshold)
            results.extend([{
                "filename": filename,
                "term": match["term"],
                "page": match["page"],
                "line_number": match["line_number"],
                "context": match["context"]
            } for match in matches])

            # Highlight terms in the PDF
            highlighted_pdf_path = os.path.join(timestamped_output_dir, f"highlighted_{filename}")
            highlight_terms_in_pdf(pdf_path, matches, highlighted_pdf_path)
    
    # Save results to CSV
    csv_output_path = os.path.join(timestamped_output_dir, "term_locations_with_context.csv")
    pd.DataFrame(results).to_csv(csv_output_path, index=False)
    print(f"\nTerm location results saved to: {csv_output_path}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python process_pdfs.py <pdf_folder> <lexicon_file> <output_folder>")
        sys.exit(1)

    pdf_folder = sys.argv[1]
    lexicon_file = sys.argv[2]
    output_folder = sys.argv[3]

    process_all_pdfs(pdf_folder, lexicon_file, output_folder)