import pdfplumber
from fuzzywuzzy import fuzz
from collections import Counter

def load_lexicon(lexicon_file):
    """Load the lexicon from a CSV file."""
    import pandas as pd
    df = pd.read_csv(lexicon_file)
    return [term.lower() for term in df['term'].tolist()]

def extract_text_by_page(pdf_path):
    """Extract text from a PDF file page by page."""
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                yield page_number + 1, page_text.lower()

def fuzzy_find_terms_in_text(page_number, text, lexicon, threshold=85):
    """Find and record occurrences of lexicon terms on a specific page."""
    matches = []
    lines = text.split("\n")
    
    for term in lexicon:
        for line_number, line in enumerate(lines):
            if fuzz.partial_ratio(term, line) >= threshold:
                matches.append({
                    "term": term,
                    "page": page_number,
                    "line_number": line_number + 1,
                    "context": line.strip()
                })
    return matches

def process_pdf_for_terms(pdf_path, lexicon, threshold=85):
    """Process a PDF to count terms and track their locations."""
    matches = []
    for page_number, page_text in extract_text_by_page(pdf_path):
        page_matches = fuzzy_find_terms_in_text(page_number, page_text, lexicon, threshold)
        matches.extend(page_matches)
    return matches