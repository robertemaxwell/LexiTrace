import pdfplumber
from fuzzywuzzy import fuzz
from collections import Counter
import re
import pandas as pd

class LexiconTerm:
    def __init__(self, category, primary_term, related_terms, use_wildcard):
        self.category = category
        self.primary_term = primary_term.lower()
        self.related_terms = [term.lower().strip() for term in related_terms.split(',')]
        self.use_wildcard = use_wildcard
        
    def get_all_terms(self):
        """Get all terms including primary and related terms."""
        return [self.primary_term] + self.related_terms

def load_lexicon(lexicon_file):
    """Load the lexicon from a CSV file."""
    df = pd.read_csv(lexicon_file)
    lexicon_terms = []
    for _, row in df.iterrows():
        term = LexiconTerm(
            category=row['category'],
            primary_term=row['primary_term'],
            related_terms=row['related_terms'],
            use_wildcard=row['use_wildcard']
        )
        lexicon_terms.append(term)
    return lexicon_terms

def extract_text_by_page(pdf_path):
    """Extract text from a PDF file page by page."""
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                yield page_number + 1, page_text.lower()

def get_surrounding_context(line, term_position, term_length, window_size=100):
    """Get surrounding context for a term, with a specified window size."""
    start = max(0, term_position - window_size)
    end = min(len(line), term_position + term_length + window_size)
    return line[start:end].strip()

def is_valid_context(context, term):
    """
    Validate if the context actually contains the term and is meaningful.
    Returns tuple of (is_valid, cleaned_context)
    """
    # Basic validation
    if not context or len(context.strip()) <= 1:
        return False, ""
    
    # Clean the context
    cleaned_context = ' '.join(context.split())
    if len(cleaned_context) < 10:  # Require more substantial context
        return False, ""
    
    # For wildcard terms
    if '*' in term:
        pattern = term.replace('*', r'\w*')  # More strict wildcard matching
        matches = list(re.finditer(pattern, cleaned_context, re.IGNORECASE))
        if not matches:
            return False, ""
        # Get surrounding context for the first match
        match = matches[0]
        context_with_window = get_surrounding_context(cleaned_context, match.start(), len(match.group()))
        return True, context_with_window
    
    # For normal terms
    else:
        term_lower = term.lower()
        context_lower = cleaned_context.lower()
        
        # Find the actual position of the term in context
        term_pos = context_lower.find(term_lower)
        if term_pos == -1:
            return False, ""
            
        # Get surrounding context
        context_with_window = get_surrounding_context(cleaned_context, term_pos, len(term))
        return True, context_with_window

def fuzzy_find_terms_in_text(page_number, text, lexicon_terms, threshold=85):
    """Find and record occurrences of lexicon terms on a specific page."""
    matches = []
    lines = text.split("\n")
    
    for lexicon_term in lexicon_terms:
        for line_number, line in enumerate(lines):
            found_match = False
            matched_term = None
            
            # Check all terms (primary and related)
            for term in lexicon_term.get_all_terms():
                if lexicon_term.use_wildcard and '*' in term:
                    # Convert wildcard pattern to regex
                    pattern = term.replace('*', r'\w*')  # More strict wildcard matching
                    if re.search(pattern, line, re.IGNORECASE):
                        # Verify the match has proper word boundaries
                        actual_matches = re.finditer(pattern, line, re.IGNORECASE)
                        for match in actual_matches:
                            matched_text = match.group()
                            if len(matched_text) >= 4:  # Require minimum length for wildcard matches
                                found_match = True
                                matched_term = matched_text
                                break
                else:
                    # For non-wildcard terms, require both high fuzzy match and exact substring
                    fuzzy_score = fuzz.partial_ratio(term, line)
                    term_in_line = term.lower() in line.lower()
                    if fuzzy_score >= threshold and term_in_line:
                        # Additional check: verify the match has proper word boundaries
                        term_lower = term.lower()
                        line_lower = line.lower()
                        term_pos = line_lower.find(term_lower)
                        
                        # Check if the term is a complete word (not part of another word)
                        prev_char = line_lower[term_pos - 1] if term_pos > 0 else ' '
                        next_char = line_lower[term_pos + len(term)] if term_pos + len(term) < len(line_lower) else ' '
                        
                        if not prev_char.isalnum() and not next_char.isalnum():
                            found_match = True
                            matched_term = term
                            break
            
            if found_match and matched_term:
                # Get context and validate it
                is_valid, cleaned_context = is_valid_context(line, matched_term)
                if is_valid:
                    matches.append({
                        "category": lexicon_term.category,
                        "primary_term": lexicon_term.primary_term,
                        "matched_term": matched_term,
                        "page": page_number,
                        "line_number": line_number + 1,
                        "context": cleaned_context
                    })
    
    return matches

def process_pdf_for_terms(pdf_path, lexicon_terms, threshold=85):
    """Process a PDF to count terms and track their locations."""
    matches = []
    for page_number, page_text in extract_text_by_page(pdf_path):
        page_matches = fuzzy_find_terms_in_text(page_number, page_text, lexicon_terms, threshold)
        matches.extend(page_matches)
    return matches