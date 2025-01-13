import pdfplumber
from fuzzywuzzy import fuzz
from collections import Counter, defaultdict
import re
import pandas as pd
from typing import List, Dict, Any, Set, Tuple

class LexiconTerm:
    def __init__(self, category, primary_term, related_terms, use_wildcard):
        self.category = category
        self.primary_term = primary_term.lower()
        self.related_terms = [term.lower().strip() for term in related_terms.split(',')]
        self.use_wildcard = use_wildcard
        
        # Pre-compile regex patterns for efficiency
        self.patterns = self._compile_patterns()
        
        # Track which related terms map to which patterns
        self.term_pattern_mapping = self._create_term_mapping()
        
    def _create_term_mapping(self) -> Dict[str, Set[str]]:
        """Create a mapping of patterns to their source terms."""
        mapping = {}
        all_terms = [self.primary_term] + self.related_terms
        
        for term in all_terms:
            if self.use_wildcard and '*' in term:
                base_term = term.replace('*', '')
                mapping[term] = {base_term}
            else:
                mapping[term] = {term}
        
        return mapping
        
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for all terms."""
        patterns = {}
        all_terms = [self.primary_term] + self.related_terms
        
        for term in all_terms:
            if self.use_wildcard and '*' in term:
                # Convert wildcard to regex pattern
                pattern = term.replace('*', r'\w+')  # More strict: require at least one character
                pattern = rf'\b{pattern}\b'  # Add word boundaries
            else:
                # For compound terms, handle variations in spacing and word boundaries
                words = term.split()
                if len(words) > 1:
                    # Allow flexible spacing between words in compound terms
                    pattern = r'\b' + r'\s+'.join(re.escape(word) for word in words) + r'\b'
                else:
                    pattern = rf'\b{re.escape(term)}\b'
            
            patterns[term] = re.compile(pattern, re.IGNORECASE)
        
        return patterns
    
    def get_all_terms(self) -> List[str]:
        """Get all terms including primary and related terms."""
        return [self.primary_term] + self.related_terms
    
    def classify_match(self, matched_text: str) -> Tuple[str, str]:
        """
        Classify a matched text to determine its specific term type.
        Returns:
            Tuple[str, str]: (term_type, parent_term) where term_type is 'primary' or 'related'
                            and parent_term is the lexicon term that matched
        """
        matched_text_lower = matched_text.lower()
        
        # Check if it matches the primary term
        if self.primary_term in matched_text_lower:
            return 'primary', self.primary_term
        
        # Check which related term it matches
        for term, pattern in self.patterns.items():
            if term != self.primary_term and pattern.search(matched_text_lower):
                return 'related', term
        
        return 'unknown', ''

class MatchClassifier:
    def __init__(self, matches: List[Dict[str, Any]]):
        self.matches = matches
        self.classified_results = self._classify_matches()
    
    def _classify_matches(self) -> Dict[str, Any]:
        """Classify matches into detailed categories and statistics."""
        results = {
            'category_stats': defaultdict(lambda: {
                'total_matches': 0,
                'unique_terms': set(),
                'term_frequencies': Counter(),
                'primary_term_matches': 0,
                'related_term_matches': 0,
                'term_types': defaultdict(set)
            }),
            'term_relationships': defaultdict(set),
            'innovation_patterns': []
        }
        
        # Process each match
        for match in self.matches:
            category = match['category']
            matched_term = match['matched_term'].lower()
            
            # Update category statistics
            cat_stats = results['category_stats'][category]
            cat_stats['total_matches'] += 1
            cat_stats['unique_terms'].add(matched_term)
            cat_stats['term_frequencies'][matched_term] += 1
            
            # Track primary vs related terms
            if match['primary_term'].lower() == matched_term:
                cat_stats['primary_term_matches'] += 1
            else:
                cat_stats['related_term_matches'] += 1
            
            # Group terms by their root/parent term
            if match.get('parent_term'):
                cat_stats['term_types'][match['parent_term']].add(matched_term)
            
            # Track relationships between terms in the same document
            results['term_relationships'][match['filename']].add(category)
        
        # Analyze innovation patterns
        for filename, categories in results['term_relationships'].items():
            if len(categories) > 1:
                results['innovation_patterns'].append({
                    'filename': filename,
                    'pattern': sorted(list(categories)),
                    'complexity': len(categories)
                })
        
        return results
    
    def get_category_statistics(self) -> pd.DataFrame:
        """Generate detailed statistics for each category."""
        stats = []
        for category, data in self.classified_results['category_stats'].items():
            stats.append({
                'category': category,
                'total_matches': data['total_matches'],
                'unique_terms': len(data['unique_terms']),
                'primary_term_matches': data['primary_term_matches'],
                'related_term_matches': data['related_term_matches'],
                'most_common_terms': dict(data['term_frequencies'].most_common(5))
            })
        return pd.DataFrame(stats)
    
    def get_innovation_patterns(self) -> pd.DataFrame:
        """Get analysis of innovation pattern combinations."""
        return pd.DataFrame(self.classified_results['innovation_patterns'])
    
    def get_term_relationships(self) -> Dict[str, Set[str]]:
        """Get mapping of documents to their innovation categories."""
        return self.classified_results['term_relationships']

def load_lexicon(lexicon_file: str) -> List[LexiconTerm]:
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

def extract_text_by_page(pdf_path: str):
    """Extract text from a PDF file page by page."""
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                yield page_number + 1, page_text.lower()

def get_surrounding_context(text: str, match_start: int, match_end: int, window_size: int = 100) -> str:
    """Get surrounding context for a match, with a specified window size."""
    start = max(0, match_start - window_size)
    end = min(len(text), match_end + window_size)
    
    # Expand to complete sentences if possible
    while start > 0 and text[start] not in '.!?\n':
        start -= 1
    while end < len(text) and text[end] not in '.!?\n':
        end += 1
    
    return text[start:end].strip()

def is_valid_match(text: str, match: re.Match, min_context_length: int = 20) -> bool:
    """
    Validate if a regex match is meaningful and has sufficient context.
    Args:
        text: The full text being searched
        match: The regex match object
        min_context_length: Minimum required context length
    Returns:
        bool: Whether the match is valid
    """
    # Get the matched text and its immediate context
    context = get_surrounding_context(text, match.start(), match.end(), 50)
    
    # Check for minimum context length
    if len(context.strip()) < min_context_length:
        return False
    
    # Check if the match is part of a URL or file path
    if re.search(r'https?://|www\.|\.com|\.pdf|\.doc|/|\\', context):
        return False
    
    # Check if the match is part of a reference or citation
    if re.search(r'\[\d+\]|\(\d{4}\)|et al\.', context):
        return False
    
    return True

def fuzzy_find_terms_in_text(page_number: int, text: str, lexicon_terms: List[LexiconTerm], threshold: int = 85) -> List[Dict[str, Any]]:
    """Find and record occurrences of lexicon terms on a specific page."""
    matches = []
    
    for lexicon_term in lexicon_terms:
        # Use pre-compiled patterns for each term
        for term, pattern in lexicon_term.patterns.items():
            for match in pattern.finditer(text):
                # Validate the match
                if is_valid_match(text, match):
                    matched_text = match.group()
                    
                    # For non-wildcard terms, verify fuzzy match score
                    if not lexicon_term.use_wildcard:
                        if fuzz.ratio(term.lower(), matched_text.lower()) < threshold:
                            continue
                    
                    # Get expanded context
                    context = get_surrounding_context(text, match.start(), match.end())
                    
                    # Classify the match
                    term_type, parent_term = lexicon_term.classify_match(matched_text)
                    
                    matches.append({
                        "category": lexicon_term.category,
                        "primary_term": lexicon_term.primary_term,
                        "matched_term": matched_text,
                        "term_type": term_type,
                        "parent_term": parent_term,
                        "page": page_number,
                        "line_number": text[:match.start()].count('\n') + 1,
                        "context": context
                    })
    
    return matches

def process_pdf_for_terms(pdf_path: str, lexicon_terms: List[LexiconTerm], threshold: int = 85) -> List[Dict[str, Any]]:
    """Process a PDF to count terms and track their locations."""
    matches = []
    for page_number, page_text in extract_text_by_page(pdf_path):
        page_matches = fuzzy_find_terms_in_text(page_number, page_text, lexicon_terms, threshold)
        matches.extend(page_matches)
    return matches