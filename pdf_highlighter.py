import fitz  # PyMuPDF

def highlight_terms_in_pdf(pdf_path, matches, output_path):
    """
    Highlights matched terms in a PDF and saves a new file.
    Arguments:
    - pdf_path (str): Path to the input PDF.
    - matches (list): List of match dictionaries containing 'page', 'matched_term', and 'category'.
    - output_path (str): Path to save the highlighted PDF.
    """
    # Define colors for different categories (in RGB format)
    category_colors = {
        "Adaptive Trial Design": (1, 0.7, 0.7),       # Light red
        "Modeling and Simulation": (0.7, 1, 0.7),     # Light green
        "Bayesian Approaches": (0.7, 0.7, 1),         # Light blue
        "Remote Trials": (1, 1, 0.7),                 # Light yellow
        "Decentralized Trials": (1, 0.7, 1),          # Light magenta
        "Pragmatic Design": (0.7, 1, 1),              # Light cyan
    }
    
    pdf_document = fitz.open(pdf_path)
    
    for match in matches:
        page_number = match['page'] - 1  # PyMuPDF uses 0-based indexing for pages
        term = match['matched_term']
        category = match['category']
        page = pdf_document[page_number]
        
        # Search for the term on the page and highlight it
        instances = page.search_for(term)
        color = category_colors.get(category, (1, 0.8, 0.8))  # Default to light red if category not found
        
        for instance in instances:
            annot = page.add_highlight_annot(instance)
            annot.set_colors(stroke=color)
            # Add a tooltip showing the category
            annot.set_info(title=category, content=f"Matched term: {term}")
    
    pdf_document.save(output_path)
    print(f"Saved highlighted PDF to: {output_path}")