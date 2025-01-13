import fitz  # PyMuPDF

def highlight_terms_in_pdf(pdf_path, matches, output_path):
    """
    Highlights matched terms in a PDF and saves a new file.
    Arguments:
    - pdf_path (str): Path to the input PDF.
    - matches (list): List of match dictionaries containing 'page' and 'term'.
    - output_path (str): Path to save the highlighted PDF.
    """
    pdf_document = fitz.open(pdf_path)
    
    for match in matches:
        page_number = match['page'] - 1  # PyMuPDF uses 0-based indexing for pages
        term = match['term']
        page = pdf_document[page_number]
        
        # Search for the term on the page and highlight it
        instances = page.search_for(term)
        for instance in instances:
            page.add_highlight_annot(instance)
    
    pdf_document.save(output_path)
    print(f"Saved highlighted PDF to: {output_path}")