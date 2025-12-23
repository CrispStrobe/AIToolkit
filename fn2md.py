import zipfile
from lxml import etree
from docx import Document

def get_formatted_text(xml_node, ns):
    """
    Parses a footnote node and returns text with Markdown formatting.
    """
    parts = []
    # Find all runs within the footnote
    for run in xml_node.xpath('.//w:r', namespaces=ns):
        text_node = run.xpath('.//w:t', namespaces=ns)
        if not text_node:
            continue
            
        text = text_node[0].text
        if not text:
            continue

        # Check for formatting properties
        is_bold = len(run.xpath('.//w:b', namespaces=ns)) > 0
        is_italic = len(run.xpath('.//w:i', namespaces=ns)) > 0
        is_underline = len(run.xpath('.//w:u', namespaces=ns)) > 0

        # Apply Markdown wrappers
        if is_bold:
            text = f"**{text}**"
        if list(filter(None, [is_italic])): # Handles complex italic tags
            text = f"_{text}_"
        if is_underline:
            text = f"<u>{text}</u>"
            
        parts.append(text)
    
    return "".join(parts).strip()

def extract_footnotes_with_style(file_path):
    """
    Extracts footnotes while preserving basic Bold, Italic, and Underline.
    """
    try:
        # We still use zipfile for the most direct access to the XML
        with zipfile.ZipFile(file_path) as z:
            try:
                xml_content = z.read('word/footnotes.xml')
            except KeyError:
                return "No footnotes found in this document."

            root = etree.fromstring(xml_content)
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            
            footnotes = []
            for footnote in root.xpath('//w:footnote', namespaces=ns):
                f_id = footnote.get(f'{{{ns["w"]}}}id')
                f_type = footnote.get(f'{{{ns["w"]}}}type')
                
                # Filter out structural separators
                if f_type in ['separator', 'continuationSeparator'] or int(f_id) <= 0:
                    continue

                formatted_text = get_formatted_text(footnote, ns)
                if formatted_text:
                    footnotes.append({
                        'id': f_id,
                        'content': formatted_text
                    })
            return footnotes

    except Exception as e:
        return f"Error during extraction: {str(e)}"

# --- Execution ---
file_name = "your_document.docx" 
results = extract_footnotes_with_style(file_name)

if isinstance(results, list):
    for note in results:
        print(f"Footnote {note['id']}: {note['content']}")
else:
    print(results)