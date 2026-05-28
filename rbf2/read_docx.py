import zipfile
import xml.etree.ElementTree as ET

def extract_text_from_docx(docx_path):
    # docx files are zip archives. The text is in word/document.xml.
    try:
        with zipfile.ZipFile(docx_path) as docx:
            xml_content = docx.read('word/document.xml')
            root = ET.fromstring(xml_content)
            
            # The namespace for Word XML elements
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            
            paragraphs = []
            for paragraph in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
                texts = [node.text for node in paragraph.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t') if node.text]
                if texts:
                    paragraphs.append(''.join(texts))
            
            return '\n'.join(paragraphs)
    except Exception as e:
        return f"Error reading docx: {e}"

if __name__ == '__main__':
    text = extract_text_from_docx('RBF2.docx')
    with open('RBF2_extracted.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    print("Done! Extracted text to RBF2_extracted.txt")
