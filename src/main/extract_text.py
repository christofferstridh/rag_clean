from docling.document_converter import DocumentConverter

def convert_doc(source):
    converter = DocumentConverter()
    text = converter.convert(source)
    return text.document.export_to_dict()