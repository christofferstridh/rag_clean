import argparse
import os
import sys
import time

import fitz  # från pymupdf
import pytesseract
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser, SentenceWindowNodeParser
from nltk.tokenize import sent_tokenize
from PIL import Image
from pyprojroot import here

try:
    from .config import Config
    from .vector_store import configure_embedding_model, get_vector_store
except ImportError:  # pragma: no cover - fallback for direct script execution
    from config import Config
    from vector_store import configure_embedding_model, get_vector_store


def _read_file_content(file_path, file_name):
    """Extract raw text content from a .md, .pdf, or image file. Returns None
    for unsupported extensions."""
    if file_name.endswith(".md"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    if file_name.endswith(".pdf"):
        with fitz.open(file_path) as f:
            content = ""
            for page in f:
                page_text = page.get_text()
                content += (page_text if isinstance(page_text, str) else str(page_text)) + "\n"
            return content

    if file_name.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")):
        with Image.open(file_path) as img:
            return pytesseract.image_to_string(img)

    return None


def build_node_parser():
    """The sentence-window replacement for the old sentence-by-sentence rows +
    manual group_entries/consolidate_groupings/get_min_max_ids logic. Each
    node is one sentence; the surrounding WINDOW_SIZE sentences on either side
    are stored as metadata ("window") and swapped back in at query time by
    MetadataReplacementPostProcessor (see run.py).
    """
    # return SentenceWindowNodeParser.from_defaults(
    #     sentence_splitter=sent_tokenize,  # same nltk tokenizer as before, handles Swedish fine
    #     window_size=Config.WINDOW_SIZE,
    #     window_metadata_key="window",
    #     original_text_metadata_key="original_text",
    # )
    return MarkdownNodeParser()


def index_document(index, node_parser, file_name, content, source):
    """(Re-)index a single file: drop any previously-indexed nodes for this
    file_name+source, then parse and insert fresh nodes. This is the
    replacement for TextEmbedding.delete_by_file_name_and_source + save_vector.
    """
    doc_id = f"{source}::{file_name}"

    # Remove any nodes from a previous run of this same file (no-op if none exist).
    try:
        index.delete_ref_doc(doc_id, delete_from_docstore=True)
    except Exception:
        pass

    document = Document(
        text=content, doc_id=doc_id, metadata={"file_name": file_name, "source": source}
    )
    nodes = node_parser.get_nodes_from_documents([document])
    if not nodes:
        return

    index.insert_nodes(nodes)


def populate_vector_db(folder_path, limit=None):
    configure_embedding_model()  # sets Settings.embed_model = OllamaEmbedding(...)
    vector_store = get_vector_store()
    index = VectorStoreIndex.from_vector_store(vector_store)
    node_parser = build_node_parser()

    folder_path = os.fspath(folder_path)
    source = os.path.basename(os.path.normpath(folder_path))
    files = sorted(os.listdir(folder_path))
    if limit is not None:
        files = files[:limit]
    total = len(files)

    for i, file_name in enumerate(files, start=1):
        file_path = os.path.join(folder_path, file_name)
        try:
            content = _read_file_content(file_path, file_name)
            if content is None:
                continue

            index_document(index, node_parser, file_name, content, source)
            print(f"Processed file {i} of {total}: {file_name}")

        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
            continue


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Populate the vector database from source files.")
    parser.add_argument("--folder", default="all_articles", help="Folder in resources to process")
    parser.add_argument(
        "--limit", type=int, default=None, help="Maximum number of files to process"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    start_time = time.perf_counter()

    args = parse_args(sys.argv[1:])
    populate_vector_db(here() / "resources" / args.folder, limit=args.limit)

    elapsed_time = time.perf_counter() - start_time
    print(f"⏱️  Tid förfluten:         {elapsed_time:.4f} sekunder")
