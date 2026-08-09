import argparse
import os
import sys
import time

import fitz  # från pymupdf
import pytesseract
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    MarkdownNodeParser,
    SentenceWindowNodeParser,
    get_leaf_nodes,
)
from nltk.tokenize import sent_tokenize
from PIL import Image
from pyprojroot import here

try:
    from .config import Config
    from .vector_store import configure_embedding_model, get_vector_store
except ImportError:  # pragma: no cover - fallback for direct script execution
    from config import Config
    from vector_store import configure_embedding_model, get_vector_store

import cv2

from llama_index.core.storage.docstore import SimpleDocumentStore


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
        # 1. Load the image in grayscale
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        # 2. Resize to boost DPI (scale up by 2x or 3x if text is small)
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # 3. Apply Otsu's thresholding to get crisp black & white pixels
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # 4. Set the correct Page Segmentation Mode (PSM)
        # --psm 6: Assumes a single uniform block of text
        # --psm 7: Assumes a single line of text
        custom_config = r"--oem 3 --psm 6"

        # 5. Run OCR
        return pytesseract.image_to_string(img, config=custom_config)

    return None


def index_document_markdownish(index, file_name, content, source):
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

    node_parser = MarkdownNodeParser()

    nodes = node_parser.get_nodes_from_documents([document])
    if not nodes:
        return

    index.insert_nodes(nodes)


def index_document_bookish(index, file_name, content, source, docstore):
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

    # Skapa en hierarki: Sektioner (stora) -> Stycken (mellan) -> Meningar (små)
    sub_chunk_sizes = [2048, 512, 128]
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=sub_chunk_sizes)

    # 2. Skapa alla noder (både stora föräldrar och små löv) från din boktext
    all_nodes = node_parser.get_nodes_from_documents([document])
    if not all_nodes:
        return

    # 3. Separera ut de minsta bitarna (löv-noderna)
    leaf_nodes = get_leaf_nodes(all_nodes)

    docstore.add_documents(all_nodes)

    index.insert_nodes(leaf_nodes, insert_kwargs={"docstore": docstore})


def populate_vector_db_bookish(folder_path, limit=None):

    # Försök ladda en sparad docstore från disk, annars skapa en ny
    docstore_path = "./storage_docstore"
    if os.path.exists(docstore_path):
        docstore = SimpleDocumentStore.from_persist_dir(persist_dir=docstore_path)
    else:
        docstore = SimpleDocumentStore()

    configure_embedding_model()  # sets Settings.embed_model = OllamaEmbedding(...)
    vector_store = get_vector_store()

    # Koppla ihop både din vektordatabas och din dokumentdatabas
    storage_context = StorageContext.from_defaults(vector_store=vector_store, docstore=docstore)

    # Skapa indexet med hela kontexten intakt
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

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

            index_document_bookish(index, file_name, content, source, docstore)

            print(f"Processed file {i} of {total}: {file_name}")

        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
            continue

    # KOM IHÅG: När ditt script har kört klart indexering_document_bookish,
    # måste du spara din docstore till disken:
    docstore.persist(persist_path=docstore_path)


def populate_vector_db_markdownish(folder_path, limit=None):
    configure_embedding_model()  # sets Settings.embed_model = OllamaEmbedding(...)
    vector_store = get_vector_store()
    index = VectorStoreIndex.from_vector_store(vector_store)

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

            index_document_markdownish(index, file_name, content, source)

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
    # populate_vector_db_markdownish(here() / "resources" / args.folder, limit=args.limit)
    populate_vector_db_bookish(here() / "resources" / args.folder, limit=args.limit)

    elapsed_time = time.perf_counter() - start_time
    print(f"⏱️  Tid förfluten:         {elapsed_time:.4f} sekunder")
