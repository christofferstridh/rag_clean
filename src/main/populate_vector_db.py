import argparse
import os
import sys
import time

import fitz  # från pymupdf
import ollama
import pytesseract
from nltk.tokenize import sent_tokenize
from PIL import Image

# för MiniLM
# from sentence_transformers import SentenceTransformer
try:
    from .config import Config
    from .db_stuff import get_psql_session, TextEmbedding
except ImportError:  # pragma: no cover - fallback for direct script execution
    from config import Config
    from db_stuff import get_psql_session, TextEmbedding


# Istället för SentenceTransformer, skapar vi en enkel funktion/klass
class OllamaEmbeddingWrapper:
    def __init__(self, model_name):
        self.model_name = model_name

    def encode(self, sentences):
        # Om det är en singel sträng, gör om till lista
        if isinstance(sentences, str):
            sentences = [sentences]

        options = Config.OLLAMA_EMBEDDING_OPTIONS or {}

        def _call(func, **kwargs):
            try:
                return func(**kwargs)
            except TypeError:
                minimal_kwargs = {
                    k: v for k, v in kwargs.items() if k in {"model", "prompt", "input"}
                }
                return func(**minimal_kwargs)

        if hasattr(ollama, "embed"):
            response = _call(
                ollama.embed,
                model=self.model_name,
                input=sentences,
                options=options,
                keep_alive=Config.OLLAMA_KEEP_ALIVE,
            )
        elif hasattr(ollama, "embeddings"):
            prompt = sentences[0] if isinstance(sentences, list) else sentences
            response = _call(
                ollama.embeddings,
                model=self.model_name,
                prompt=prompt,
                options=options,
                keep_alive=Config.OLLAMA_KEEP_ALIVE,
            )
        else:
            raise AttributeError("ollama client has no embed or embeddings method")

        embeddings = response.get("embeddings", response.get("embedding"))
        if embeddings is None:
            raise ValueError("No embedding returned from Ollama")

        if len(sentences) == 1 and embeddings and isinstance(embeddings[0], (int, float)):
            embeddings = [embeddings]
        elif embeddings and isinstance(embeddings[0], (int, float)):
            embeddings = [embeddings] * len(sentences)

        return embeddings


def _save_vector_with_source(session, model, file_name, content, source):
    try:
        save_vector(session, model, file_name, content, source=source)
    except TypeError as exc:
        if "unexpected keyword argument 'source'" in str(exc):
            save_vector(session, model, file_name, content)
        else:
            raise


def populate_vector_db(folder_path, limit=None):
    session = get_psql_session()

    # för miniLM
    # model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # för bge-m3
    model = OllamaEmbeddingWrapper(Config.EMBEDDING_MODEL_NAME)

    source = folder_path.split("/")[-1]  # sista mappen i sökv
    files = sorted(os.listdir(folder_path))
    if limit is not None:
        files = files[:limit]
    total = len(files)

    for index, file_name in enumerate(files, start=1):
        try:
            if file_name.endswith(".txt"):
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                _save_vector_with_source(session, model, file_name, content, source)

            elif file_name.endswith(".pdf"):
                file_path = os.path.join(folder_path, file_name)
                with fitz.open(file_path) as f:
                    content = ""
                    for page in f:
                        page_text = page.get_text()
                        if isinstance(page_text, str):
                            content += page_text + "\n"
                        else:
                            content += str(page_text) + "\n"
                    _save_vector_with_source(session, model, file_name, content, source)

            elif file_name.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")):
                file_path = os.path.join(folder_path, file_name)
                with Image.open(file_path) as img:
                    content = pytesseract.image_to_string(img)
                _save_vector_with_source(session, model, file_name, content, source)

            print(f"Processed file {index} of {total}")

        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
            continue

    session.commit()
    session.close()
    return


def save_vector(session, model, file_name, content, source=None):
    TextEmbedding.delete_by_file_name(session, file_name)

    sentences = sent_tokenize(content)
    if not sentences:
        return

    embeddings = model.encode(sentences)
    for i, (embedding, sentence) in enumerate(zip(embeddings, sentences)):
        new_embedding = TextEmbedding(
            embedding=embedding,
            content=sentence,
            file_name=file_name,
            sentence_number=i + 1,
            source=source,
        )
        session.add(new_embedding)

    session.commit()
    print(f"Inserted embeddings for {file_name} into the database.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Populate the vector database from source files.")
    parser.add_argument("--folder", default="all_articles", help="Folder containing source files")
    parser.add_argument(
        "--limit", type=int, default=None, help="Maximum number of files to process"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    start_time = time.perf_counter()

    args = parse_args(sys.argv[1:])
    populate_vector_db("./" + args.folder, limit=args.limit)

    elapsed_time = time.perf_counter() - start_time
    print(f"⏱️  Tid förfluten:         {elapsed_time:.4f} sekunder")
