import os
import sys
import time

import fitz  # från pymupdf
import ollama
import pytesseract
from nltk.tokenize import sent_tokenize
from PIL import Image

try:
    from .config import Config
    from .database_connect_embeddings import get_psql_session, TextEmbedding
except ImportError:  # pragma: no cover - fallback for direct script execution
    from config import Config
    from database_connect_embeddings import get_psql_session, TextEmbedding


# Istället för SentenceTransformer, skapar vi en enkel funktion/klass
class OllamaEmbeddingWrapper:
    def __init__(self, model_name):
        self.model_name = model_name

    def encode(self, sentences):
        # Om det är en singel sträng, gör om till lista
        if isinstance(sentences, str):
            sentences = [sentences]

        embeddings = []
        for sentence in sentences:
            response = ollama.embeddings(model=self.model_name, prompt=sentence)
            embeddings.append(response["embedding"])

        return embeddings


def populate_vector_db(folder_path):
    session = get_psql_session()
    TextEmbedding.truncate(session)
    session.commit()
    # model = SentenceTransformer(Config.EMBEDDING_MODEL_NAME, device="cuda")
    model = OllamaEmbeddingWrapper(Config.EMBEDDING_MODEL_NAME)
    files = os.listdir(folder_path)
    total = len(files)

    for index, file_name in enumerate(files, start=1):
        try:
            if file_name.endswith(".txt"):
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    save_vector(session, model, file_name, content)

            if file_name.endswith(".pdf"):
                file_path = os.path.join(folder_path, file_name)
                with fitz.open(file_path) as f:
                    content = ""
                    for page in f:
                        content += page.get_text() + "\n"
                    save_vector(session, model, file_name, content)

            if file_name.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")):
                file_path = os.path.join(folder_path, file_name)
                img = Image.open(file_path)
                content = pytesseract.image_to_string(img)
                save_vector(session, model, file_name, content)

            print(f"Processed file {index} of {total}")

        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
            continue

    session.close()
    return


def save_vector(session, model, file_name, content):
    sentences = sent_tokenize(content)

    # embeddings = embed(model='nomic-embed-text', input=sentences)['embeddings']
    embeddings = model.encode(sentences)
    for i, (embedding, sentence) in enumerate(zip(embeddings, sentences)):
        new_embedding = TextEmbedding(
            embedding=embedding, content=sentence, file_name=file_name, sentence_number=i + 1
        )
        session.add(new_embedding)
    session.commit()
    print(f"Inserted embeddings for {file_name} into the database.")


if __name__ == "__main__":
    start_time = time.perf_counter()

    folderpath = "all_articles"

    if len(sys.argv) > 1:
        folderpath = sys.argv[1]

    populate_vector_db("./" + folderpath)

    elapsed_time = time.perf_counter() - start_time
    print(f"⏱️  Tid förfluten:         {elapsed_time:.4f} sekunder")
