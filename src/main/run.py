import os
import resource
import subprocess
import sys
import threading
import time

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from ollama import chat

from llama_index.core.schema import MetadataMode

try:
    from .config import Config
    from .vector_store import configure_embedding_model, get_vector_store
except ImportError:  # pragma: no cover - fallback for direct script execution
    from config import Config
    from vector_store import configure_embedding_model, get_vector_store

from pathlib import Path


def search_by_query(query, num_matches=5):
    """Search the vector database for relevant text passages and return grouped context.

    Replaces the old hand-rolled pipeline:
      - get_filtered_matches / is_unique_to_window  -> MMR retrieval (vector_store_query_mode="mmr")
      - group_entries / consolidate_groupings / get_min_max_ids / get_surrounding_sentences
        -> MetadataReplacementPostProcessor, which swaps each matched sentence back out
           for its precomputed "window" of surrounding sentences (built at ingestion
           time by SentenceWindowNodeParser in populate_vector_db.py)

    Returns:
        A list of NodeWithScore, each already expanded to its surrounding window
        and de-duplicated/diversified via MMR.
    """
    configure_embedding_model()  # sets Settings.embed_model = OllamaEmbedding(...)
    vector_store = get_vector_store()
    index = VectorStoreIndex.from_vector_store(vector_store)

    retriever = index.as_retriever(
        similarity_top_k=num_matches,
        vector_store_query_mode="mmr",
        vector_store_kwargs={"mmr_threshold": Config.MMR_THRESHOLD},
    )
    nodes = retriever.retrieve(query)

    postprocessor = MetadataReplacementPostProcessor(target_metadata_key="window")
    nodes = postprocessor.postprocess_nodes(nodes)

    return nodes


class WSLGPUMonitor(threading.Thread):
    def __init__(self, interval=0.05):
        super().__init__()
        self.interval = interval
        self.stopped = False
        self.baseline_vram = self._get_vram()
        self.peak_vram = self.baseline_vram

    def _get_vram(self):
        try:
            # Hämtar aktuellt VRAM från Windows
            result = subprocess.run(
                ["nvidia-smi.exe", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception:
            pass
        return 0.0

    def run(self):
        while not self.stopped:
            current = self._get_vram()
            if current > self.peak_vram:
                self.peak_vram = current
            time.sleep(self.interval)

    def stop(self):
        self.stopped = True


def main(query):
    # Force embeddings to CPU
    os.environ["OLLAMA_NUM_GPU"] = "0"
    start_time = time.perf_counter()
    scored_nodes = search_by_query(query)
    # reset to use GPU for reasoning model
    os.environ["OLLAMA_NUM_GPU"] = str(Config.OLLAMA_NUM_GPU)

    selected_scores = [n.score for n in scored_nodes if n.score is not None][:5]

    EXCLUDE_FROM_LLM = ["document_id", "doc_id", "ref_doc_id", "_node_content", "_node_type"]
    for n in scored_nodes:
        n.node.excluded_llm_metadata_keys = EXCLUDE_FROM_LLM

    context_text = "\n\n---\n\n".join(
        n.get_content(metadata_mode=MetadataMode.LLM) for n in scored_nodes
    )

    # PurePath isolerar endast ok tecken
    safe_name = Path(query).name

    with open(f"debug_out/search_debug {safe_name}.txt", "w", encoding="utf-8") as debug_file:
        debug_file.write("DEBUG SEARCH OUTPUT\n")
        debug_file.write(f"query: {query}\n")
        debug_file.write(f"num_nodes_returned: {len(scored_nodes)}\n")
        debug_file.write(f"selected_scores: {selected_scores}\n")
        for n in scored_nodes:
            debug_file.write(
                f"- file: {n.metadata.get('file_name')} score: {n.score} content: {n.get_content()!r}\n"
            )

    prompt = f"""
You are a retrieval-augmented assistant.

Answer in the same language as the question.

Use ONLY the context below to answer the question.
If the answer is not in the context, say that you don't know.

Context:
{context_text}

Question:
{query}
"""

    with open(f"debug_out/prompt_debug {safe_name}.txt", "w", encoding="utf-8") as prompt_file:
        prompt_file.write("PROMPT DEBUG OUTPUT\n")
        prompt_file.write(f"selected_group_count: {len(scored_nodes)}\n")
        prompt_file.write(f"selected_scores: {selected_scores}\n")
        prompt_file.write(f"context_text length: {len(context_text)}\n")
        prompt_file.write("--- CONTEXT_TEXT START ---\n")
        prompt_file.write(context_text)
        prompt_file.write("\n--- CONTEXT_TEXT END ---\n")
        prompt_file.write(f"question: {query}\n")

    options = Config.OLLAMA_REASONING_OPTIONS or {}
    options = {
        "temperature": 0.0,
        "top_p": 1.0,
        "num_predict": Config.OLLAMA_NUM_PREDICT,
        "num_ctx": Config.OLLAMA_CONTEXT_SIZE,
        **options,
    }

    print(f"\nFRÅGA:\n{query}\nSVAR:\n")
    response = chat(
        model=Config.REASONING_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        options=options,
        keep_alive=Config.OLLAMA_KEEP_ALIVE,
    )
    print(response.message.content)
    print(f"Denna fråga tog {(time.perf_counter() - start_time):.4f} sekunder")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        queries = [sys.argv[1]]
    else:
        queries = [
            # 1a artikeln (vatten etc) - mellansvår
            "Omfattar ICESCR rättigheter till vatten?",
            # vattenartikeln - svår
            "Vad gäller i Australisk lag kring rättighet till vatten i strand-zon(på engelska riparian water)?",
            # 1998 artikeln - lätt
            """I "Human Rights Act 1998" så står det något om en mordbrännare(på engelska arsonist) som ansåg sig ha rätt att vara i klassrummet, vad gällde det?""",
            # thailand - svår
            "Varför dödades de thailändska skogshuggarna?",
        ]
    # --- 1. STARTA MÄTNINGAR ---
    gpu_tracker = WSLGPUMonitor(interval=0.02)  # Mäter var 20:e millisekund
    gpu_tracker.start()
    start_time = time.perf_counter()

    for query in queries:
        main(query)

    # --- 2. STOPPA MÄTNINGAR ---
    elapsed_time = time.perf_counter() - start_time
    gpu_tracker.stop()
    gpu_tracker.join()

    # Hämta CPU och RAM från Ubuntu
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_ram_mb = usage.ru_maxrss / 1024
    total_cpu_time = usage.ru_utime + usage.ru_stime

    # Beräkna hur mycket VRAM ditt skript faktiskt lade till
    vram_used_by_script = max(0.0, gpu_tracker.peak_vram - gpu_tracker.baseline_vram)

    # --- 3. PRESENTERA RESULTAT ---
    print("\n📊 === RESURSUTNYTTJANDE ===")
    print(f"⏱️  Tid förfluten:         {elapsed_time:.4f} sekunder")
    print(f"💻 Total CPU-tid:        {total_cpu_time:.4f} sekunder")
    print(f"🧠 Max RAM-minne (Host):  {peak_ram_mb:.2f} MB")
    print(f"📟 Total VRAM-topp (GPU): {gpu_tracker.peak_vram:.2f} MB")
    print(f"📈 VRAM allokerat av kod: {vram_used_by_script:.2f} MB")
