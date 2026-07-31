# Check if a matches context window overlaps with another matches context window.
import gc
import os
import resource
import subprocess
import sys
import threading
import time

from ollama import chat

try:
    from .config import Config
    from .database_connect_embeddings import TextEmbedding, get_psql_session
    from .populate_vector_db import OllamaEmbeddingWrapper
    from .retrieve_vector_data import search_embeddings
except ImportError:  # pragma: no cover - fallback for direct script execution
    from config import Config
    from database_connect_embeddings import TextEmbedding, get_psql_session
    from populate_vector_db import OllamaEmbeddingWrapper
    from retrieve_vector_data import search_embeddings


def is_unique_to_window(existing_matches, current_match, group_window_size=5):

    for match in existing_matches:
        if match[3] != current_match[3]:
            continue
        if (
            match[1] > current_match[1] + group_window_size
            or match[1] < current_match[1] - group_window_size
        ):
            continue
        else:
            return False

    return True


# Getting unique matches from search results
def get_filtered_matches(search_results):
    unique_count = 0
    matches = []
    for result in search_results:
        if unique_count >= 5:
            break
        if is_unique_to_window(matches, result):
            unique_count += 1
            matches.append(result)

    return matches


def group_entries(entry_ids, file_names, index_of_interest, group_window_size):

    # Identify if an entry with index index_of_interest needs grouping with other entries.

    # If it needs no grouping, return an array with just its index (will be handled as in get_surrounding_sentences)
    # If it needs grouping with one or more entries, return array of indices of those entries.

    file_name_of_interest = file_names[index_of_interest]

    group_idxs = [index_of_interest]

    for idx, file_name in enumerate(file_names):
        if idx == index_of_interest:
            continue

        is_nearby_by_position = abs(idx - index_of_interest) <= group_window_size
        is_same_file = file_name == file_name_of_interest

        if is_nearby_by_position or is_same_file:
            group_idxs.append(idx)

    return group_idxs


def consolidate_groupings(grouped_entries):
    # Given a list of lists with grouped entries, combine all lists that have one or more elements in common, then remove duplicates.
    # This should result in a number of lists equal to the number of matched contexts we want

    # Assumes we have run the function group_entries on each entry

    original_groups = grouped_entries[:]
    combined_groups = []

    while len(original_groups):
        current_grouping = original_groups[0][:]
        original_groups.remove(original_groups[0])
        for other_entry in original_groups:
            for idx in current_grouping:
                if idx in other_entry:
                    current_grouping += other_entry
                    original_groups.remove(other_entry)
                    break

        current_grouping = list(set(current_grouping))
        combined_groups.append(current_grouping)

    return combined_groups


def get_min_max_ids(entry_ids, file_names, combined_groups, group_window_size):

    min_ids = []
    max_ids = []

    for group in combined_groups:
        min_id = min([entry_ids[i] for i in group])
        max_id = max([entry_ids[i] for i in group])

        min_id = min_id - group_window_size
        max_id = max_id + group_window_size

        min_ids.append(min_id)
        max_ids.append(max_id)

    return min_ids, max_ids


def get_surrounding_sentences(entry_ids, file_names, group_window_size, session):

    grouped_entries = []
    for idx, id in enumerate(entry_ids):
        grouped_entries.append(
            group_entries(
                entry_ids, file_names, index_of_interest=idx, group_window_size=group_window_size
            )
        )

    combined_groups = consolidate_groupings(grouped_entries)
    min_ids, max_ids = get_min_max_ids(entry_ids, file_names, combined_groups, group_window_size)
    surrounding_sentences = []

    for min_id, max_id in zip(min_ids, max_ids):
        surrounding_sentences.append(
            session.query(
                TextEmbedding.id,
                TextEmbedding.sentence_number,
                TextEmbedding.content,
                TextEmbedding.file_name,
            )
            .filter(TextEmbedding.id >= min_id)
            .filter(TextEmbedding.id <= max_id)
            .all()
        )

    return surrounding_sentences


def search_by_query(query, num_matches=5, group_window_size=5):

    session = get_psql_session()
    # model = SentenceTransformer(Config.EMBEDDING_MODEL_NAME, device='cuda')
    model = OllamaEmbeddingWrapper(Config.EMBEDDING_MODEL_NAME)
    query_embedding = model.encode(query)[0]

    del model
    gc.collect()

    # just nu är max 1 modell aktiv genom export i bashrc men man kan ockås Tvinga Ollama att omedelbart kasta ut embedding-modellen ur VRAM
    # ollama.generate(
    # model='ryanshillington/Qwen3-Embedding-0.6B:latest',
    # keep_alive=0
    # )

    search_results = search_embeddings(
        query_embedding, session=session, limit=num_matches * (2 * group_window_size + 1)
    )
    filtered_matches = get_filtered_matches(search_results)

    entry_ids = [i[0] for i in filtered_matches]
    file_names = [i[3] for i in filtered_matches]

    return get_surrounding_sentences(
        entry_ids=entry_ids,
        file_names=file_names,
        group_window_size=group_window_size,
        session=session,
    )


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


# ==========================================


if __name__ == "__main__":
    # --- 1. STARTA MÄTNINGAR ---
    gpu_tracker = WSLGPUMonitor(interval=0.02)  # Mäter var 20:e millisekund
    gpu_tracker.start()
    start_time = time.perf_counter()

    # ---------------

    # query = "What is the most prolific area of human rights transgressions in Asia?"
    # om man söker på förekomst typ ordmoln bland artiklarna så är det ?"
    # query = "Why where the thai woodcutters killed?"
    # vi förväntar oss ett svar i stil med: "They were mistaken for terrorists"

    query = "Varför dödades de thailändska skogshuggarna?"

    if len(sys.argv) > 1:
        query = sys.argv[1]

    # Force embeddings to CPU
    os.environ["OLLAMA_NUM_GPU"] = "0"
    context = search_by_query(query)
    # reset to use GPU for reasoning model
    os.environ["OLLAMA_NUM_GPU"] = "1"

    # print (f"query: {query}")
    # print (f"context: {context}")

    # prompt = f"<|content_start>{context}<|content_end> {query}"
    # response = chat(model='phi4-mini_4096ctx', messages=[
    # {
    #     'role': 'user',
    #     'content': prompt,
    # },
    # ])
    prompt = f"""
You are a retrieval-augmented assistant.

Answer in the same language as the question.

Use ONLY the context below to answer the question.
If the answer is not in the context, say that you don't know.

Context:
{context}

Question:
{query}
"""
    response = chat(
        # model='mistral:7b-instruct-q4_K_M',
        model=Config.REASONING_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        stream=False,  # ska vara mer lightweight,
        # extra_body={"chat_template_kwargs" : {"enable_thinking": False}} # tror det bara är för qwen 3.5 - nej, funkar inte: got an unexpected keyword argument 'extra_body
    )

    print(response.message.content)

    # -----------------------------

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
