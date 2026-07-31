from main import generate_corpus, populate_vector_db, retrieve_vector_data, run
from main.database_connect_embeddings import TextEmbedding


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("request failed")

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=30):
        self.calls.append((url, params, headers, timeout))
        if not self._responses:
            return FakeResponse(200, payload={})
        return self._responses.pop(0)


class FakeQuery:
    def __init__(self, rows):
        self.rows = rows

    def filter(self, *args, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def all(self):
        return self.rows


class FakeSessionWithQuery:
    def __init__(self, rows):
        self.rows = rows

    def query(self, *args, **kwargs):
        return FakeQuery(self.rows)


class FakeEmbeddingModel:
    def __init__(self, model_name):
        self.model_name = model_name

    def encode(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences]
        return [[float(len(sentence))] for sentence in sentences]


class FakeSavedSession:
    def __init__(self):
        self.added = []
        self.committed = False
        self.closed = False

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.committed = True

    def close(self):
        self.closed = True


def test_sanitize_filename_replaces_non_alphanumeric():
    assert generate_corpus.sanitize_filename("Hello, World! 123") == "Hello_World_123"


def test_request_json_retries_on_rate_limit(monkeypatch):
    monkeypatch.setattr(generate_corpus.time, "sleep", lambda *_args, **_kwargs: None)

    session = FakeSession(
        [
            FakeResponse(429, headers={"Retry-After": "0"}),
            FakeResponse(200, payload={"ok": True}),
        ]
    )

    assert generate_corpus._request_json("https://example.org", {"q": "x"}, session) == {"ok": True}


def test_fetch_page_content_raises_when_no_pages():
    session = FakeSession([FakeResponse(200, payload={"query": {"pages": {}}})])

    try:
        generate_corpus.fetch_page_content("Example", session=session)
    except ValueError as exc:
        assert "No page data returned" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_generate_corpus_writes_articles_to_output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_corpus.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        generate_corpus,
        "search_wikipedia_titles",
        lambda search_term, results=10, session=None: ["Example Title"],
    )
    monkeypatch.setattr(
        generate_corpus,
        "fetch_page_content",
        lambda title, session=None: f"content for {title}",
    )

    articles = generate_corpus.generate_corpus(
        search_term="human rights",
        num_articles=1,
        output_dir=str(tmp_path),
    )

    assert len(articles) == 1
    assert (tmp_path / "Example_Title.txt").exists()


def test_generate_corpus_returns_empty_list_on_search_failure(monkeypatch):
    monkeypatch.setattr(
        generate_corpus,
        "search_wikipedia_titles",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert generate_corpus.generate_corpus(num_articles=3, output_dir="/tmp") == []


def test_ollama_embedding_wrapper_encode_handles_strings_and_lists(monkeypatch):
    class FakeOllama:
        def embeddings(self, model, prompt):
            return {"embedding": [1.0, 2.0]}

    monkeypatch.setattr(populate_vector_db, "ollama", FakeOllama())

    wrapper = populate_vector_db.OllamaEmbeddingWrapper("demo")
    assert wrapper.encode("hello") == [[1.0, 2.0]]
    assert wrapper.encode(["hello", "there"]) == [[1.0, 2.0], [1.0, 2.0]]


def test_save_vector_adds_embeddings_to_session():
    session = FakeSavedSession()
    model = FakeEmbeddingModel("demo")

    populate_vector_db.save_vector(session, model, "demo.txt", "First sentence. Second sentence.")

    assert len(session.added) == 2
    assert session.added[0].content == "First sentence."
    assert session.committed is True


def test_populate_vector_db_processes_txt_files(tmp_path, monkeypatch):
    target_file = tmp_path / "sample.txt"
    target_file.write_text("Some text", encoding="utf-8")

    monkeypatch.setattr(populate_vector_db, "get_psql_session", lambda: FakeSavedSession())
    monkeypatch.setattr(populate_vector_db.TextEmbedding, "truncate", lambda session: None)
    monkeypatch.setattr(populate_vector_db, "save_vector", lambda *args, **kwargs: None)

    populate_vector_db.populate_vector_db(str(tmp_path))

    assert target_file.exists()


def test_text_embedding_truncate_and_str():
    session = type(
        "FakeSession", (), {"execute": lambda self, *_: None, "commit": lambda self: None}
    )()

    TextEmbedding.truncate(session)
    embed = TextEmbedding(content="hello", file_name="a.txt", sentence_number=1)

    assert str(embed).startswith("hello")


def test_search_embeddings_orders_and_limits_results():
    session = FakeSessionWithQuery([("row-1", 1, "x", "file")])

    results = retrieve_vector_data.search_embeddings([0.1, 0.2], session=session, limit=1)

    assert results == [("row-1", 1, "x", "file")]


def test_get_surrounding_sentences_queries_rows_for_each_group(monkeypatch):
    class FakeQueryResult:
        def __init__(self, rows):
            self.rows = rows

        def filter(self, *args, **kwargs):
            return self

        def all(self):
            return self.rows

    class FakeSession:
        def __init__(self):
            self.queries = []

        def query(self, *args, **kwargs):
            self.queries.append((args, kwargs))
            return FakeQueryResult([("row", 1, "content", "file")])

    session = FakeSession()
    result = run.get_surrounding_sentences([10, 11], ["a", "a"], 1, session)

    assert result == [[("row", 1, "content", "file")]]
    assert len(session.queries) == 1


def test_search_by_query_uses_embedding_and_search_helpers(monkeypatch):
    class FakeSession:
        pass

    class FakeWrapper:
        def __init__(self, model_name):
            self.model_name = model_name

        def encode(self, query):
            return [[0.1, 0.2]]

    calls = {}

    def fake_search_embeddings(query_embedding, session, limit):
        calls["query_embedding"] = query_embedding
        calls["limit"] = limit
        return [(10, 1, "content", "file")]

    monkeypatch.setattr(run, "get_psql_session", lambda: FakeSession())
    monkeypatch.setattr(run, "OllamaEmbeddingWrapper", FakeWrapper)
    monkeypatch.setattr(run, "search_embeddings", fake_search_embeddings)
    monkeypatch.setattr(run, "get_filtered_matches", lambda results: results)
    monkeypatch.setattr(run, "get_surrounding_sentences", lambda **kwargs: ["surrounding"])

    result = run.search_by_query("hello", num_matches=1, group_window_size=1)

    assert result == ["surrounding"]
    assert calls["limit"] == 3
    assert calls["query_embedding"] == [0.1, 0.2]


def test_wsl_gpu_monitor_reads_vram_and_stops(monkeypatch):
    class FakeCompletedProcess:
        def __init__(self, stdout):
            self.returncode = 0
            self.stdout = stdout

    monkeypatch.setattr(run.subprocess, "run", lambda *args, **kwargs: FakeCompletedProcess("12\n"))
    monkeypatch.setattr(run.time, "sleep", lambda *_args, **_kwargs: None)

    monitor = run.WSLGPUMonitor(interval=0)
    assert monitor._get_vram() == 12.0
    monitor.stop()
    monitor.run()
