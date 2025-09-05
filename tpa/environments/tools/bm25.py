"""BM25 lexical retrieval over multiple corpora (title, artist, album, lyrics, attributes)."""
import bm25s
import os
import json
import numpy as np
from tpa.environments.tools.utils import entity_str


class BM25Tool:
    """BM25 retriever wrapper across multiple text corpora.

    Args:
        cache_dir (str): Directory where BM25 indices and metadata live.
    """
    def __init__(self, cache_dir="./cache"):
        self.cache_dir = cache_dir
        self.retriever = self._load_retrievers()
        self.track_ids = self._load_track_ids()
        self.track_id_to_index = {tid: i for i, tid in enumerate(self.track_ids)}
        self._weight_mask = None  # numpy array or None

    def _load_track_ids(self) -> list:
        track_index_path = os.path.join(self.cache_dir, "bm25", "track_index.json")
        if not os.path.exists(track_index_path):
            self.build_index()
        with open(track_index_path, "r") as f:
            return json.load(f)

    def _load_retrievers(self) -> dict:
        """Load BM25 retrievers for each supported corpus.

        Returns:
            dict: Mapping from corpus name to `bm25s.BM25` instance.
        """
        try:
            return {
                "title": self._load_model("title"),
                "artist": self._load_model("artist"),
                "album": self._load_model("album"),
                "lyrics": self._load_model("lyrics"),
                "attributes": self._load_model("attributes"),
            }
        except Exception:
            self.build_index()
            return {
                "title": self._load_model("title"),
                "artist": self._load_model("artist"),
                "album": self._load_model("album"),
                "lyrics": self._load_model("lyrics"),
                "attributes": self._load_model("attributes"),
            }

    def _load_model(self, corpus_name: str) -> bm25s.BM25:
        return bm25s.BM25.load(f"{self.cache_dir}/bm25/{corpus_name}", load_corpus=True)

    def build_index(self):
        """Build BM25 indices for all supported corpora from cached metadata."""
        test_metadata = json.load(
            open(os.path.join(self.cache_dir, "metadata", "test_metadata.json"), "r")
        )
        track_index, title_corpus, artist_corpus, album_corpus, lyrics_corpus, attributes_corpus = [], [], [], [], [], []
        for k, v in test_metadata.items():
            track_index.append(k)
            title_corpus.append(",".join(v["track_name"]).lower())
            artist_corpus.append(",".join(v["artist_name"]).lower())
            album_corpus.append(",".join(v["album_name"]).lower())
            lyrics_corpus.append(v["lyrics"].lower())
            attributes_corpus.append(",".join(v["tag_list"]).lower())

        corpus_list = [title_corpus, artist_corpus, album_corpus, lyrics_corpus, attributes_corpus]
        corpus_names = ["title", "artist", "album", "lyrics", "attributes"]

        os.makedirs(os.path.join(self.cache_dir, "bm25"), exist_ok=True)
        with open(os.path.join(self.cache_dir, "bm25", "track_index.json"), "w") as f:
            json.dump(track_index, f, indent=2)

        for corpus, name in zip(corpus_list, corpus_names):
            corpus_tokens = bm25s.tokenize(corpus)
            retriever = bm25s.BM25()
            retriever.index(corpus_tokens)
            retriever.save(f"{self.cache_dir}/bm25/{name}", corpus=corpus)

    def _update_track_pool(self, track_pool: list[str] | None):
        """Restrict retrieval to a candidate set of track ids.

        Args:
            track_pool (list[str] | None): Allowed track ids; None means no restriction.
        """
        any_retriever = next(iter(self.retriever.values()))
        num_docs = any_retriever.scores["num_docs"]
        mask = np.zeros(num_docs, dtype=np.float32)
        for tid in track_pool:
            idx = self.track_id_to_index.get(tid)
            if idx is not None and 0 <= idx < num_docs:
                mask[idx] = 1.0
        self._weight_mask = mask

    def _init_track_pool(self):
        """Clear any previously set candidate restriction mask."""
        self._weight_mask = None

    def bm25_retrieval(self, corpus_type: str, query: str, k: int):
        """Run BM25 retrieval over the specified corpus.

        Args:
            corpus_type (str): One of {"title", "artist", "album", "lyrics", "attributes"}.
            query (str): Search query string (will be lowercased).
            k (int): Number of results to return.

        Returns:
            list[str]: Ranked track ids.
        """
        try:
            bm25_model = self.retriever[corpus_type]
            query_tokens = bm25s.tokenize([query.lower()])
            doc_scores = bm25_model.retrieve(
                query_tokens, k=k, return_as="tuple", weight_mask=self._weight_mask
            )
            docs = doc_scores.documents[0]
            ids = [doc["id"] for doc in docs]
            return [self.track_ids[idx] for idx in ids]
        except Exception as e:
            raise ValueError(f"Error in bm25_retrieval: {e}")


# bm25_tool = BM25Tool()
# bm25_tool.build_index()
# print(bm25_tool.bm25_retrieval("metadata", "popularity", 10))
