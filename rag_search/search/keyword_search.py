"""In-memory BM25 index and its disk persistence."""

from collections import Counter, defaultdict
import math
import os
import pickle
import traceback

from rag_search.config.defaults import DEFAULT_SEARCH_LIMIT, BM25_K1, BM25_B, CACHE_DIR
from rag_search.search_models import SearchResult, format_search_result
from rag_search.search.tokenization import tokenize_text
from rag_search.sources.movie_loader import load_movies


def smart_print(message, should_print=True):
    if should_print:
        print(message)


class InvertedIndex:
    def __init__(self, cache_dir: str = CACHE_DIR, *, verbose: bool = False) -> None:
        self.verbose = verbose
        self.cache_dir = cache_dir
        self.index = defaultdict(set)
        self.index_path = os.path.join(cache_dir, "index.pkl")

        self.docmap: dict[int, dict] = {}
        self.docmap_path = os.path.join(cache_dir, "docmap.pkl")

        self.term_frequencies_path = os.path.join(cache_dir, "term_frequencies.pkl")
        self.term_frequencies = defaultdict(Counter)

        self.doc_lengths = defaultdict(int)
        self.doc_lengths_path = os.path.join(cache_dir, "doc_lengths.pkl")

        self.avg_doc_length: float = 0.0
        self.avg_doc_length_path = os.path.join(cache_dir, "avg_doc_length.pkl")

    def __add_documents(self, doc_id, text):
        text_tokenized = tokenize_text(text)
        self.doc_lengths[doc_id] = len(text_tokenized)
        self.term_frequencies[doc_id].update(text_tokenized)
        for token in set(text_tokenized):
            self.index[token].add(doc_id)

    def __get_avg_doc_length(self) -> float:
            lengths = self.doc_lengths.values()
            return sum(lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, ())
        return sorted(list(doc_ids))

    def get_tf(self, doc_id: int, token: str) -> int:
        return self.term_frequencies[doc_id][token]

    def get_idf(self, token: str) -> float:
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_tfidf(self, doc_id: int, token: str) -> float:
        return self.get_tf(doc_id, token) * self.get_idf(token)

    def get_bm25_idf(self, token: str) -> float:
        N = len(self.docmap)
        df = len(self.index[token])
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id: int, token: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        tf = self.get_tf(doc_id, token)
        doc_length = self.doc_lengths[doc_id]
        length_norm = 1 - b + b * (doc_length / self.avg_doc_length)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def get_bm25(self, doc_id: int, token: str) -> float:
        return self.get_bm25_tf(doc_id, token) * self.get_bm25_idf(token)

    def bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[SearchResult]:
        query_tokens = tokenize_text(query)
        docs_bm25_scores = defaultdict(float)

        for query_token in query_tokens:
            for doc_id in self.get_documents(query_token):
                docs_bm25_scores[doc_id] += self.get_bm25(doc_id, query_token)

        sorted_docs = sorted(docs_bm25_scores.items(), key=lambda item: item[1], reverse=True)

        results = []
        for doc_id, score in sorted_docs[:limit]:
            doc = self.docmap[doc_id]
            formatted_result = format_search_result(
                doc_id=doc["id"],
                title=doc["title"],
                document=doc["description"],
                score=score,
            )
            results.append(formatted_result)

        return results

    def build(self, documents: list[dict] | None = None) -> bool:
        try:
            smart_print("Processing movies index Data...", self.verbose)
            movies = load_movies() if documents is None else documents
            self.index = defaultdict(set)
            self.docmap = {}
            self.term_frequencies = defaultdict(Counter)
            self.doc_lengths = defaultdict(int)

            for movie in movies:
                self.docmap[movie["id"]] = movie
                self.__add_documents(movie["id"], f"{movie['title']} {movie['description']}")

            self.avg_doc_length = self.__get_avg_doc_length()

            smart_print("Movies Index Data processesed successfully!", self.verbose)
            return True
        except Exception as ex:
            raise Exception(f"Error: Movies Index built failed.\nInternal Error: {ex}\n{traceback.format_exc()}")

    def save(self) -> bool:
        smart_print("Saving Movies Indexes to HardDisk...", self.verbose)

        try:
            os.makedirs(self.cache_dir, exist_ok=True)

            with open(self.index_path, 'wb') as file:
                pickle.dump(self.index, file)

            with open(self.docmap_path, 'wb') as file:
                pickle.dump(self.docmap, file)

            with open(self.term_frequencies_path, 'wb') as file:
                pickle.dump(self.term_frequencies, file)

            with open(self.doc_lengths_path, 'wb') as file:
                pickle.dump(self.doc_lengths, file)

            with open(self.avg_doc_length_path, 'wb') as file:
                pickle.dump(self.avg_doc_length, file)

            smart_print("Movies Index-Data saved successfully in caching folder.", self.verbose)

            return True
        except Exception as ex:
            raise Exception(f"Error: Movies Index Data Save failed.\nInternal Error: {ex}\n{traceback.format_exc()}")

    def load(self) -> bool:
        smart_print("Loading Movies Indexes from HardDisk...", self.verbose)

        try:
            if not os.path.exists(self.index_path) : raise Exception(f"Could not load 'index.pkl' file.\nThe file path: {self.index_path} does not exist.")
            with open(self.index_path, "rb") as file:
                self.index = pickle.load(file=file)

            if not os.path.exists(self.docmap_path) : raise Exception(f"Could not load 'docmap.pkl' file.\nThe file path: {self.docmap_path} does not exist.")
            with open(self.docmap_path, "rb") as file:
                self.docmap = pickle.load(file=file)

            if not os.path.exists(self.term_frequencies_path) : raise Exception(f"Could not load 'term_frequencies.pkl' file.\nThe file path: {self.term_frequencies_path} does not exist.")
            with open(self.term_frequencies_path, "rb") as file:
                self.term_frequencies = pickle.load(file=file)

            if not os.path.exists(self.doc_lengths_path) : raise Exception(f"Could not load 'doc_lengths.pkl' file.\nThe file path: {self.doc_lengths_path} does not exist.")
            with open(self.doc_lengths_path, "rb") as file:
                self.doc_lengths = pickle.load(file=file)

            if not os.path.exists(self.avg_doc_length_path) : raise Exception(f"Could not load 'avg_doc_length.pkl' file.\nThe file path: {self.avg_doc_length_path} does not exist.")
            with open(self.avg_doc_length_path, "rb") as file:
                self.avg_doc_length = pickle.load(file=file)

            smart_print("Movies-Indexes loaded successfully!", self.verbose)
            return True
        except Exception as ex:
            raise Exception(f"Error: Movies Index Data Loading failed.\nInternal Error: {ex}\n{traceback.format_exc()}")
