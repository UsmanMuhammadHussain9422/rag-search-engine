from .search_utils import DEFAULT_SEARCH_LIMIT, BM25_K1, BM25_B, CACHE_DIR, smart_print, load_movies, load_stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter
import string
import pickle
import math
import os
import traceback

_verbose = False

@DeprecationWarning
def search_command_old(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    """
    Search for movies whose titles contain any token from the query (case-insensitive, ignoring punctuation).
    Returns up to 'limit' matching movies as a dict of {id: movie}.
    """
    movies = load_movies()
    results = []
    
    # Preprocess query once (outside loop)
    query_tokens = tokenize_text(query)
    
    for movie in movies:
        # Preprocess title once per movie
        title_tokens = tokenize_text(movie["title"])
        
        # Check for intersection (any query token matches any title token as substring)
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break  # Early exit to avoid unnecessary checks

    return results

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies_idx = InvertedIndex()
    movies_idx.load()
    
    query_tokens = tokenize_text(query)
    seen, results = set(), []
    for query_token in query_tokens:
        matching_doc_ids = movies_idx.get_documents(query_token)
        for doc_id in matching_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = movies_idx.docmap[doc_id]
            results.append(doc)
            if len(results) >= limit:
                return results

    return results

def tf_commmand(doc_id: int, term: str) -> int:
    movies_idx = InvertedIndex()
    movies_idx.load()
    return movies_idx.get_tf(doc_id, term)

def idf_command(term: str) -> float:
    movies_idx = InvertedIndex()
    movies_idx.load()
    return movies_idx.get_idf(term)

def bm25_idf_command(term: str) -> float:
    movies_idx = InvertedIndex()
    movies_idx.load()
    return movies_idx.get_bm25_idf(term)

def tfidf_command(doc_id:int, term: str) -> float:
    movies_idx = InvertedIndex()
    movies_idx.load()
    return movies_idx.get_tfidf(doc_id, term)

def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, term, k1, b)

def build_command(verbose=False):
    
    global _verbose
    _verbose = verbose

    midx = InvertedIndex()
    midx.build()
    midx.save()

def has_matching_token(query_tokens, title_tokens):
    return any(q_token in t_token for q_token in query_tokens for t_token in title_tokens)

def preprocess_text(text: str) -> str:
    """Preprocess text: remove punctuation, lowercase, tokenize."""
    return remove_punctuation(text).lower()

def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans('', '', string.punctuation))

def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    filtered_words = remove_stopwords(tokens)
    stemmed_words = stemmer(filtered_words)
    return stemmed_words

def remove_stopwords(words: list[str]) -> list[str]:
    stopwords = set(load_stopwords())  # Convert to set for fast lookup; cache if possible
    return [word for word in words if word not in stopwords]

def stemmer(texts: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    return [stemmer.stem(text) for text in texts]

class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        
        self.docmap: dict[int, dict] = {}
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.term_frequencies = defaultdict(Counter)
        
        self.doc_lengths = defaultdict(int)
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")
        
        self.avg_doc_length: float = 0.0
        self.avg_doc_length_path = os.path.join(CACHE_DIR, "avg_doc_length.pkl")

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
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))
    
    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        return self.term_frequencies[doc_id][token]

    def get_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_tfidf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        N = len(self.docmap) # total number of documents in the collection
        df = len(self.index[token]) # document frequency (how many documents contain this term)
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.avg_doc_length
        
        # Length normalization factor
        length_norm = 1 - b + b * (doc_length / avg_doc_length)

        # Apply to term frequency
        tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm)

        return tf_component
    
    def build(self) -> bool:
        try:
            smart_print("Processing movies index Data...", _verbose)
            movies = load_movies()

            for movie in movies:
                self.docmap[movie["id"]] = movie
                self.__add_documents(movie["id"], f"{movie['title']} {movie['description']}")

            self.avg_doc_length = self.__get_avg_doc_length()

            smart_print("Movies Index Data processesed successfully!", _verbose)
            return True
        except Exception as ex:
            raise Exception(f"Error: Movies Index built failed.\nInternal Error: {ex}\n{traceback.format_exc()}")
        
    def save(self) -> bool:
        smart_print("Saving Movies Indexes to HardDisk...", _verbose)

        try:
            os.makedirs(CACHE_DIR, exist_ok=True)

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

            smart_print("Movies Index-Data saved successfully in caching folder.", _verbose)
            
            return True
        except Exception as ex:
            raise Exception(f"Error: Movies Index Data Save failed.\nInternal Error: {ex}\n{traceback.format_exc()}")

    def load(self) -> bool:
        smart_print("Loading Movies Indexes from HardDisk...", _verbose)

        try:
            if not os.path.exists(self.index_path) : raise Exception(f"Could not load 'index.pkl' file.\nThe file path: {self.index_path} does not exist.")
            with open(self.index_path, "rb") as file:
                self.index = pickle.load(file=file)
        
            if not os.path.exists(self.index_path) : raise Exception(f"Could not load 'docmap.pkl' file.\nThe file path: {self.docmap_path} does not exist.")
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

            smart_print("Movies-Indexes loaded successfully!", _verbose)
            return True
        except Exception as ex:
            raise Exception(f"Error: Movies Index Data Loading failed.\nInternal Error: {ex}\n{traceback.format_exc()}")
