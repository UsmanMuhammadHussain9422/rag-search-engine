from .search_utils import DEFAULT_SEARCH_LIMIT, CACHE_DIR, smart_print, load_movies, load_stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict
import string
import pickle
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
    query_tokens = preprocess_text(query)
    
    for movie in movies:
        # Preprocess title once per movie
        title_tokens = preprocess_text(movie["title"])
        
        # Check for intersection (any query token matches any title token as substring)
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break  # Early exit to avoid unnecessary checks

    return results

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies_idx = InvertedIndex()
    if movies_idx.load() == False:
        return []
    
    query_tokens = preprocess_text(query)
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

def build_command(verbose=False):
    
    global _verbose
    _verbose = verbose

    midx = InvertedIndex()
    midx.build()
    midx.save()

def has_matching_token(query_tokens, title_tokens):
    return any(q_token in t_token for q_token in query_tokens for t_token in title_tokens)

def preprocess_text(text: str) -> list[str]:
    """Preprocess text: remove punctuation, lowercase, tokenize."""
    processed_text = remove_punctuation(text).lower()
    return stemmer(
            remove_stopwords(
                tokenize_text(processed_text)
        )
    )

def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans('', '', string.punctuation))

def tokenize_text(text: str) -> list[str]:
    return text.split()

def remove_stopwords(words: list[str]) -> list[str]:
    stopwords = set(load_stopwords())  # Convert to set for fast lookup; cache if possible
    return [word for word in words if word not in stopwords]

def stemmer(texts: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    unique_texts = set(texts)
    return [stemmer.stem(text) for text in unique_texts]

class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")

    def __add_documents(self, doc_id, text):
        text_tokenized = preprocess_text(text)
        for token in set(text_tokenized):
            self.index[token].add(doc_id)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))
    
    def build(self) -> bool:
        try:
            smart_print("Processing movies index Data...", _verbose)
            movies = load_movies()

            for movie in movies:
                self.docmap[movie["id"]] = movie
                self.__add_documents(movie["id"], f"{movie['title']} {movie['description']}")
            
            smart_print("Movies Index Data processesed successfully!", _verbose)
            return True
        except Exception as ex:
            smart_print(f"Error: Movies Index built failed.\nInternal Error: {ex}\n{traceback.format_exc()}")
            return False
        
    def save(self) -> bool:
        smart_print("Saving Movies Indexes to HardDisk...", _verbose)

        try:
            os.makedirs(CACHE_DIR, exist_ok=True)

            with open(self.index_path, 'wb') as file:
                pickle.dump(self.index, file)
            
            with open(self.docmap_path, 'wb') as file:
                pickle.dump(self.docmap, file)

            smart_print("Movies Index-Data saved successfully in caching folder.", _verbose)
            
            return True
        except Exception as ex:
            smart_print(f"Error: Movies Index Data Save failed.\nInternal Error: {ex}\n{traceback.format_exc()}")
            return False
    
    def load(self) -> bool:
        smart_print("Loading Movies Indexes from HardDisk...", _verbose)

        try:
            if not os.path.exists(self.index_path) : raise Exception(f"Could not load 'index.pkl' file.\nThe file path: {self.index_path} does not exist.")
            with open(self.index_path, "rb") as file:
                self.index = pickle.load(file=file)
        
            if not os.path.exists(self.index_path) : raise Exception(f"Could not load 'docmap.pkl' file.\nThe file path: {self.docmap_path} does not exist.")
            with open(self.docmap_path, "rb") as file:
                self.docmap = pickle.load(file=file)

            smart_print("Movies-Indexes loaded successfully!", _verbose)
            return True
        except Exception as ex:
            smart_print(f"Error: Movies Index Data Loading failed.\nInternal Error: {ex}\n{traceback.format_exc()}")
            return False
        
