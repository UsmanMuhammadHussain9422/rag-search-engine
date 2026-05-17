from collections import defaultdict
import os
from typing import Sequence
from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.typing import NDArray
from .search_utils import CACHE_DIR, load_movies

class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings: NDArray[np.floating] | None = None
        self.documents: Sequence[dict] | None = None
        self.document_map: dict[str, dict] = defaultdict(dict)

        self.movie_embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")

    def generate_embedding(self, text: str) -> NDArray[np.floating]:
        if not text or text.isspace():
            raise ValueError("Cannot generate embeddings for empty text.")

        return self.model.encode([text], convert_to_numpy=True)[0]
        
    
    def build_embeddings(self, documents: Sequence[dict]) -> NDArray[np.floating]:
        self.documents = documents
        self.document_map = defaultdict(dict)
        movie_strings = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            movie_strings.append(f"{doc['title']}: {doc['description']}")

        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True, convert_to_numpy=True)

        with open(self.movie_embeddings_path, 'wb') as file:
            np.save(file, self.embeddings)

        return self.embeddings
    
    def load_or_create_embeddings(self, documents: Sequence[dict]) -> NDArray[np.floating]:
        self.documents = documents
        self.document_map = defaultdict(dict)
        for doc in documents:
            self.document_map[doc["id"]] = doc
        
        if os.path.exists(self.movie_embeddings_path):
            self.embeddings = np.load(self.movie_embeddings_path, allow_pickle=False)
            if (self.embeddings is not None and self.documents is not None and 
                len(self.embeddings) == len(self.documents)):
                return self.embeddings
            
        return self.build_embeddings(documents)
    
def verify_model() -> None:
    search_instance = SemanticSearch()
    print(f"Model loaded: {search_instance.model}")
    print(f"Max sequence length: {search_instance.model.max_seq_length}")
    return None 

def embed_text(text: str):
    sentence_Search = SemanticSearch()
    embedding = sentence_Search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    documents: list[dict] = load_movies()
    search_instance = SemanticSearch()
    embeddings = search_instance.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")