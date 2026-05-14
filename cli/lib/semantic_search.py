from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, text: str):
        if not text or str.isspace(text) :
            raise ValueError(f"Cannot generate embeddings for empty text.")
        return self.model.encode([text])[0]

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