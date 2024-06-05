from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

class EmbeddingWrapper(Embeddings):
    def __init__(self,chunk_size, model_name="jinaai/jina-embeddings-v2-base-en"):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.max_length = chunk_size  # Set a smaller max length to avoid GPU memory issues
        self.batch_size = 32    # Reduce batch size to fit into GPU memory

    def embed_documents(self, texts):
        all_embeddings = []
        for text in texts:
            # Truncate or split the text into smaller chunks
            tokens = self.model.tokenizer.tokenize(text)
            if len(tokens) <= self.max_length:
                chunks = [text]
            else:
                chunks = [self.model.tokenizer.convert_tokens_to_string(tokens[i:i + self.max_length])
                          for i in range(0, len(tokens), self.max_length)]

            # Encode each chunk separately and average their embeddings
            embeddings = self.model.encode(chunks, batch_size=self.batch_size, convert_to_numpy=True)
            if len(embeddings) > 1:
                embedding = embeddings.mean(axis=0)
            else:
                embedding = embeddings[0]
            all_embeddings.append(embedding)
        return all_embeddings

    def embed_query(self, query):
        # Ensure query is passed as a list and the output is a single vector
        return self.model.encode([query], convert_to_numpy=True)[0]