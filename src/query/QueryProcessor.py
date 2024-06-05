import ollama
import torch


class QueryProcessor:
    def __init__(self, pg_vector_store, device=None):
        self.pg_vector_store = pg_vector_store
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

    def embed_query(self, query):
        return query

    def similarity_search_with_score(self, query, k=5):
        results = self.pg_vector_store.similarity_search_with_score(query, k=k)
        return results

    def generate_response(self, query, k=5):
        similar_docs = self.similarity_search_with_score(query, k=k)
        context = "\n".join([doc.page_content for doc, _ in similar_docs])

        formatted_input = (
            f"Answer the question based only on the following context:\n\n"
            f"{context}\n\n"
            f"---\n\n"
            f"Answer the question based on the above context: {query}"
        )

        print(formatted_input)

        response = self.call_llama_model(formatted_input)
        return response['message']['content']

    def call_llama_model(self, formatted_input):
        response = ollama.chat(model='llama3', messages=[
            {'role': 'user', 'content': formatted_input}
        ])
        return response
