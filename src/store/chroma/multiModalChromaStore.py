import base64
import io

from PIL import Image
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema import Document
from langchain.storage import InMemoryStore
from langchain.vectorstores import Chroma


class MultimodalChromaVectorStore:
    def __init__(self, embedding_wrapper, collection_name="fpml-documents"):
        self.embedding_wrapper = embedding_wrapper
        self.store = InMemoryStore()
        self.client = Chroma(collection_name=collection_name, embedding_function=self.embedding_wrapper)
        self.collection_name = collection_name

    def store_documents(self, data, data_types):
        documents = []
        for doc_id, content in data.items():
            doc_type = data_types.get(doc_id, "text")
            if doc_type == "text":
                embedding = self.embedding_wrapper.embed_query(content)
            elif doc_type == "image":
                image_data = Image.open(io.BytesIO(base64.b64decode(content)))
                inputs = self.embedding_wrapper.image_processor(images=image_data, return_tensors="pt")
                embedding = self.embedding_wrapper.image_model.get_image_features(**inputs).detach().numpy().flatten()
            else:
                embedding = self.embedding_wrapper.embed_query(content)

            document = Document(page_content=content, metadata={"doc_id": doc_id, "data_type": doc_type})
            documents.append((document, embedding.tolist()))

        self.collection.add_documents(documents)

    def list_documents(self):
        return self.collection.list_documents()

    def get_document(self, doc_id):
        return self.collection.get_document(doc_id)

    def delete_document(self, doc_id):
        self.collection.delete_document(doc_id)

    def similarity_search_with_score(self, query, query_type="text", k=5):
        if query_type == "text":
            query_embedding = self.embedding_wrapper.embed_query(query)
        elif query_type == "image":
            image_data = Image.open(io.BytesIO(base64.b64decode(query)))
            inputs = self.embedding_wrapper.image_processor(images=image_data, return_tensors="pt")
            query_embedding = self.embedding_wrapper.image_model.get_image_features(**inputs).detach().numpy().flatten()
        else:
            raise ValueError("Unsupported query type")

        results = self.collection.query(query_embedding=query_embedding.tolist(), n_results=k)
        return results

    def create_multi_vector_retriever(self):
        return MultiVectorRetriever(vector_store=self.collection)