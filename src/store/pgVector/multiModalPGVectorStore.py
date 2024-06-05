from langchain.vectorstores.pgvector import PGVector
from langchain.schema import Document
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import logging
from PIL import Image
import io
import base64

class MultimodalPGVectorStore:
    def __init__(self, connection_string, embedding_wrapper, collection_name="fpml-documents"):
        self.connection_string = connection_string
        self.collection_name = collection_name
        self.embedding_wrapper = embedding_wrapper
        self.engine = create_engine(self.connection_string)
        self.Session = sessionmaker(bind=self.engine)

    def store_documents(self, data, data_types):
        documents = []
        for doc_id, content in data.items():
            doc_type = data_types.get(doc_id, "text")
            document = Document(page_content=content, metadata={"doc_id": doc_id, "data_type": doc_type})
            documents.append(document)

        PGVector.from_documents(
            embedding=self.embedding_wrapper,
            documents=documents,
            connection_string=self.connection_string,
            collection_name=self.collection_name,
            use_jsonb=True,
            create_extension=True
        )

    def similarity_search_with_score(self, query,query_type="text", k=5):
        # Embed the query based on its type
        if query_type == "image":
            image_data = Image.open(io.BytesIO(base64.b64decode(query)))
            inputs = self.embedding_wrapper.image_processor(images=image_data, return_tensors="pt")
            query_embedding = self.embedding_wrapper.image_model.get_image_features(**inputs).detach().numpy().flatten()
        else:
            query_embedding = self.embedding_wrapper.embed_query(query)


        vector_store = PGVector(
            connection_string=self.connection_string,
            embedding_function=self.embedding_wrapper,
            collection_name=self.collection_name,
            use_jsonb=True,
            create_extension=True
        )
        results = vector_store.similarity_search_with_score_by_vector(query_embedding, k=k)
        return results

    def __del__(self):
        if hasattr(self, 'engine'):
            self.engine.dispose()
