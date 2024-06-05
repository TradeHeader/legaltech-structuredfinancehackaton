from langchain.vectorstores.pgvector import PGVector
from langchain.schema import Document
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import logging

class PGVectorStore:
    def __init__(self, connection_string, embedding_wrapper, collection_name="fpml-documents"):
        self.connection_string = connection_string
        self.collection_name = collection_name
        self.embedding_wrapper = embedding_wrapper

        # Create an SQLAlchemy engine
        self.engine = create_engine(self.connection_string)
        self.Session = sessionmaker(bind=self.engine)

    def store_documents(self, data):
        documents = []
        for doc_id, content in data:
            document = Document(page_content=content, metadata={"doc_id": doc_id})
            documents.append(document)

        PGVector.from_documents(
            embedding=self.embedding_wrapper,
            documents=documents,
            connection_string=self.connection_string,  # Use the engine instead of a connection object
            collection_name=self.collection_name,
            use_jsonb=True,
            create_extension=True
        )



    def similarity_search_with_score(self, query_embedding, k=5):
        vector_store = PGVector(
            connection_string=self.connection_string,  # Use the engine instead of a connection object
            embedding_function=self.embedding_wrapper,
            collection_name=self.collection_name,
            use_jsonb=True,
            create_extension=True
        )
        results = vector_store.similarity_search_with_score(query_embedding, k=k)
        return results

    def __del__(self):
        if hasattr(self, 'engine'):
            self.engine.dispose()
