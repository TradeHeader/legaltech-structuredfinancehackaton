import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class MarkdownProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def process(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file at {self.file_path} does not exist.")

        try:
            # Load the markdown content with utf-8 encoding
            loader = TextLoader(self.file_path, encoding="utf-8")
            documents = loader.load()

            # Split the content into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=0)
            chunks = text_splitter.split_documents(documents)
            # Extract the text content from chunks
            chunk_texts = [chunk.page_content for chunk in chunks]

            return chunk_texts

        except Exception as e:
            raise RuntimeError(f"Error processing the file {self.file_path}: {e}")
