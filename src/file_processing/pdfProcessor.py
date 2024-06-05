import os
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFProcessor:
    def __init__(self, file_path, chunk_size, chunk_overlap):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file at {self.file_path} does not exist.")

        try:
            # Load the PDF content
            document = fitz.open(self.file_path)
            pdf_text = ""
            for page in document:
                pdf_text += page.get_text()

            # Split the content into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            chunks = text_splitter.split_text(pdf_text)

            return chunks

        except Exception as e:
            raise RuntimeError(f"Error processing the file {self.file_path}: {e}")
