from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from langchain.embeddings.base import Embeddings
import base64
from PIL import Image
import io

class MultimodalEmbeddingWrapper(Embeddings):
    def __init__(self, chunk_size, text_model_name="jinaai/jina-embeddings-v2-base-es", image_model_name="openai/clip-vit-base-patch32"):
        self.text_model = SentenceTransformer(text_model_name, trust_remote_code=True)
        self.image_model = CLIPModel.from_pretrained(image_model_name)
        self.image_processor = CLIPProcessor.from_pretrained(image_model_name)
        self.max_length = chunk_size
        self.batch_size = 32

    def embed_documents(self, texts=None, images=None, tables=None):
        all_embeddings = []

        # Embed text data
        if texts:
            for text in texts:
                tokens = self.text_model.tokenizer.tokenize(text)
                if len(tokens) <= self.max_length:
                    chunks = [text]
                else:
                    chunks = [self.text_model.tokenizer.convert_tokens_to_string(tokens[i:i + self.max_length])
                              for i in range(0, len(tokens), self.max_length)]
                embeddings = self.text_model.encode(chunks, batch_size=self.batch_size, convert_to_numpy=True)
                embedding = embeddings.mean(axis=0) if len(embeddings) > 1 else embeddings[0]
                all_embeddings.append(embedding)

        # Embed image data
        if images:
            for image in images:
                image_data = Image.open(io.BytesIO(base64.b64decode(image)))
                inputs = self.image_processor(images=image_data, return_tensors="pt")
                outputs = self.image_model.get_image_features(**inputs)
                all_embeddings.append(outputs.detach().numpy().flatten())

        # Embed table data as text
        if tables:
            for table in tables:
                tokens = self.text_model.tokenizer.tokenize(table)
                if len(tokens) <= self.max_length:
                    chunks = [table]
                else:
                    chunks = [self.text_model.tokenizer.convert_tokens_to_string(tokens[i:i + self.max_length])
                              for i in range(0, len(tokens), self.max_length)]
                embeddings = self.text_model.encode(chunks, batch_size=self.batch_size, convert_to_numpy=True)
                embedding = embeddings.mean(axis=0) if len(embeddings) > 1 else embeddings[0]
                all_embeddings.append(embedding)

        return all_embeddings

    def embed_query(self, query):
        return self.text_model.encode([query], convert_to_numpy=True)[0]
