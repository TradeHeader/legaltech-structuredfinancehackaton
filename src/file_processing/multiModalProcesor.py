import os
import base64
from PIL import Image
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.html import partition_html
from unstructured.partition.xml import partition_xml
from transformers import CLIPProcessor, CLIPModel
import torch
import ollama

class MultiModalProcessor:
    def __init__(self, output_dir="partition_output"):
        self.output_dir = output_dir
        self.text_elements = []
        self.table_elements = []
        self.image_elements = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def encode_image(self, image_path):
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def summarize_text(self, text):
        response = ollama.chat(model='llama3', messages=[
            {'role': 'user', 'content': f"Summarize the following text:\n\n {text}"}
        ])
        summary = response['message']['content']
        return summary

    def summarize_table(self, text):
        response = ollama.chat(model='llama3', messages=[
            {'role': 'user', 'content': f"Resume la siguiente tabla:\n\n {text}"}
        ])

        response = ollama.chat(model='llama3', messages=[
            {'role': 'user', 'content': f"Genere una descripción a partir de la siguiente tabla\n\n {text}"}
        ])
        summary = response['message']['content']
        print(f"Table summary: {summary}")
        return summary

    def process_file(self, file_path):
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.pdf':
            self.process_pdf(file_path)
        elif file_extension == '.html' or file_extension == '.htm':
            self.process_html(file_path)
        elif file_extension == '.xml':
            self.process_xml(file_path)
        else:
            raise ValueError("Unsupported file type")

    def process_pdf(self, file_path):
        print("Starting PDF processing...")
        raw_pdf_elements = partition_pdf(
            filename=file_path,
            extract_images_in_pdf=False,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=500,
            new_after_n_chars=400,
            combine_text_under_n_chars=100,
        )

        print("PDF partitioning completed. Processing elements...")

        for i, element in enumerate(raw_pdf_elements):
            if 'CompositeElement' in str(type(element)):
                print(f"Processing text element {i + 1}/{len(raw_pdf_elements)}")
                self.text_elements.append(element.text)
            elif 'Table' in str(type(element)):
                print(f"Processing table element {i + 1}/{len(raw_pdf_elements)}")
                summarized_table = self.summarize_table(element.text)
                self.table_elements.append(summarized_table)

        print("PDF processing completed.")

    def process_html(self, file_path):
        print("Starting HTML processing...")
        raw_html_elements = partition_html(
            filename=file_path,
            extract_images_in_html=False,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=1000,
            new_after_n_chars=800,
            combine_text_under_n_chars=500,
        )

        print("HTML partitioning completed. Processing elements...")

        for i, element in enumerate(raw_html_elements):
            if 'CompositeElement' in str(type(element)):
                print(f"Processing text element {i + 1}/{len(raw_html_elements)}")
                self.text_elements.append(element.text)
            elif 'Table' in str(type(element)):
                print(f"Processing table element {i + 1}/{len(raw_html_elements)}")
                summarized_table = self.summarize_table(element.text)
                self.table_elements.append(summarized_table)

        print("HTML processing completed.")

    def process_xml(self, file_path):
        print("Starting XML processing...")
        raw_xml_elements = partition_xml(
            filename=file_path,
            extract_images_in_xml=False,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=1000,
            new_after_n_chars=800,
            combine_text_under_n_chars=500,
        )

        print("XML partitioning completed. Processing elements...")

        for i, element in enumerate(raw_xml_elements):
            if 'CompositeElement' in str(type(element)):
                print(f"Processing text element {i + 1}/{len(raw_xml_elements)}")
                self.text_elements.append(element.text)
            elif 'Table' in str(type(element)):
                print(f"Processing table element {i + 1}/{len(raw_xml_elements)}")
                summarized_table = self.summarize_table(element.text)
                self.table_elements.append(summarized_table)

        print("XML processing completed.")

    def process_folder(self, folder_path):
        all_data = {}
        all_data_types = {}
        file_counter = 0

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                self.text_elements = []
                self.table_elements = []
                self.image_elements = []
                try:
                    self.process_file(file_path)
                    data, data_types = self.get_data()
                    all_data.update({file_counter + i: v for i, (k, v) in enumerate(data.items())})
                    all_data_types.update({file_counter + i: v for i, (k, v) in enumerate(data_types.items())})
                    file_counter += len(data)
                except Exception as e:
                    print(f"Failed to process {file}: {e}")

        return all_data, all_data_types

    def get_data(self):
        print("Compiling data for storage...")
        data = {i: text for i, text in enumerate(self.text_elements)}
        data.update({len(data) + i: table for i, table in enumerate(self.table_elements)})
        data.update({len(data) + len(self.table_elements) + i: image for i, image in enumerate(self.image_elements)})

        data_types = {i: "text" for i in range(len(self.text_elements))}
        data_types.update({len(self.text_elements) + i: "table" for i in range(len(self.table_elements))})
        data_types.update(
            {len(self.text_elements) + len(self.table_elements) + i: "image" for i in range(len(self.image_elements))})

        print("Data compilation completed.")
        return data, data_types