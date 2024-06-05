import csv
import os
import re
import ollama
import torch


class RAGSearch:
    def __init__(self, pg_vector_store, device=None):
        self.pg_vector_store = pg_vector_store
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
    def get_enum_descriptions(self, enum_class):
        descriptions = {}
        for member in enum_class:
            description = member.__doc__.strip()
            descriptions[member.name] = description
        return descriptions

    def get_enum_value(self, enum_class, enum_name):
        try:
            return enum_class[enum_name].value
        except KeyError:
            return None

    def process_document_to_question(self,enum_class,enum_name,definition):
        # Search for enum_name in the first column
        csv_file_path = "C:/Users/David/Documents/TradeHeader/RAG/resources/"+str(enum_class.__name__)+"_defitions.csv"
        if os.path.exists(csv_file_path):
            question = None
            with open(csv_file_path, "r", newline="", encoding='utf-8') as csvfile_search:
                reader = csv.reader(csvfile_search)
                for row in reader:
                    if row[0] == enum_name:
                        question = row[2]
                        break

            if(question is not None):
                return question


        # First prompt: Get an extensive definition
        formatted_input = (
            f"Can you give me an extensive definition of the following element?\n\n"
            f"{enum_name +": " +definition}\n\n"
            f"---\n\n"
        )
        response1 = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': formatted_input}])['message'][
            'content']

        # Second prompt: Formulate a question
        formatted_input = (
            f"Given that definition, imagine I have a large document. "
            f"Based on this extensive definition, formulate a very good question that can only be answered with the content of the large document by 'yes' or 'no'. "
            f"The question should help me determine if the document is referring to the enum value defined in the definition or another one. "
            f"I expect that you answer me with 'Question: {{question}}'\n\n"
            f"{response1}\n\n"
        )
        response2 = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': formatted_input}])['message'][
            'content']

        # Extract the question using a regular expression
        match = re.search(r'Question:\s*(.*)', response2)
        if match:
            question = match.group(1).strip()
        else:
            question = "No question found."

        fieldnames = ["enum_value", "definition", "question"]

        # Check if the file exists
        if os.path.exists(csv_file_path):
            # File exists, open in append mode
            with open(csv_file_path, "a", newline="", encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({"enum_value": enum_name, "definition": response1, "question": question})
        else:
            # File doesn't exist, create it and write the headers
            with open(csv_file_path, "w", newline="", encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({"enum_value": enum_name, "definition": response1, "question": question})
        return question

    def similarity_search_with_score(self, query, k=10):
        results = self.pg_vector_store.similarity_search_with_score(query, k=k)
        return results

    def call_llama_model(self, formatted_input):
        response = ollama.chat(model='llama3', messages=[
            {'role': 'user', 'content': formatted_input}
        ])
        return response

    def generate_response(self, query, k=10):
        similar_docs = self.similarity_search_with_score(query, k=k)
        context = "\n".join([doc.page_content for doc, _ in similar_docs])

        formatted_input = (
            f"Responda la pregunta basándose únicamente en el siguiente contexto:\n\n"
            f"{context}\n\n"
            f"---\n\n"
            f"Responda la pregunta basándose en el contexto anterior: {query}"
        )

        response = self.call_llama_model(formatted_input)
        return response['message']['content']

    def search_enum_in_documents(self, enum_class, is_list, k=10):
        enum_descriptions = self.get_enum_descriptions(enum_class)
        results = []
        n_descriptions = len(enum_descriptions)
        counterDescriptions = 0

        print(f"Starting process for enum class: {enum_class.__name__}\n{'=' * 50}\n")

        for enum_name, description in enum_descriptions.items():
            print(f"Processing description {counterDescriptions + 1}/{n_descriptions} for enum: {enum_name}")
            question = self.process_document_to_question(enum_class, enum_name, description)

            affirmative_count = 0  # Count of affirmative responses

            formatted_input = (
                f"Traduceme lo siguiente al castellano, solo quiero que respongas la traducción, ningún mensaje tuyo propio:\n\n"
                f"{question}\n\n"
            )

            pregunta = self.call_llama_model(formatted_input)['message']['content']

            context_elements = self.similarity_search_with_score(pregunta, k)
            context_elements = [ce for ce in context_elements if ce[1] >= 0.65]
            n_context_elements = len(context_elements)

            if n_context_elements == 0:
                print(f"        Enum Value not accepted.")
                counterDescriptions += 1
                continue

            context = "\n".join([doc.page_content for doc, _ in context_elements])

            formatted_input = (
                f"Responde a la pregunta basándote únicamente en el siguiente contexto:\n\n"
                f"{context}\n\n"
                f"---\n\n"
                f"Responde a la pregunta basada en el contexto anterior unicamente con 'Sí' solo si estás 100% seguro que la respuesta es afirmativa. Si no estás 100% seguro, responde unicamente 'No'.\n\n"
                f"Pregunta: {pregunta}"
            )
            # Assuming context is a list of tuples where each tuple contains (page_content, score)

            response = self.call_llama_model(formatted_input)['message']['content']
            if "Si" in response or "Sí" in response:  # This logic may vary based on your model's response format
                affirmative_count += 1
                enum_value = self.get_enum_value(enum_class, enum_name)  # Replace with your actual mapping function
                results.append(enum_value)
                print(f"        Enum Value accepted.")

            else:
                print(f"        Enum Value not accepted.")

            counterDescriptions += 1

            if not is_list and results:
                break

        if is_list:
            return results
        else:
            if results:
                return results[0]
            else:
                return None

        print(f"\nFinished processing for enum class: {enum_class.__name__}\n{'=' * 50}\n")