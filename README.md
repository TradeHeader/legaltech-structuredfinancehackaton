# Legal Documents Retrieval and Analysis System

## Project Overview

This project involves building a Retrieval-Augmented Generation (RAG) system to store, process, and analyze legal documents. The core components of this system include:

1. **Vectorized Database**: Legal documents are stored in a vectorized database using PostgreSQL with PGVector for efficient vector storage and retrieval.
2. **Text Embedding**: We use the JinaAI model, which is freely available, to generate embeddings for the text in legal documents.
3. **Data Extraction**: The Python library `unstructured` is used to extract structured data from the legal documents.
4. **Data Processing**: The LLaMA3 model is utilized to process the extracted data from the vectorized database. This involves applying queries to process the data and converting it into Python-CDM format with the help of LLaMA3.

## Installation

To set up the project, you will need to install several libraries. Follow the steps below to install the necessary dependencies:

1. **Install PostgreSQL and PGVector**
    - Ensure you have PostgreSQL installed on your system.
    - Install the PGVector extension for PostgreSQL to enable vectorized storage.

2. **Install Python Libraries**
    - **Unstructured**: Obtain the library from its Git repository.
    - **JinaAI**: Obtain the model from its Git repository on Hugging Face.
    - **LLaMA3**: You can get the LLaMA3 model from Ollama.

## Usage

1. **Vectorizing Legal Documents**: 
    - Load your legal documents into the system.
    - Use the JinaAI model to generate embeddings for these documents.
    - Store the vectorized documents in the PostgreSQL database with PGVector.

2. **Extracting Data**:
    - Use the `unstructured` library to extract data from the documents.
    - Ensure the extracted data is correctly structured and ready for further processing.

3. **Processing Data**:
    - Use the LLaMA3 model to process the data extracted from the vectorized database.
    - Apply necessary queries to transform the data into the desired format.
    - Convert the processed data into Python-CDM format using LLaMA3.

## Main Components

1. **main.py**: This file contains a chatbot interface to make queries to the LLaMA3 model. The chatbot will access the data stored in the RAG and provide relevant responses based on the queries.

2. **test_unstructured.ipynb**: Execute this file if you want to store any data in the repository. It handles the extraction and storage of data from legal documents.

3. **test_builder.ipynb**: Execute this file if you want to build the `contractDetails` CDM object. It processes the data and creates the desired contract details structure.

## Contact

If you need any assistance or have any queries regarding this project, please contact:

- **Manuel Martos**: mmartos@tradeheader.com
- **David Carrascosa**: dcarrascosa@tradeheader.com

## Conclusion

This project aims to streamline the storage, retrieval, and analysis of legal documents by leveraging advanced AI models and vectorized databases. By following the installation and usage instructions, you can set up and utilize this system efficiently for your legal document management needs.