"""Index source documents and persist in vector embedding database."""

# Copyright (c) 2023 Brent Benson
#
# This file is part of [project-name], licensed under the MIT License.
# See the LICENSE file in this repository for details.

import os
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

from transformers import AutoTokenizer
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.vectorstores.pgvector import PGVector

SOURCE_DOCUMENTS = ["source_documents/5008_Federalist Papers.pdf"]
COLLECTION_NAME = "doc_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def main():
    print("Ingesting...")
    all_docs = ingest_docs(SOURCE_DOCUMENTS)
    print("Persisting...")
    db = generate_embed_index(all_docs)
    print("Done.")


def ingest_docs(source_documents):
    all_docs = []
    for source_doc in source_documents:
        print(source_doc)
        docs = pdf_to_chunks(source_doc)
        all_docs = all_docs + docs
    return all_docs


def pdf_to_chunks(pdf_file):
    # Use the tokenizer from the embedding model to determine the chunk size
    # so that chunks don't get truncated.
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        separators=["\n \n", "\n\n", "\n", " ", ""],
        chunk_size=512,
        chunk_overlap=0,
    )
    loader = PyPDFLoader(pdf_file)
    docs = loader.load_and_split(text_splitter)
    return docs


def generate_embed_index(docs):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR")
    opensearch_url = os.getenv("OPENSEARCH_URL")
    postgres_conn = os.getenv("POSTGRES_CONNECTION")
    if chroma_persist_dir:
        db = create_index_chroma(docs, embeddings, chroma_persist_dir)
    elif opensearch_url:
        db = create_index_opensearch(docs, embeddings, opensearch_url)
    elif postgres_conn:
        db = create_index_postgres(docs, embeddings, postgres_conn)
    else:
        # You can add additional vector stores here
        raise EnvironmentError("No vector store environment variables found.")
    return db


def create_index_chroma(docs, embeddings, persist_dir):
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )
    db.persist()
    return db


def create_index_opensearch(docs, embeddings, url):
    username = os.getenv("OPENSEARCH_USERNAME")
    password = os.getenv("OPENSEARCH_PASSWORD")
    db = OpenSearchVectorSearch.from_documents(
        docs,
        embeddings,
        index_name=COLLECTION_NAME,
        opensearch_url=url,
        http_auth=(username, password),
        use_ssl=False,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
    )
    return db


def create_index_postgres(docs, embeddings, connection_string):
    db = PGVector.from_documents(
        docs,
        embeddings,
        collection_name=COLLECTION_NAME,
        connection_string=connection_string,
    )
    return db


if __name__ == "__main__":
    main()
