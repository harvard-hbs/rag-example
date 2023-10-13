# Retrieval-Augmented Generation (RAG) Example

This repository contains an example of code for demonstrating
retrieval-augmented generation or RAG which provides a mechanism for
incorporating domain-specific content into interactions with large
language models (LLMs).

## LangChain

This project depends on the open source
[LangChain](https://www.langchain.com/) library which provides
abstractions and orchestration on top of the these features (among others):

- Document indexing and vector embedding
- Prompt customization, prompt templates, task-specific prompts
- Support for a diverse set of LLMs and LLM interface APIs
- Memory support
- Multi-stage pipelines/chains integrating all features and multiple invokation

## Document Indexing

Document indexing and vector embedding provides a cost-effective
strategy for integrating domain-specific information into large
language model interactions. It allows for domain-driven behavior on
top of general purpose LLMs&mdash;the most common case&mdash; but is
consistent and compatible for use with special purpose or fine-tuned
LLMs.

The basic idea of the approach is to go through a pre-processing step
of indexing domain-specific document text:

- Convert documents into pure text (if needed)
- Break text into appropriately sized chunks (if needed) 
- Generate vector embeddings (a numeric representation of the text
  that represents the semantic information in the text)
- Store the chunks indexed by embedding into a database for retrieval.

This is done in the `index_documents.py` program with this code:

```{python}
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

SOURCE_DOCUMENT = "source_documents/5008_Federalist Papers.pdf"
COLLECTION_NAME = "federalist_papers"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PERSIST_DIR = "doc_index"

def main():
    docs = pdf_to_chunks(SOURCE_DOCUMENT)
    db = generate_embed_index(docs, COLLECTION_NAME, PERSIST_DIR)
    db.persist()

def pdf_to_chunks(pdf_file):
    loader = PyPDFLoader(pdf_file)
    docs = loader.load_and_split()
    return docs

def generate_embed_index(docs, collection_name, persist_dir):
    embeddings = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL)
    db = Chroma.from_documents(documents=docs,
                               embedding=embeddings,
                               collection_name=collection_name,
                               persist_directory=persist_dir)
    return db
```

## Integrating Retrieval

## Memory

## Streamlit Chat UI




