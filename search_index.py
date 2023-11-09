"""The simplest script for embedding-based retrieval."""

# Copyright (c) 2023 Brent Benson
#
# This file is part of [project-name], licensed under the MIT License.
# See the LICENSE file in this repository for details.

import os
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

COLLECTION_NAME = "doc_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def main():
    # Same model as used to create persisted embedding index
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Access persisted embeddings
    db = get_embed_db(embeddings)

    # Example query to for similarity indexing
    prompt = (
        "How should government responsibility be divided between "
        "the states and the federal government?"
    )

    # Display matched documents and similarity scores
    print(f"Finding document matches for '{prompt}'")
    docs_scores = db.similarity_search_with_score(prompt)
    for doc, score in docs_scores:
        print(f"\nSimilarity score (lower is better): {score}")
        print(doc.metadata)
        print(doc.page_content)


def get_embed_db(embeddings):
    chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR")
    db = get_chroma_db(embeddings, chroma_persist_dir)
    return db


def get_chroma_db(embeddings, persist_dir):
    db = Chroma(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )
    return db


if __name__ == "__main__":
    main()
