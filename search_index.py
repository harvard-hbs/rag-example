"""The simplest script for embedding-based retrieval."""

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import pprint

COLLECTION_NAME = "federalist_papers"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PERSIST_DIR = "doc_index"

def main():
    # Same model as used to create persisted embedding index
    embeddings = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL)

    # Access persisted embeddings
    db = Chroma(embedding_function=embeddings,
                collection_name=COLLECTION_NAME,
                persist_directory=PERSIST_DIR)
                
    # Example query to for similarity indexing
    prompt = ("How should government responsibility be divided between "
              "the states and the federal government?")

    # Display matched documents and similarity scores
    print(f"Finding document matches for '{prompt}'")
    docs_scores = db.similarity_search_with_score(prompt)
    for doc, score in docs_scores:
        print(f"\nSimilarity score (lower is better): {score}")
        print(doc.metadata)
        print(doc.page_content)

if __name__ == "__main__":
    main()
    
