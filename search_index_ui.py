"""User interface for seeing vector database matches."""

# Copyright (c) 2023 Brent Benson
#
# This file is part of [project-name], licensed under the MIT License.
# See the LICENSE file in this repository for details.

import os
import streamlit as st
import pprint
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores import OpenSearchVectorSearch

from streamlit.logger import get_logger

logger = get_logger(__name__)

COLLECTION_NAME = "doc_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

ANSWER_ROLE = "Document Index"
FIRST_MESSAGE = "Enter text to find document matches."
QUESTION_ROLE = "Searcher"
PLACE_HOLDER = "Your message"


# Cached shared objects
@st.cache_resource
def load_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, multi_process=False)
    return embeddings


@st.cache_resource
def get_embed_db():
    chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR")
    opensearch_url = os.getenv("OPENSEARCH_URL")
    if chroma_persist_dir:
        db = get_chroma_db(embeddings, chroma_persist_dir)
    elif opensearch_url:
        db = get_opensearch_db(embeddings, opensearch_url)
    else:
        # You can add additional vector stores here
        raise EnvironmentError("No vector store environment variables found.")
    return db


def get_chroma_db(embeddings, persist_dir):
    db = Chroma(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )
    return db


def get_opensearch_db(embeddings, url):
    username = os.getenv("OPENSEARCH_USERNAME")
    password = os.getenv("OPENSEARCH_PASSWORD")
    db = OpenSearchVectorSearch(
        embedding_function=embeddings,
        index_name=COLLECTION_NAME,
        opensearch_url=url,
        http_auth=(username, password),
        use_ssl=False,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
    )
    return db


embeddings = load_embeddings()
db = get_embed_db()


def save_message(role, content, sources=None):
    logger.info(f"message: {role} - '{content}'")
    msg = {"role": role, "content": content, "sources": sources}
    st.session_state["messages"].append(msg)
    return msg


def write_message(msg):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["sources"]:
            for doc in msg["sources"]:
                st.text(pprint.pformat(doc.metadata))
                st.write(doc.page_content)


st.title("Show Document Matches")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
    save_message(ANSWER_ROLE, FIRST_MESSAGE)

for msg in st.session_state["messages"]:
    write_message(msg)

if prompt := st.chat_input(PLACE_HOLDER):
    msg = save_message(QUESTION_ROLE, prompt)
    write_message(msg)

    docs_scores = db.similarity_search_with_score(prompt)
    docs = []
    for doc, score in docs_scores:
        doc.metadata["similarity_score"] = score
        docs.append(doc)

    msg = save_message(ANSWER_ROLE, "Matching Documents", docs)
    write_message(msg)
