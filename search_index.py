import streamlit as st
import pprint

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

from streamlit.logger import get_logger
logger = get_logger(__name__)

COLLECTION_NAME = "federalist_papers"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PERSIST_DIR = "doc_index"

ANSWER_ROLE = "Document Index"
FIRST_MESSAGE = "Enter text to find document matches."
QUESTION_ROLE = "Searcher"
PLACE_HOLDER = "Your message"

# Cached shared objects
@st.cache_resource
def load_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL)
    return embeddings

@st.cache_resource
def get_embed_db():
    db = Chroma(embedding_function=embeddings,
                collection_name=COLLECTION_NAME,
                persist_directory=PERSIST_DIR)
    return db

embeddings = load_embeddings()
db = get_embed_db()

def save_message(role, content, sources=None):
    logger.info(f"message: {role} - '{content}'")
    msg = {"role": role,
           "content": content,
           "sources": sources}
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
