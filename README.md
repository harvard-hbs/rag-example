# Retrieval-Augmented Generation (RAG) Example

This repository contains an example of code for demonstrating
retrieval-augmented generation (RAG) which provides a mechanism for
incorporating domain-specific content into generative AI interactions
with large language models (LLMs).

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
- Store the chunks indexed by embedding into a persistent database for
  retrieval.

This example uses a PDF version of The Federalist Papers as source
document, but there are additional langchain document ingesting tools
for Word documents and many other document types. You can directly
use the langchain `Document` object inside your custom text
ingestion code.

Document ingesting tools can and do store metadata along with each
chunk of text in the langchain `Document` object. In the case of the
`PyPDFLOader` it saves the name of the PDF file and the page number
of the chunk within the PDF file. You can add your own metadata
to document chunks before you persist them.

It is not necessary to use the same language model for generating
vector embedding indexes that is used for generating responses. In our
case we are using a much small `all-MiniLM-L6-v2` model from
HuggingFace to generate the embedding for indexing. It **is**
necessary to use the same embeddings model for the retrieval part of
the process as was used to create the persisted vector embeddings, so
you will see this same model used in the Testing Retrieval section and
in the full retrieval chain and chatbot.

The code for converting the PDF document into text and breaking it
into page-level chunks is in `index_documents.py`:

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

Running the `index_documents.py` program will create the `doc_index`
directory with persisted vector embeddings. These embeddings are
already created in this repository, but can be deleted and re-created
if you want to index a different set of documents.

```
python index_documents.py
```

## Testing Retrieval

The eventual goal is to have the indexed documents searched as part of
the the LLM interaction, but you can also test how various queries
match against your document store. Here is some example code:

```
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import pprint

COLLECTION_NAME = "federalist_papers"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PERSIST_DIR = "doc_index"

# Same model as used to create persisted embedding index
embeddings = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL)

# Access persisted embeddings
db = Chroma(embedding_function=embeddings,
                collection_name=COLLECTION_NAME,
                persist_directory=PERSIST_DIR)
                
# Example query to for similarity indexing
prompt = "How should government responsibility be divided between the states and the federal government?"

# Display matched documents and similarity scores
docs_scores = db.similarity_seardch_with_score(prompt)
for doc, score in docs_scores:
    print(f"similarity_score: {score}")
    pprint.pprint(doc)
```

This same approach is used in the `streamlit` user interface in
`search_index.py` that can be run to graphically see the matched
documents.

```
streamlit run search_index.py
```

<kbd>![Image of document matching UI](images/search_index_image.png)</kbd>

## Creating the full chain

The chain for interaction with the LLM has the following pieces:

- A `ConversationalRetrievalChain` which connects to the persisted vector
  database index and looks up document chunks to pass to the LLM with the
  appropriate prompting.
- A 'ConversationalBufferWindowMemory` that provides a level of memory so
  the chatbot can refer to earlier parts of the conversation.
- The LLM chat interface, `AzureChatOpenAI` in our case.

```
# Access persisted embeddings and expose through langchain retriever
embeddings = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL)
db = Chroma(embedding_function=embeddings,
                collection_name=COLLECTION_NAME,
                persist_directory=PERSIST_DIR)
retriever = db.as_retriever()

# This can be any LLM supported by LangChain
llm = AzureChatOpenAI(temperature=0.5,
                      deployment_name=model_name,
                      verbose=VERBOSE)

# Establish a memory buffer for conversational continuity
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    output_key="answer",
    return_messages=True,
    window_size=MEMORY_WINDOW_SIZE)

# Put together all of the components into the full
# chain with memory and retrieval-agumented generation
query_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=memory,
    retriever=retriever,
    verbose=VERBOSE,
    return_source_documents=True)

prompt = "How should government responsibility be divided between the states and the federal government?"
query_response = query_chain({"queston": prompt})
```

## Streamlit Chat UI

The chatbot demonstration user interface uses the
[Streamlit](https://streamlit.io/) framework, the `st.chat_message`
and `st.chat_input` components, and its ability to support multi-user
sessions to interact with the conversational retrieval chain. 

```
streamlit run document_chatbot.py
```

<kbd>![Image of document chatbot UI](images/document_chatbot_image.png)</kbd>


