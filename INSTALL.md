# Installation of RAG Example

This file describes the steps needed to run the retrieval-agumented
generation example code in this repository.

It is recommended to perform this installation in a virtual
environment rather than installed directly on a machine for all of the
standard reasonse.

The [`requirements.txt`](requirements.txt) file has been created without
pinned versions to aspirationally allow running against the latest
version of these libraries, but the actual versions used to create
and run these examples is shown here, just in case:

```
langchain==0.0.313
sentence-transformers==2.2.2
chromadb==0.4.14
streamlit==1.27.2
openai==0.28.1
```

The example should run on any version of Python 3 that is supported by
the underlying libraries, but it was tested on Python 3.11.4.

## Steps

1. Clone git repository

This can be done with either the HTTPS or SSH URL obtained by clicking
on the `<> Code` button on the `Code` tab of the repository in GitHub.

```
git clone https://github.com/harvard-hbs/rag-example.git
```

or

```
git clone git@github.com:harvard-hbs/rag-example.git
```

2. Install requirements

```
pip install -r requirements.txt
```

3. Configure and test LLM access


