# Build BioMistral Medical RAG Chatbot using BioMistral Open Source LLM 
# Load the google drive

from google.colab import drive 
drive.mount("/content/drive")

# Installation

!pip install langchain sentence-transformers chromadb llama-cpp-python langchain_community pypdf

# Importing libraries

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA, LLMChain

# Import the document

loader = PyPDFDirectoryLoader("/content/drive/MyDrive/BioMistral/Data")
docs = loader.load()

# Chunking

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

len(chunks)

# Embeddings creations

import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"
embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

# Vector Store creation

vectorstore = Chroma.from_documents(chunks, embeddings)

# LLM Model loading

llm = LlamaCpp(
    model_path="/content/drive/MyDrive/BioMistral/BioMistral-78.04K.M.gguf",
    temperature=0.2,
    max_tokens=2048,
    top_p=1
)

# Use LLM and retriever and query to generate final response

template = """
<|context|>
You are a Medical Assistant that follows the instructions and generates an accurate response based on the query and the context provided.
Please be truthful and give direct answers.
</|context|>
<|user|>
{query}
</|user|>
<|assistant|>
"""

from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {
        "context": vectorstore.as_retriever(),
        "query": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke(query)

import sys
while True:
    user_input = input("Input query: ")
    if user_input.lower() == 'exit':
        print("Exiting...")
        sys.exit()
    if user_input == "":
        continue
    result = rag_chain.invoke(user_input)
    print("Answer:", result)
