# Build BioMistral Medical RAG Chatbot using BioMistral Open Source LLM 
# 🩺 Medical Chatbot using RAG (Retrieval-Augmented Generation)

This project is a Medical Chatbot designed using Retrieval-Augmented Generation (RAG) architecture. The chatbot utilizes advanced machine learning models, including large language models (LLMs) and embeddings, to provide accurate and relevant responses based on the provided context. The chatbot processes medical documents and allows users to ask health-related questions. 


<img width="45%" src="https://thumbs.dreamstime.com/b/chatbot-healthcare-abstract-concept-vector-illustration-artificial-intelligence-caregiver-chatbot-healthcare-use-anonymous-214748027.jpg"> 

---

### 📄 Data Source: 
**HealthyHeart.pdf**  
[Healthy Heart PDF](https://www.nhlbi.nih.gov/files/docs/public/heart/healthyheart.pdf)

---

## Frameworks and Technologies Used:

- **Langchain**: <u>Pipeline</u> management and framework integration.
- **Llama**: <u>Large Language Model (LLM)</u> used for natural language understanding.
- **Sentence-Transformers**: <u>Embedding model</u> to generate dense representations for each document chunk.
- **Chroma**: <u>Vector store</u> for storing and retrieving document embeddings efficiently.

---

## LLM Model:
**BioMistral-7B**  
[BioMistral-7B-GGUF Model](https://huggingface.co/MazivarPanahi/BioMistral-7B-GGUF/tree/main)

---

##  Embeddings Model:
**PubMedBert-Base-embeddings**  
[PubMedBert Embeddings](https://huggingface.co/NeuML/pubmedbert-base-embeddings)

---

## Process Overview:

1. **Load the Document**: The chatbot loads and parses the `HealthyHeart.pdf` document.
2. **Chunking**: The document is split into smaller, manageable chunks.
3. **Embedding Creation**: Each chunk is transformed into dense embedding vectors using Sentence-Transformers.
4. **Vector Store**: Chunks and their embeddings are stored in Chroma Vector Store for efficient retrieval.
5. **LLM Model Loading**: The BioMistral-7B model is loaded to handle natural language understanding.
6. **Application Chain**: The full application chain is built end-to-end.
7. **Query the Chatbot**: Users can input queries to interact with the chatbot.
8. **Retriever**: The chatbot retrieves relevant document chunks from the Vector Store using K-Nearest Neighbors (KNN) search.
9. **LLM Interaction**: Both the query and relevant documents are passed to the LLM.
10. **Response Generation**: The chatbot generates responses based on the query and document context.

---

## 📚 Resources:

- ➡️ [Langchain Documentation](https://langchain.com/docs)
- ➡️ [BioMistral-7B Model](https://huggingface.co/MazivarPanahi/BioMistral-7B-GGUF/tree/main)
- ➡️ [PubMedBert Embeddings](https://huggingface.co/NeuML/pubmedbert-base-embeddings)
