import os
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()
import traceback
from groq import Groq
print(os.getenv("GROQ_API_KEY"))
##Read all the pdfs inside the directory
### Read all the pdf's inside the directory
def process_all_pdfs(pdf_directory):
    all_documents = []
    pdf_dir = Path(pdf_directory)  
    # Find all PDF files recursively
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            
            # Add source information to metadata
            for doc in documents:
                doc.metadata['source_file'] = pdf_file.name
                doc.metadata['file_type'] = 'pdf'
            
            all_documents.extend(documents)
            
        except Exception as e:
            print(f"  ✗ Error: {e}") 
    return all_documents

all_pdf_documents = process_all_pdfs("./data")
print(all_pdf_documents)


def split_documents(documents,chunk_size=300,chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    
    # Show example of a chunk
    # if split_docs:
    #     print(f"\nExample chunk:")
    #     print(f"Content: {split_docs[0].page_content[:200]}...")
    #     print(f"Metadata: {split_docs[0].metadata}")
    
    return split_docs
    
chunks=split_documents(all_pdf_documents)


class EmbeddingManager:
    
    def __init__(self, model_name: str = r"C:\models\all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
       
        if not self.model:
            raise ValueError("Model not loaded")
        
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings

class VectorStore:
    
    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "./data/vector_store"):
       
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        try:
            # Create persistent ChromaDB client
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings for RAG"}
            )
            
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
       
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        
        # Prepare data for ChromaDB
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate unique ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            
            # Prepare metadata
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)
            
            # Document content
            documents_text.append(doc.page_content)
            
            # Embedding
            embeddings_list.append(embedding.tolist())
        
        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
 
        except Exception as e:
            raise

embedding_manager=EmbeddingManager()
embedding_manager
vectorstore=VectorStore()
### Convert the text to embeddings
texts=[doc.page_content for doc in chunks]

## Generate the Embeddings

embeddings=embedding_manager.generate_embeddings(texts)

##store int he vector dtaabase
vectorstore.add_documents(chunks,embeddings)

class RAGRetriever:
    
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
     
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
       
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        # print(query_embedding)
        # Search in vector store
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            print("RES",results)

            
            # Process results
            retrieved_docs = []
            
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]
                
                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1 - distance
                    # if similarity_score >= score_threshold:
                    retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                    })
                    print("Retrivedfiles",retrieved_docs)
                
            else:
                print(retrieved_docs)
            return retrieved_docs
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []

rag_retriever=RAGRetriever(vectorstore,embedding_manager)
data=rag_retriever.retrieve("What year did the candidate completed his B. Tech Computer Science and Engineering?")
print("data",data)

class GroqLLM:
    def __init__(self, model_name: str = "llama-3.1-8b-instant", api_key: str =None):
        """
        Initialize Groq LLM
        
        Args:
            model_name: Groq model name (qwen2-72b-instruct, llama3-70b-8192, etc.)
            api_key: Groq API key (or set GROQ_API_KEY environment variable)
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable or pass api_key parameter.")
        
        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model_name,
            temperature=0.1,
            max_tokens=1024
        )
        
        print(f"Initialized Groq LLM with model: {self.model_name}")

    def generate_response(self, query: str, context: str, max_length: int = 500) -> str:
        """
        Generate response using retrieved context
        
        Args:
            query: User question
            context: Retrieved document context
            max_length: Maximum response length
            
        Returns:
            Generated response string
        """
        
        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful AI assistant. Use the following context to answer the question accurately and concisely.
            Context:
            {context}
            Question: {question}
            Answer: Provide a clear and informative answer based on the context above. If the context doesn't contain enough information to answer the question, say so."""
        )
        
        # Format the prompt
        formatted_prompt = prompt_template.format(context=context, question=query)
        
        try:
            # Generate response
            messages = [HumanMessage(content=formatted_prompt)]
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
        
    def generate_response_simple(self, query: str, context: str) -> str:
        """
        Simple response generation without complex prompting
        
        Args:
            query: User question
            context: Retrieved context
            
        Returns:
            Generated response
        """
        simple_prompt = f"""Based on this context: {context}

        Question: {query}

        Answer:"""
        
        try:
            messages = [HumanMessage(content=simple_prompt)]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"

# Initialize Groq LLM (you'll need to set GROQ_API_KEY environment variable)
# try:
#     groq_llm = GroqLLM(api_key=os.getenv("GROQ_API_KEY"))
#     print("Groq LLM initialized successfully!")
# except ValueError as e:
#     print(f"Warning: {e}")
#     print("Please set your GROQ_API_KEY environment variable to use the LLM.")
#     groq_llm = None

# rag_retriever.retrieve("what is the overall experience of candidate?")



client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def ask_llm(prompt: str) -> str:
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="llama-3.1-8b-instant",
    )
    print("cccc",chat_completion)
    return chat_completion.choices[0].message.content



### Initialize the Groq LLM (set your GROQ_API_KEY in environment)
def rag_simple(query, retriever, top_k=3):

    results = retriever.retrieve(query, top_k=top_k)
    context = "\n\n".join([doc["content"] for doc in results]) if results else ""

    if not context:
        return "No relevant context found to answer the question"

    prompt = f"""

You are a helpful AI assistant.

Use the provided context to answer the question.

Guidelines:
- Prefer using the context
- Give exact answers when possible
- If multiple answers exist, choose the most relevant
- Avoid making up information
- Keep answers clear and concise

Context:
{context}

Question:
{query}

Answer:
"""

    return ask_llm(prompt)


answer = rag_simple("When does candidate completed his B.Tech?", rag_retriever)
print(answer)
