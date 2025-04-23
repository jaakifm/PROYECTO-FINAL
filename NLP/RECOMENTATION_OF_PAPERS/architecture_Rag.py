# RAG System for Scientific Articles on Melanoma with Streamlit Interface and Local LLM
# Requirements: pip install pypdf langchain sentence-transformers faiss-cpu streamlit llama-cpp-python

import os
import re
import numpy as np
import tempfile
from typing import List, Dict, Tuple

# For the interface
import streamlit as st

# For processing PDFs
from pypdf import PdfReader

# For splitting text into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

# For creating embeddings
from sentence_transformers import SentenceTransformer

# For vector search
import faiss

# For running the local LLM
from llama_cpp import Llama


class MelanomaRAGSystem:
    def __init__(self, model_name="all-MiniLM-L6-v2", llm_path=None):
        """
        Initialize the RAG system for melanoma.
        
        Args:
            model_name: Embedding model to use
            llm_path: Path to the local LLM model
        """
        self.model = SentenceTransformer(model_name)
        self.chunks = []
        self.embeddings = None
        self.index = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        # To store filenames and their metadata
        self.doc_metadata = {}
        
        # Initialize the local LLM if a path is provided
        self.llm = None
        self.llm_path = llm_path
        if llm_path and os.path.exists(llm_path):
            try:
                with st.spinner("Loading local LLM model... This may take a moment."):
                    self.llm = Llama(
                        model_path=llm_path,
                        n_ctx=4096,         # Context window size
                        n_threads=4,        # Number of CPU threads to use
                        n_gpu_layers=0      # Set higher if you have a GPU
                    )
                st.sidebar.success("✅ Local LLM loaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Failed to load local LLM: {str(e)}")
                st.sidebar.warning("The system will fall back to basic text retrieval.")

    def extract_text_from_pdf(self, pdf_file, filename: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_file: PDF file (BytesIO)
            filename: Filename
            
        Returns:
            Extracted text from PDF
        """
        reader = PdfReader(pdf_file)
        text = ""
        
        for page in reader.pages:
            text += page.extract_text() + "\n"
            
        # Basic text cleaning
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with one
        
        return text
    
    def process_uploaded_files(self, uploaded_files) -> None:
        """
        Process multiple uploaded PDF documents and create the search index.
        
        Args:
            uploaded_files: List of files uploaded through Streamlit
        """
        all_chunks = []
        
        for uploaded_file in uploaded_files:
            try:
                # Extract text from PDF
                text = self.extract_text_from_pdf(uploaded_file, uploaded_file.name)
                
                # Split text into chunks
                chunks = self.text_splitter.split_text(text)
                
                # Add metadata (document source)
                doc_chunks = [
                    {"content": chunk, "source": uploaded_file.name}
                    for chunk in chunks
                ]
                
                all_chunks.extend(doc_chunks)
                
                # Save document information
                self.doc_metadata[uploaded_file.name] = {
                    "total_chunks": len(chunks),
                    "size": uploaded_file.size,
                    "type": uploaded_file.type
                }
                
                st.sidebar.success(f"Processed: {uploaded_file.name} - {len(chunks)} chunks extracted")
                
            except Exception as e:
                st.sidebar.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        self.chunks.extend(all_chunks)
        
        # Create embeddings and search index
        self._create_index()
        
        st.sidebar.success(f"Processing complete: {len(self.chunks)} total chunks")
    
    def _create_index(self) -> None:
        """
        Create embeddings for all chunks and build the FAISS index.
        """
        if not self.chunks:
            st.warning("No chunks to index")
            return
            
        # Extract only the content of the chunks to create embeddings
        texts = [chunk["content"] for chunk in self.chunks]
        
        with st.spinner('Creating embeddings... This may take a moment'):
            # Create embeddings
            self.embeddings = self.model.encode(texts)
            
            # Normalize embeddings for cosine similarity search
            faiss.normalize_L2(self.embeddings)
            
            # Create FAISS index
            vector_dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(vector_dimension)  # Inner product index (cosine similarity)
            self.index.add(self.embeddings)
        
        st.sidebar.info(f"Index created with dimension {vector_dimension}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for the most relevant chunks for a query.
        
        Args:
            query: User query
            top_k: Number of results to return
            
        Returns:
            List of most relevant chunks with their scores
        """
        if not self.index:
            st.warning("Index has not been created. Upload and process documents first.")
            return []
            
        # Create embedding for the query
        query_embedding = self.model.encode([query])
        
        # Normalize for cosine similarity search
        faiss.normalize_L2(query_embedding)
        
        # Search the index
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # -1 indicates that not enough results were found
                results.append({
                    "content": self.chunks[idx]["content"],
                    "source": self.chunks[idx]["source"],
                    "score": float(scores[0][i])
                })
                
        return results
    
    def answer_query(self, query: str, top_k: int = 5, use_llm: bool = True) -> Dict:
        """
        Answer a query based on retrieved documents.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            use_llm: Whether to use the LLM for answer generation
            
        Returns:
            Dictionary with answer and context
        """
        # Retrieve relevant documents
        relevant_docs = self.search(query, top_k=top_k)
        
        if not relevant_docs:
            return {
                "answer": "No relevant information found for this query.",
                "sources": [],
                "context": []
            }
        
        # Extract unique sources
        sources = list(set([doc["source"] for doc in relevant_docs]))
        
        # Build context
        context = [f"{i+1}. {doc['content']} (Score: {doc['score']:.4f})" for i, doc in enumerate(relevant_docs)]
        raw_context = [doc["content"] for doc in relevant_docs]
        
        # Generate answer
        if use_llm and self.llm is not None:
            try:
                # Construct the prompt for the LLM
                combined_context = "\n\n".join(raw_context)
                
                prompt = f"""You are a helpful medical AI assistant specializing in melanoma research. 
Use only the information provided in the context to answer the question. 
If the context doesn't contain enough information to answer the question fully, 
acknowledge the limitations of your knowledge.

Context:
{combined_context}

Question: {query}

Answer:"""
                
                # Generate response with the local LLM
                with st.spinner("Generating answer with local LLM..."):
                    llm_response = self.llm(
                        prompt,
                        max_tokens=1024,
                        stop=["Question:", "Context:"],
                        echo=False
                    )
                
                answer = llm_response["choices"][0]["text"].strip()
                is_llm_generated = True
                
            except Exception as e:
                # Fallback to basic concatenation if LLM fails
                st.warning(f"LLM generation failed: {str(e)}. Falling back to basic retrieval.")
                answer = (
                    f"Based on the consulted documents, we found the following relevant information "
                    f"about '{query}':\n\n" + "\n\n".join(raw_context)
                )
                is_llm_generated = False
        else:
            # Basic concatenation of retrieved content
            answer = (
                f"Based on the consulted documents, we found the following relevant information "
                f"about '{query}':\n\n" + "\n\n".join(raw_context)
            )
            is_llm_generated = False
        
        return {
            "answer": answer,
            "sources": sources,
            "context": context,
            "is_llm_generated": is_llm_generated
        }

    def extract_melanoma_terms(self, result_context: List[str]) -> List[str]:
        """
        Extract specific terms related to melanoma from the retrieved context.
        
        Args:
            result_context: List of text fragments from the retrieved context
            
        Returns:
            List of specific melanoma terms
        """
        # List of common terms related to melanoma
        # In a real implementation, this could be much more sophisticated
        melanoma_terms = [
            "melanoma", "nevus", "ABCDE", "Breslow", "Clark", "metastasis", 
            "melanocyte", "melanin", "nodular", "lentigo maligna", "acral", 
            "BRAF", "immunotherapy", "staging", "dermatoscopy", "biopsy",
            "AJCC", "TNM", "mitosis", "ulceration", "regression", "sentinel",
            "dermoscopy", "mitotic index", "micrometastasis", "PD-1",
            "PD-L1", "CTLA-4", "epithelioid", "spindle-shaped", "MAPK", "MEK",
            "radiotherapy", "chemotherapy", "adjuvant therapy"
        ]
        
        # Search for terms in the context
        found_terms = set()
        term_contexts = {}
        
        for fragment in result_context:
            fragment_lower = fragment.lower()
            for term in melanoma_terms:
                if term.lower() in fragment_lower:
                    found_terms.add(term)
                    # Capture a context phrase for the term
                    term_index = fragment_lower.find(term.lower())
                    start = max(0, term_index - 50)
                    end = min(len(fragment), term_index + len(term) + 50)
                    context_phrase = fragment[start:end].strip()
                    term_contexts[term] = context_phrase
        
        # Sort terms alphabetically
        found_terms = sorted(list(found_terms))
        
        return found_terms, term_contexts

    def suggest_readings(self, query: str, top_k: int = 3) -> List[str]:
        """
        Suggest papers for reading based on the query.
        
        Args:
            query: User query
            top_k: Maximum number of articles to suggest
            
        Returns:
            List of suggested sources
        """
        relevant_docs = self.search(query, top_k=top_k*2)  # Search for more to get variety
        
        # Extract unique sources
        sources = list(set([doc["source"] for doc in relevant_docs]))
        
        # Limit to the requested number
        return sources[:top_k]


def main():
    st.set_page_config(
        page_title="Melanoma RAG System",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 RAG System for Scientific Articles on Melanoma")
    st.markdown("""
    This system allows you to extract information from PDFs of scientific articles on melanoma 
    and query specific information using a local LLM to analyze and respond to your questions.
    """)
    
    # LLM model path
    default_llm_path = r"C:\Users\jakif\.lmstudio\models\lmstudio-community\DeepSeek-R1-Distill-Llama-8B-GGUF\DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
    
    # Initialize the RAG system in the session state
    if 'melanoma_rag' not in st.session_state:
        # Sidebar for LLM model settings
        with st.sidebar:
            st.header("🧠 LLM Model Settings")
            use_custom_model = st.checkbox("Use custom LLM model path", value=False)
            
            if use_custom_model:
                llm_path = st.text_input("Path to local LLM model (GGUF format)", value=default_llm_path)
            else:
                llm_path = default_llm_path
            
            # Check if model exists
            if not os.path.exists(llm_path):
                st.error(f"Model not found at: {llm_path}")
                st.info("The system will run without LLM capabilities.")
                llm_path = None
        
        st.session_state.melanoma_rag = MelanomaRAGSystem(llm_path=llm_path)
        st.session_state.uploaded_files = []
        st.session_state.has_processed = False
        st.session_state.query_history = []
        st.session_state.llm_path = llm_path
    
    # Sidebar for uploading files
    st.sidebar.title("📁 Upload Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDFs of scientific articles",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    # If new files have been uploaded
    if uploaded_files and uploaded_files != st.session_state.uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.session_state.has_processed = False
    
    # Button to process files
    if st.sidebar.button("Process Documents") and st.session_state.uploaded_files:
        st.session_state.melanoma_rag.process_uploaded_files(st.session_state.uploaded_files)
        st.session_state.has_processed = True
    
    # Show statistics of processed documents
    if st.session_state.has_processed:
        st.sidebar.header("📊 Statistics")
        st.sidebar.info(f"Total documents: {len(st.session_state.melanoma_rag.doc_metadata)}")
        st.sidebar.info(f"Total chunks: {len(st.session_state.melanoma_rag.chunks)}")
        
        # List of documents
        st.sidebar.header("📑 Processed Documents")
        for doc_name, metadata in st.session_state.melanoma_rag.doc_metadata.items():
            st.sidebar.markdown(f"**{doc_name}**")
            st.sidebar.markdown(f"- Chunks: {metadata['total_chunks']}")
            st.sidebar.markdown(f"- Size: {metadata['size']/1024:.2f} KB")
    
    # Main area
    if st.session_state.has_processed:
        # Query
        st.header("🔎 Make a Query")
        query = st.text_input("What would you like to know about melanoma?", 
                             placeholder="E.g.: What are the risk factors for melanoma?")
        
        top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=5)
        
        # Option to use LLM or not
        use_llm = st.checkbox("Use Local LLM to generate response", value=True)
        if use_llm and st.session_state.melanoma_rag.llm is None:
            st.warning("Local LLM not available. Response will use basic text retrieval.")
        
        col1, col2 = st.columns([1, 3])
        search_button = col1.button("Search")
        clear_button = col2.button("Clear results")
        
        if clear_button:
            st.session_state.query_history = []
            st.experimental_rerun()
        
        # Perform search
        if search_button and query:
            with st.spinner('Querying documents...'):
                result = st.session_state.melanoma_rag.answer_query(
                    query=query, 
                    top_k=top_k,
                    use_llm=use_llm
                )
                terms, term_contexts = st.session_state.melanoma_rag.extract_melanoma_terms(result["context"])
                suggested_readings = st.session_state.melanoma_rag.suggest_readings(query)
                
                # Save to history
                st.session_state.query_history.append({
                    "query": query,
                    "result": result,
                    "terms": terms,
                    "term_contexts": term_contexts,
                    "readings": suggested_readings
                })
        
        # Show results from history
        if st.session_state.query_history:
            st.header("📝 Results")
            
            # Create tabs for each query
            tabs = st.tabs([f"Query: {item['query'][:20]}..." for item in reversed(st.session_state.query_history)])
            
            for i, (tab, item) in enumerate(zip(tabs, reversed(st.session_state.query_history))):
                with tab:
                    st.subheader("Query")
                    st.write(item['query'])
                    
                    st.subheader("Answer")
                    if item['result'].get('is_llm_generated', False):
                        st.markdown("*Response generated by local LLM*")
                    else:
                        st.markdown("*Basic retrieval response*")
                    st.write(item['result']['answer'])
                    
                    st.subheader("Sources")
                    for source in item['result']['sources']:
                        st.markdown(f"- {source}")
                    
                    # Show relevant terms (without nested expanders)
                    if item['terms']:
                        st.subheader("📚 Specific Terms Found")
                        terms_table = []
                        for term in item['terms']:
                            terms_table.append({
                                "Term": term,
                                "Context": item['term_contexts'].get(term, 'Not available')
                            })
                        st.table(terms_table)
                    
                    # Recommended readings
                    st.subheader("📖 Recommended Readings")
                    for reading in item['readings']:
                        st.markdown(f"- {reading}")
                    
                    # Show detailed results
                    with st.expander("View full context"):
                        for ctx in item['result']['context']:
                            st.markdown(f"**{ctx}**")
                            st.markdown("---")
    else:
        # Show message if no documents have been processed
        st.info("👈 Please upload some PDF documents and click 'Process Documents' to begin.")
        
        # Example usage
        st.header("🧪 Usage Example")
        st.markdown("""
        1. Upload PDFs of scientific articles on melanoma using the sidebar.
        2. Click "Process Documents" to index the content.
        3. Type a query like "What are the prognostic factors for melanoma?"
        4. Toggle the "Use Local LLM" option to switch between basic retrieval and LLM-generated responses.
        5. Explore the results, specific terms, and recommended readings.
        """)


if __name__ == "__main__":
    main()