import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
import os
import traceback
import tempfile
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torchvision import models
from llama_cpp import Llama
from audio_recorder_streamlit import audio_recorder  # Importamos el grabador de audio
import whisper
#librerias extra para el rag
import re
from typing import List, Dict, Tuple
from pypdf import PdfReader
# For splitting text into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

# For creating embeddings
from sentence_transformers import SentenceTransformer

# For vector search
import faiss


# Page configuration
st.set_page_config(
    page_title="Melanoma Multi-Modal Classifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

#Introduccion RAG

def clean_llm_output(text):
    """
    Clean the LLM output to remove duplicate content and internal tokens.
    
    Args:
        text: Raw text from LLM
        
    Returns:
        Cleaned text
    """
    # Remove any <think> or </think> tags and content between them
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # Remove standalone </think> tags
    text = re.sub(r'</think>', '', text)
    
    # Remove any userStyle tags
    text = re.sub(r'<userStyle>.*?</userStyle>', '', text, flags=re.DOTALL)
    
    # Find duplicate paragraphs (common in some LLM outputs)
    paragraphs = text.split('\n\n')
    unique_paragraphs = []
    
    for p in paragraphs:
        # Clean and standardize for comparison
        cleaned_p = p.strip()
        if cleaned_p and cleaned_p not in unique_paragraphs:
            unique_paragraphs.append(cleaned_p)
    
    # Reassemble the text
    cleaned_text = '\n\n'.join(unique_paragraphs)
    
    # Additional cleanup for any remaining artifacts
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Normalize whitespace
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text


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
                
                st.sidebar.success(f"Processed: {uploaded_file.name} ")
                
            except Exception as e:
                st.sidebar.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        self.chunks.extend(all_chunks)
        
        # Create embeddings and search index
        self._create_index()
        
        st.sidebar.success(f"Processing complete")
    
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
                
                prompt = f"""You are a precision-focused melanoma research AI assistant. Follow these rules strictly:

1. SOURCING & NON-REPETITION:
- Use ONLY information from the provided context, marked with "Context:" 
- Never repeat the same fact, even if rephrased
- For repeated queries, respond: "Per previous context: [1-sentence summary]"
- Combine duplicate facts across context documents into one definitive statement

2. RESPONSE STRUCTURE:
Context:
- [Relevant source name/location]
- Key finding: [Exact statistic/claim]
- Gap: [Missing information]

[Non-context]: [ONLY if critical for safety/understanding]

3. PROHIBITED:
- No general knowledge unless labeled [Non-context]
- No rephrasing of already stated facts
- No "in conclusion" or summary unless new context exists

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
                
                raw_answer = llm_response["choices"][0]["text"].strip()
                
                # Clean up the response
                answer = clean_llm_output(raw_answer)
                
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






# Function to transcribe audio with Whisper - load model only when needed
def transcribe_with_whisper(audio_path):
    """Load whisper model and transcribe audio file"""
    # Import whisper here to avoid early initialization issues
    
    
    # Load model when function is called
    model = whisper.load_model("base")  # Options: "tiny", "base", "small", "medium", "large"
    
    # Transcribe
    result = model.transcribe(audio_path)
    return result["text"].strip()

# Function to process audio and get transcription
def process_audio_to_text(audio_bytes):
    """Process audio bytes and return transcribed text"""
    if not audio_bytes:
        return None
        
    # Save audio bytes to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio_path = temp_audio.name
    
    try:
        # Transcribe with Whisper
        with st.spinner("Transcribing your response..."):
            transcription = transcribe_with_whisper(temp_audio_path)
        return transcription
    except Exception as e:
        st.error(f"Transcription error: {e}")
        st.error(traceback.format_exc())
        return None
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

# Function to load data with better error handling
@st.cache_data
def load_data():
    try:
        # Check if file exists
        if not os.path.exists('dataset_answers.json'):
            # If not, create sample data
            st.warning("File 'dataset_answers.json' not found. Creating sample data.")
            
            # Sample data
            sample_data = {
                "name": "Melanoma Severity Classification",
                "description": "Training data for melanoma severity classification",
                "data": [
                    {"text": "it's grown a lot", "label": "highly_concerning"},
                    {"text": "it has grown somewhat", "label": "moderately_concerning"},
                    {"text": "I think it might be slightly larger", "label": "mildly_concerning"},
                    {"text": "no", "label": "not_concerning"}
                ]
            }
            
            # Save sample data
            with open('melanoma_data.json', 'w') as f:
                json.dump(sample_data, f, indent=2)
            
            return sample_data
        
        # Read the file
        with open('dataset_answers.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    except json.JSONDecodeError:
        st.error("Error decoding JSON file. The file might be corrupted.")
        # Create minimal structure to avoid errors
        return {
            "name": "Data Error",
            "description": "Could not load data correctly",
            "data": []
        }
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.error(traceback.format_exc())
        # Create minimal structure to avoid errors
        return {
            "name": "Data Error",
            "description": "Could not load data correctly",
            "data": []
        }

# Function to get labels and their IDs
@st.cache_data
def get_labels(data):
    try:
        if not data or "data" not in data or not data["data"]:
            # If no data, use default labels
            default_labels = ["not_concerning", "mildly_concerning", "moderately_concerning", "highly_concerning"]
            label_to_id = {label: idx for idx, label in enumerate(default_labels)}
            id_to_label = {idx: label for idx, label in enumerate(default_labels)}
            return default_labels, label_to_id, id_to_label
        
        labels = [item["label"] for item in data["data"]]
        unique_labels = sorted(set(labels))
        
        # If no labels, use default labels
        if not unique_labels:
            default_labels = ["not_concerning", "mildly_concerning", "moderately_concerning", "highly_concerning"]
            label_to_id = {label: idx for idx, label in enumerate(default_labels)}
            id_to_label = {idx: label for idx, label in enumerate(default_labels)}
            return default_labels, label_to_id, id_to_label
        
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        id_to_label = {idx: label for idx, label in enumerate(unique_labels)}
        return unique_labels, label_to_id, id_to_label
    except Exception as e:
        st.error(f"Error processing labels: {str(e)}")
        # Default labels in case of error
        default_labels = ["not_concerning", "mildly_concerning", "moderately_concerning", "highly_concerning"]
        label_to_id = {label: idx for idx, label in enumerate(default_labels)}
        id_to_label = {idx: label for idx, label in enumerate(default_labels)}
        return default_labels, label_to_id, id_to_label

# Function to load text model and tokenizer
@st.cache_resource
def load_text_model(label_to_id, id_to_label):
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("rjac/biobert-ICD10-L3-mimic")
        
        # Correct label configuration
        num_labels = len(label_to_id)
        id2label = {str(i): label for i, label in id_to_label.items()}
        label2id = {label: str(i) for label, i in label_to_id.items()}
        
        # Check if fine-tuned model exists
        if not os.path.exists("./finetuned_model"):
            st.error("Fine-tuned text model not found. Please make sure the model is available at './finetuned_model'.")
            return None, None
        
        # Load fine-tuned model
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                "./finetuned_model",
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id
            )
            return tokenizer, model
        except Exception as e:
            st.error(f"Could not load fine-tuned text model: {str(e)}")
            st.error(traceback.format_exc())
            return None, None
    except Exception as e:
        st.error(f"Error loading text model: {str(e)}")
        st.error(traceback.format_exc())
        return None, None

# Function to load vision model
@st.cache_resource
def load_vision_model():
    try:
        # Update path to the optimized ensemble model
        model_path = "C:/Users/jakif/CODE/PROYECTO-FINAL/COMPUTER_VISION/Ensamblado_mejores_modelos/optimized_ensemble_model.pth"
        
        # Check if vision model exists
        if not os.path.exists(model_path):
            st.error(f"Optimized ensemble model not found. Please make sure the model is available at '{model_path}'.")
            return None
        
        # Define the paths to the base models needed for the ensemble
        vit_model_path = "C:/Users/jakif/CODE/PROYECTO-FINAL/COMPUTER_VISION/vision_transformers/best_vit_model.pth"
        efficient_model_path = "C:/Users/jakif/CODE/PROYECTO-FINAL/COMPUTER_VISION/melanoma_model_1_torch_EFFICIENTNETB0_harvard_attention.pth"
        
        if not os.path.exists(vit_model_path) or not os.path.exists(efficient_model_path):
            st.error(f"Base models not found. Make sure ViT and EfficientNet models are available.")
            return None
        
        # Define the necessary model architecture classes
        class SEBlock(torch.nn.Module):
            def __init__(self, channel, reduction=16):
                super(SEBlock, self).__init__()
                self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
                self.fc = torch.nn.Sequential(
                    torch.nn.Linear(channel, channel // reduction, bias=False),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(channel // reduction, channel, bias=False),
                    torch.nn.Sigmoid()
                )

            def forward(self, x):
                b, c, _, _ = x.size()
                y = self.avg_pool(x).view(b, c)
                y = self.fc(y).view(b, c, 1, 1)
                return x * y.expand_as(x)
        
        class SEEfficientNet(torch.nn.Module):
            def __init__(self, num_classes=1):
                super(SEEfficientNet, self).__init__()
                # Load base model
                self.base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
                
                # Get feature layer
                self.features = self.base_model.features
                
                # Add SE blocks
                for i in range(len(self.features)):
                    if hasattr(self.features[i], 'block'):
                        channels = self.features[i]._blocks[-1].project_conv.out_channels
                        se_block = SEBlock(channels)
                        setattr(self, f'se_block_{i}', se_block)
                
                # Classification layer
                self.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(p=0.3),
                    torch.nn.Linear(in_features=1280, out_features=512),
                    torch.nn.BatchNorm1d(512),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(p=0.4),
                    torch.nn.Linear(in_features=512, out_features=128),
                    torch.nn.BatchNorm1d(128),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(p=0.3),
                    torch.nn.Linear(in_features=128, out_features=num_classes)
                )
            
            def forward(self, x):
                for i in range(len(self.features)):
                    x = self.features[i](x)
                    if hasattr(self, f'se_block_{i}'):
                        se_block = getattr(self, f'se_block_{i}')
                        x = se_block(x)
                
                x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
                x = torch.flatten(x, 1)
                
                return self.classifier(x)

        # Function to create the ViT model
        def create_vit_classifier(num_classes=2, dropout_rate=0.2):
            try:
                import timm
                model = timm.create_model('vit_large_patch16_224', pretrained=True, drop_rate=dropout_rate)
                in_features = model.head.in_features
                model.head = torch.nn.Sequential(
                    torch.nn.Dropout(dropout_rate),
                    torch.nn.Linear(in_features, num_classes)
                )
                return model
            except ImportError:
                st.error("Could not load the ViT model. The timm library is not installed.")
                return None

        # Define the ensemble model
        class EnsembleModel(torch.nn.Module):
            def __init__(self, vit_model, efficientnet_model, vit_weight=0.6, efficientnet_weight=0.4):
                super(EnsembleModel, self).__init__()
                self.vit_model = vit_model
                self.efficientnet_model = efficientnet_model
                self.vit_weight = vit_weight
                self.efficientnet_weight = efficientnet_weight
                
            def forward(self, x):
                vit_output = self.vit_model(x)
                efficientnet_output = self.efficientnet_model(x)
                
                # Adjust dimensions
                if vit_output.shape[1] > 1 and efficientnet_output.shape[1] == 1:
                    eff_sigmoid = torch.sigmoid(efficientnet_output).view(-1, 1)
                    efficientnet_probs = torch.cat([1 - eff_sigmoid, eff_sigmoid], dim=1)
                    vit_probs = torch.nn.functional.softmax(vit_output, dim=1)
                elif vit_output.shape[1] == 1 and efficientnet_output.shape[1] == 1:
                    vit_probs = torch.sigmoid(vit_output)
                    efficientnet_probs = torch.sigmoid(efficientnet_output)
                else:
                    vit_probs = torch.nn.functional.softmax(vit_output, dim=1)
                    efficientnet_probs = torch.nn.functional.softmax(efficientnet_output, dim=1)
                
                # Combine with weights
                ensemble_output = (self.vit_weight * vit_probs + 
                                  self.efficientnet_weight * efficientnet_probs)
                
                return ensemble_output
        
        try:
            # Load individual models
            vit_model = create_vit_classifier(num_classes=2)
            vit_checkpoint = torch.load(vit_model_path, map_location=torch.device('cpu'))
            if 'model_state_dict' in vit_checkpoint:
                vit_model.load_state_dict(vit_checkpoint['model_state_dict'])
            else:
                vit_model.load_state_dict(vit_checkpoint)
            vit_model.eval()
            
            efficientnet_model = SEEfficientNet(num_classes=1)
            eff_checkpoint = torch.load(efficient_model_path, map_location=torch.device('cpu'))
            if 'model_state_dict' in eff_checkpoint:
                efficientnet_model.load_state_dict(eff_checkpoint['model_state_dict'])
            else:
                efficientnet_model.load_state_dict(eff_checkpoint)
            efficientnet_model.eval()
            
            # Load the optimized ensemble model
            ensemble_checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            
            # Get optimized weights
            best_weights = ensemble_checkpoint['best_weights']
            vit_weight, efficientnet_weight = best_weights
            
            # Create the ensemble model
            ensemble_model = EnsembleModel(
                vit_model, 
                efficientnet_model, 
                vit_weight=vit_weight, 
                efficientnet_weight=efficientnet_weight
            )
            
            # Load ensemble state if it exists
            if 'model_state_dict' in ensemble_checkpoint:
                ensemble_model.load_state_dict(ensemble_checkpoint['model_state_dict'])
            
            # Set to evaluation mode
            ensemble_model.eval()
            
            return ensemble_model
            
        except Exception as e:
            st.error(f"Error loading ensemble model: {str(e)}")
            st.error(traceback.format_exc())
            
            # Fallback: Create a dummy model for demonstration
            class DummyEnsemble(torch.nn.Module):
                def __init__(self):
                    super(DummyEnsemble, self).__init__()
                
                def forward(self, x):
                    # Return a binary classification output (two probabilities)
                    return torch.tensor([[0.5, 0.5]])
            
            dummy_model = DummyEnsemble()
            st.warning("Using a simulated ensemble model for demonstration purposes")
            return dummy_model
            
    except Exception as e:
        st.error(f"Error loading vision model: {str(e)}")
        st.error(traceback.format_exc())
        
        # Return a dummy model that always predicts 0.5 probability
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super(DummyModel, self).__init__()
            
            def forward(self, x):
                # Always return a middle probability (0.5)
                return torch.tensor([[0.5, 0.5]])
        
        dummy_model = DummyModel()
        st.warning("Using a dummy model that returns fixed predictions for demonstration purposes")
        return dummy_model

# Function to load LLM model for melanoma information chatbot
@st.cache_resource
def load_llm_model():
    MODEL_PATH = r"C:\Users\jakif\.lmstudio\models\QuantFactory\Bio-Medical-Llama-3-8B-GGUF\Bio-Medical-Llama-3-8B.Q4_K_S.gguf"
    
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}. Please verify the path.")
        return None
    
    try:
        model = Llama(
            model_path=MODEL_PATH,
            n_ctx=4096,           # Context size
            n_gpu_layers=-1,      # Use as many GPU layers as possible (-1)
            n_batch=512,          # Batch size for inference
            verbose=False         # Silence logs
        )
        return model
    except Exception as e:
        st.error(f"Error loading LLM model: {str(e)}")
        st.error(traceback.format_exc())
        return None

# Function to generate responses from the LLM model
def generate_response(prompt, model):
    if model is None:
        return "Model could not be loaded. Please check the model path and try again."
    
    # Context for dermatologist specialist
    system_prompt = """
    You act as a dermatologist specializing in melanomas. Your goal is to provide accurate and educational medical information about melanomas, including:

    - Identifying warning signs
    - Risk factors
    - Prevention methods
    - Diagnosis and treatment options
    - When to seek medical attention
    -Critical information
    - Common misconceptions
    - Statistics and research findings

    Remember that you are not diagnosing specific cases and should always recommend consulting a physician for individual cases. Base your answers on up-to-date scientific evidence.
    """
    
    # Format for LLaMA model
    full_prompt = f"""<|system|>
{system_prompt}
<|user|>
{prompt}
<|assistant|>"""
    
    try:
        # Generate response
        response = model.create_completion(
            full_prompt,
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            stop=["<|user|>", "<|system|>"],
            echo=False
        )
        
        # Extract text from response
        return response['choices'][0]['text'].strip()
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return f"I'm sorry, I encountered an error while generating a response. Please try again."

# Function to classify text response
def classify_text_response(response, model, tokenizer, id_to_label):
    try:
        # Put model in evaluation mode
        model.eval()
        
        # Tokenize input
        inputs = tokenizer(response, return_tensors="pt", padding=True, truncation=True)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            score = probs[0, pred_class].item()
        
        # Get label
        label_name = id_to_label[pred_class]
        
        # Return full probabilities for all classes
        all_probs = probs[0].tolist()
        class_probs = {id_to_label[i]: prob for i, prob in enumerate(all_probs)}
        
        return label_name, score, class_probs
    except Exception as e:
        st.error(f"Text classification error: {str(e)}")
        st.error(traceback.format_exc())
        return "error", 0.0, {}

# Function to preprocess and classify image
# Función para preprocesar y clasificar imágenes con enfoque de dos etapas
def classify_image(image, model, id_to_label):
    try:
        # Define transformaciones de imagen
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Preprocesar la imagen
        img_tensor = transform(image).unsqueeze(0)  # Añadir dimensión de lote
        
        # Realizar predicción
        with torch.no_grad():
            outputs = model(img_tensor)
            
            # PRIMERA ETAPA: Clasificación binaria (benigno vs maligno)
            binary_result = ""
            binary_confidence = 0.0
            
            if outputs.shape[1] == 2:  # Modelo con salida softmax [benigno_prob, maligno_prob]
                binary_probs = outputs[0]
                maligno_prob = binary_probs[0].item()
                benigno_prob = binary_probs[1].item()
                
                # Determinar resultado binario
                if maligno_prob > 0.5:
                    binary_result = "Maligno"
                    binary_confidence = maligno_prob
                else:
                    binary_result = "Benigno"
                    binary_confidence = benigno_prob
                
            elif outputs.shape[1] == 1:  # Modelo con salida sigmoid
                maligno_prob = torch.sigmoid(outputs)[0][0].item()
                benigno_prob = 1.0 - maligno_prob
                
                # Determinar resultado binario
                if maligno_prob > 0.5:
                    binary_result = "Maligno"
                    binary_confidence = maligno_prob
                else:
                    binary_result = "Benigno"
                    binary_confidence = benigno_prob
            
            # SEGUNDA ETAPA: Asignar a una de las 4 categorías según la confianza
            # Crear un mapeo para convertir el resultado binario y confianza a una categoría de preocupación
            
            # Inicializar variables
            pred_class = 0  # por defecto: no preocupante
            class_probs = {}
            
            # Asignar categoría según el resultado binario y nivel de confianza
            if binary_result == "Benigno":
                if binary_confidence > 0.9:  # Muy alta confianza de que es benigno
                    pred_class = 0  # no preocupante
                    class_probs = {
                        id_to_label[0]: binary_confidence,
                        id_to_label[1]: 1.0 - binary_confidence,
                        id_to_label[2]: 0.0,
                        id_to_label[3]: 0.0
                    }
                else:  # Confianza moderada de que es benigno
                    pred_class = 1  # levemente preocupante
                    class_probs = {
                        id_to_label[0]: 0.0,
                        id_to_label[1]: binary_confidence,
                        id_to_label[2]: 1.0 - binary_confidence,
                        id_to_label[3]: 0.0
                    }
            else:  # Maligno
                if binary_confidence > 0.8:  # Alta confianza de que es maligno
                    pred_class = 3  # altamente preocupante
                    class_probs = {
                        id_to_label[0]: 0.0,
                        id_to_label[1]: 0.0,
                        id_to_label[2]: 1.0 - binary_confidence,
                        id_to_label[3]: binary_confidence
                    }
                else:  # Confianza moderada de que es maligno
                    pred_class = 2  # moderadamente preocupante
                    class_probs = {
                        id_to_label[0]: 0.0,
                        id_to_label[1]: 0.0,
                        id_to_label[2]: binary_confidence,
                        id_to_label[3]: 1.0 - binary_confidence
                    }
            
            # Obtener nombre de etiqueta
            label_name = id_to_label[pred_class]
            score = class_probs[label_name]
            
            # Añadir el resultado binario al diccionario de resultados
            class_probs["binary_result"] = binary_result
            class_probs["binary_confidence"] = binary_confidence

        return label_name, score, class_probs
    except Exception as e:
        st.error(f"Error de clasificación de imagen: {str(e)}")
        st.error(traceback.format_exc())
        return "error", 0.0, {"error": str(e)}
# Logic-based fusion of text and image predictions
def combine_predictions(text_probs, image_probs, unique_labels, user_responses=None):
    try:
        # Initialize combined scores
        combined_probs = {}
        
        # Define base weights
        base_text_weight = 0.4
        base_image_weight = 0.6
        text_weight = base_text_weight
        image_weight = base_image_weight
        
        # Apply logical rules to adjust weights based on content of responses
        if user_responses:
            all_responses = " ".join(user_responses.values())
            
            # Rule 1: If responses mention size changes, increase text weight for "growth" indicators
            growth_keywords = ['grown', 'growing', 'larger', 'bigger', 'increased', 'expanded', 'size']
            if any(keyword in all_responses.lower() for keyword in growth_keywords):
                if 'highly_concerning' in text_probs and text_probs['highly_concerning'] > 0.3:
                    text_weight += 0.15
                    image_weight -= 0.15
                    st.info("Detected mentions of growth in your responses, giving higher weight to text analysis.")
            
            # Rule 2: If responses mention color changes, increase text weight for color indicators
            color_keywords = ['color', 'dark', 'darkened', 'black', 'red', 'multicolor', 'colors', 'changed']
            if any(keyword in all_responses.lower() for keyword in color_keywords):
                if 'moderately_concerning' in text_probs and text_probs['moderately_concerning'] > 0.3:
                    text_weight += 0.1
                    image_weight -= 0.1
                    st.info("Detected mentions of color changes in your responses, giving higher weight to text analysis.")
            
            # Rule 3: If responses mention bleeding or itching, increase text weight significantly
            symptom_keywords = ['bleed', 'bleeding', 'itch', 'itchy', 'painful', 'tender', 'hurt']
            if any(keyword in all_responses.lower() for keyword in symptom_keywords):
                text_weight += 0.2
                image_weight -= 0.2
                st.info("Detected mentions of symptoms in your responses, giving higher weight to text analysis.")
            
            # Rule 4: If responses mention sun exposure or family history, increase text weight
            risk_keywords = ['sun', 'sunburn', 'fair skin', 'family history', 'melanoma history', 'skin cancer']
            if any(keyword in all_responses.lower() for keyword in risk_keywords):
                text_weight += 0.1
                image_weight -= 0.1
                st.info("Detected mentions of risk factors in your responses, giving higher weight to text analysis.")
        
        # Rule 5: Strong agreement between models increases confidence
        if text_probs and image_probs:
            # Create a copy of image_probs without non-label keys for comparison
            filtered_image_probs = {k: v for k, v in image_probs.items() if k in unique_labels}
            
            # Find highest probability class for each model
            text_max_class = max(text_probs, key=text_probs.get) if text_probs else "error"
            image_max_class = max(filtered_image_probs, key=filtered_image_probs.get) if filtered_image_probs else "error"
            
            # If both models agree, boost that class
            if text_max_class == image_max_class and text_max_class != "error":
                st.success(f"Both models agree on the classification: {text_max_class.replace('_', ' ').title()}")
                
                # Get average confidence
                avg_confidence = (text_probs[text_max_class] + filtered_image_probs[text_max_class]) / 2
                
                # Apply confidence boost
                for label in unique_labels:
                    if label == text_max_class:
                        # Boost the agreed class by 20%
                        if label in text_probs:
                            text_probs[label] = min(1.0, text_probs[label] * 1.2)
                        if label in filtered_image_probs:
                            filtered_image_probs[label] = min(1.0, filtered_image_probs[label] * 1.2)
        
        # Normalize weights to sum to 1
        total_weight = text_weight + image_weight
        text_weight = text_weight / total_weight
        image_weight = image_weight / total_weight
        
        # Combine probabilities for each class
        for label in unique_labels:
            text_prob = text_probs.get(label, 0.0)
            # Use the filtered image probs to avoid non-label keys
            image_prob = filtered_image_probs.get(label, 0.0) if 'filtered_image_probs' in locals() else image_probs.get(label, 0.0)
            combined_probs[label] = (text_prob * text_weight) + (image_prob * image_weight)
        
        # Log the weights used
        st.write(f"Diagnostic weighting: Patient responses ({text_weight:.2f}), Image analysis ({image_weight:.2f})")
        
        # Get the highest probability class
        max_label = max(combined_probs, key=combined_probs.get) if combined_probs else "error"
        max_score = combined_probs.get(max_label, 0.0)
        
        # Add the binary classification result to the combined probs if available
        if "binary_result" in image_probs:
            combined_probs["binary_result"] = image_probs["binary_result"]
        if "binary_confidence" in image_probs:
            combined_probs["binary_confidence"] = image_probs["binary_confidence"]
            
        return max_label, max_score, combined_probs
    except Exception as e:
        st.error(f"Error combining predictions: {str(e)}")
        st.error(traceback.format_exc())
        # Return the image prediction as fallback
        if image_probs:
            # Filter out non-label keys
            filtered_image_probs = {k: v for k, v in image_probs.items() if k in unique_labels}
            if filtered_image_probs:
                max_label = max(filtered_image_probs, key=filtered_image_probs.get)
                max_score = filtered_image_probs.get(max_label, 0.0)
                return max_label, max_score, image_probs
        return "error", 0.0, {"error": str(e)}

# Function to visualize classification results and risk levels
def visualize_results(text_label, text_score, image_label, image_score, combined_label, combined_score):
    # Define color mapping
    color_map = {
        'not_concerning': '#4CAF50',  # Green
        'mildly_concerning': '#FFEB3B',  # Yellow
        'moderately_concerning': '#FF9800',  # Orange
        'highly_concerning': '#F44336',   # Red
        'error': '#9E9E9E'  # Gray for errors
    }
    
    # Ensure all labels are strings and handle None values
    text_label = str(text_label) if text_label is not None else "error"
    image_label = str(image_label) if image_label is not None else "error"
    combined_label = str(combined_label) if combined_label is not None else "error"
    
    # Create data for visualization - ensure strict order
    data = {
        'Model': ['Questionnaire Results', 'Image Analysis', 'Combined Diagnosis'],
        'Confidence': [text_score, image_score, combined_score],
        'Classification': [text_label, image_label, combined_label]
    }
    
    # Create colors list in the same order
    colors = [color_map.get(text_label, '#9E9E9E'), 
              color_map.get(image_label, '#9E9E9E'), 
              color_map.get(combined_label, '#9E9E9E')]
    
    # Create dataframe with consistent order
    chart_data = pd.DataFrame(data)
    
    # Create a bar chart for confidence levels
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create barplot with explicit index ordering
    bars = ax.bar(range(len(data['Model'])), data['Confidence'], color=colors)
    
    # Set x-axis ticks and labels
    ax.set_xticks(range(len(data['Model'])))
    ax.set_xticklabels(data['Model'])
    
    # Add classification labels directly mapped to each bar
    label_texts = [l.replace('_', ' ').title() for l in data['Classification']]
    
    for i, (bar, label_text) in enumerate(zip(bars, label_texts)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            label_text,
            ha='center',
            va='bottom',
            fontsize=12
        )
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map['not_concerning'], label='Not Concerning'),
        Patch(facecolor=color_map['mildly_concerning'], label='Mildly Concerning'),
        Patch(facecolor=color_map['moderately_concerning'], label='Moderately Concerning'),
        Patch(facecolor=color_map['highly_concerning'], label='Highly Concerning')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_title('Confidence by Diagnostic Method', fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Confidence Level')
    ax.set_xlabel('Analysis Method')
    
    plt.tight_layout()
    
    # Show plot
    st.pyplot(fig)

# Define the list of questions grouped by categories
questions_by_category = {
    "Growth and Evolution": [
        "Has the lesion grown or changed in size in recent months?",
        "Have you noticed any change in its shape over time?",
        "Has the color of the lesion changed recently?"
    ],
    "Appearance": [
        "Is the lesion larger than 6mm (about the size of a pencil eraser)?",
        "Does the lesion look different from other moles or spots on your body?"
    ],
    "Symptoms": [
        "Is the lesion itchy?",
        "Does the lesion bleed without being injured?",
        "Is the area around the lesion red or swollen?",
        "Do you feel pain or tenderness in the lesion?",
        "Has the lesion formed a scab or crust that doesn't heal?"
    ],
    "Additional Risk Factors": [
        "Is the lesion regularly exposed to the sun?",
        "Have you had severe sunburns in the past, especially as a child?",
        "Do you have a family history of melanoma or skin cancer?",
        "Do you have many moles (more than 50) on your body?",
        "Do you have fair skin that burns easily in the sun?"
    ]
}

# Flatten the questions list for other functions
all_questions = []
for category, questions in questions_by_category.items():
    all_questions.extend(questions)
# Function to implement Grad-CAM visualization without external packages
def apply_custom_gradcam(image, model, id_to_label):
    """
    Apply a custom implementation of Grad-CAM visualization to an image.
    
    Args:
        image: PIL Image to visualize
        model: PyTorch model for classification
        id_to_label: Dictionary mapping label IDs to label names
        
    Returns:
        visualization: PIL Image with Grad-CAM heatmap overlay
        original_image: PIL Image resized to match model input
        pred_label: Predicted label
        binary_result: Binary classification result (Malignant/Benign)
        binary_confidence: Confidence in binary classification
    """
    try:
        import torch
        import numpy as np
        import torch.nn.functional as F
        import torchvision.transforms as transforms
        from PIL import Image
        import matplotlib.pyplot as plt
        import cv2
        
        # Define image transformation
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Preprocess the image for visualization (without normalization)
        vis_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        # Create a class to capture activations and gradients
        class GradCamHook:
            def __init__(self, module):
                self.activations = None
                self.gradients = None
                self.forward_hook = module.register_forward_hook(self.hook_forward)
                self.backward_hook = module.register_full_backward_hook(self.hook_backward)
                
            def hook_forward(self, module, input, output):
                self.activations = output.detach()
                
            def hook_backward(self, module, grad_input, grad_output):
                self.gradients = grad_output[0].detach()
                
            def remove(self):
                self.forward_hook.remove()
                self.backward_hook.remove()
        
        # Helper function to find the most suitable layer for Grad-CAM
        def find_target_layer(model):
            # For ensemble model with EfficientNet
            if hasattr(model, 'efficientnet_model') and hasattr(model.efficientnet_model, 'features'):
                return model.efficientnet_model.features[-1]
            # For ensemble model with ViT
            elif hasattr(model, 'vit_model') and hasattr(model.vit_model, 'blocks'):
                return model.vit_model.blocks[-1]
            # Generic case: find the last convolutional layer
            else:
                target_layer = None
                for name, module in reversed(list(model.named_modules())):
                    if isinstance(module, torch.nn.Conv2d):
                        target_layer = module
                        break
                return target_layer
        
        # Get the target layer
        target_layer = find_target_layer(model)
        if target_layer is None:
            st.error("Could not find a suitable target layer for Grad-CAM")
            return None, image, "error", "Unknown", 0.0
        
        # Register hooks
        grad_cam_hook = GradCamHook(target_layer)
        
        # Make a copy of the model that requires gradients
        model.eval()
        model.zero_grad()
        
        # Preprocess the image
        img_tensor = transform(image).unsqueeze(0)
        img_tensor.requires_grad = True
        
        # Get the unnormalized image for visualization
        vis_img_tensor = vis_transform(image)
        vis_img_array = vis_img_tensor.numpy().transpose(1, 2, 0)
        
        # Forward pass
        if hasattr(model, 'vit_model') and hasattr(model, 'efficientnet_model'):
            # Handle ensemble model
            outputs = model(img_tensor)
            # Extract binary classification results
            if outputs.shape[1] == 2:  # Softmax outputs [benign_prob, malignant_prob]
                binary_probs = outputs[0]
                malignant_prob = binary_probs[1].item()
                benign_prob = binary_probs[0].item()
                
                # Determine binary result
                if malignant_prob > 0.5:
                    binary_result = "Malignant"
                    binary_confidence = malignant_prob
                    # CORRECCIÓN: En lugar de asignar directamente la clase 3 o 2,
                    # verificamos cuántas clases tiene el modelo y mapeamos según corresponda
                    pred_class = 1  # En el caso binario, usamos la clase 1 (maligno)
                else:
                    binary_result = "Benign"
                    binary_confidence = benign_prob
                    # CORRECCIÓN: Similar a lo anterior
                    pred_class = 0  # En el caso binario, usamos la clase 0 (benigno)
            else:
                # Para otros formatos de salida, usamos argmax
                pred_class = torch.argmax(outputs, dim=1).item()
                binary_result = "Unknown"
                binary_confidence = 0.0
        else:
            # Modelo genérico
            outputs = model(img_tensor)
            pred_class = torch.argmax(outputs, dim=1).item()
            binary_result = "Unknown"
            binary_confidence = 0.0
        
        # Get class name
        mapped_class = min(pred_class, len(id_to_label) - 1)  # Ensure we don't go out of bounds
        pred_label = id_to_label[mapped_class]
        
        # Backward pass to get gradients
        if outputs.shape[1] > 1:  # Multi-class case
            model.zero_grad()
            # CORRECCIÓN: Asegurarse de que pred_class no esté fuera de los límites
            target_class = min(pred_class, outputs.shape[1] - 1)
            outputs[0, target_class].backward()
        else:  # Binary case
            model.zero_grad()
            outputs.backward()
        
        # Get activations and gradients
        activations = grad_cam_hook.activations
        gradients = grad_cam_hook.gradients
        
        # Remove hooks
        grad_cam_hook.remove()
        
        # Compute Grad-CAM
        if activations is not None and gradients is not None:
            # Global average pooling of gradients
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
            
            # Weight the activations by the gradients
            cam = torch.sum(activations * weights, dim=1, keepdim=True)
            
            # Apply ReLU to focus on features that have a positive influence
            cam = F.relu(cam)
            
            # Normalize the CAM
            cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            
            # Convert to numpy array
            cam = cam[0, 0].detach().cpu().numpy()
            
            # Apply colormap to the heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Resize heatmap to match the original image
            heatmap = cv2.resize(heatmap, (224, 224))
            
            # Overlay heatmap on original image
            alpha = 0.4  # Transparency factor
            visualization = heatmap * alpha + vis_img_array * 255 * (1 - alpha)
            visualization = np.clip(visualization, 0, 255).astype(np.uint8)
            
            # Convert back to PIL Image
            visualization = Image.fromarray(visualization)
            original_image = Image.fromarray((vis_img_array * 255).astype(np.uint8))
            
            return visualization, original_image, pred_label, binary_result, binary_confidence
        else:
            st.warning("Could not capture activations or gradients for Grad-CAM")
            return None, image, pred_label, binary_result, binary_confidence
            
    except Exception as e:
        st.error(f"Error generating Grad-CAM visualization: {str(e)}")
        st.error(traceback.format_exc())
        return None, image, "error", "Unknown", 0.0
def text_to_speech(text, language='en'):
    """
    Convert text to speech and return audio bytes.
    
    Args:
        text: Text to convert to speech
        language: Language code (e.g., 'es' for Spanish, 'en' for English)
        
    Returns:
        Audio bytes that can be played in Streamlit
    """
    try:
        from gtts import gTTS
        import tempfile
        import os
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            temp_filename = fp.name
        
        # Generate speech
        tts = gTTS(text=text, lang=language, slow=False)
        tts.save(temp_filename)
        
        # Read the audio file
        with open(temp_filename, 'rb') as f:
            audio_bytes = f.read()
        
        # Clean up
        os.unlink(temp_filename)
        
        return audio_bytes
    except Exception as e:
        st.error(f"Error generating speech: {str(e)}")
        return None
# Main function
def main():
    # Title and description
    st.title("Multi-Modal Melanoma Diagnostic System")
    st.write("""
    This advanced application combines the analysis of your responses with a computer vision model
    to provide a more accurate assessment of melanoma risk. Complete the questionnaire (by text or voice) and upload
    an image of the lesion for a comprehensive analysis.
    """)
    
    # Load data
    data = load_data()
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
    if not data or "data" not in data or not data["data"]:
        st.error("No data available for analysis.")
        return
    
    unique_labels, label_to_id, id_to_label = get_labels(data)
    
    # Load models
    tokenizer, text_model = load_text_model(label_to_id, id_to_label)
    vision_model = load_vision_model()
    
    # Check if models are loaded
    text_model_loaded = tokenizer is not None and text_model is not None
    vision_model_loaded = vision_model is not None
    
    # Show model loading status
    if not text_model_loaded or not vision_model_loaded:
        st.warning("One or more models could not be loaded. Some functionality may be limited.")
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Questionnaire & Diagnosis", "General information about Melanoma", "Specific information"])
    
    with tab1:
        # Store the state of the app
        if 'step' not in st.session_state:
            st.session_state.step = 1  # 1: Questionnaire, 2: Image, 3: Results
        
        if 'responses' not in st.session_state:
            st.session_state.responses = {}
            
        if 'audio_files' not in st.session_state:
            st.session_state.audio_files = {}
        
        if 'text_result' not in st.session_state:
            st.session_state.text_result = None
        
        if 'image_result' not in st.session_state:
            st.session_state.image_result = None
        
        if 'uploaded_image' not in st.session_state:
            st.session_state.uploaded_image = None
        
        # First, add a new function to classify single responses
        def classify_single_response(response, model, tokenizer, id_to_label):
            """Classify a single response and return its concern level"""
            if not response.strip() or not model or not tokenizer:
                return None, 0.0, {}
            
            try:
                # Use the existing classification function
                label_name, score, class_probs = classify_text_response(response, model, tokenizer, id_to_label)
                return label_name, score, class_probs
            except Exception as e:
                st.error(f"Error classifying response: {str(e)}")
                return None, 0.0, {}
        # Step 1: Complete questionnaire
        if st.session_state.step == 1:
            st.header("Skin Lesion Assessment Questionnaire")
            st.write("Please answer the following questions about the skin lesion. You can type or record your answers:")
            
            # Make sure models are loaded before using them
            tokenizer, text_model = load_text_model(label_to_id, id_to_label)
            
            # Store classifications in session state if not already present
            if 'response_classifications' not in st.session_state:
                st.session_state.response_classifications = {}
            
            # Toggle for input method selection
            input_method = st.radio("Select your preferred input method:", ["Text", "Voice"], horizontal=True)
            
            # Create a multi-step form for each category
            all_completed = True
            
            # Define colors for different concern levels
            concern_colors = {
                'not_concerning': '#4CAF50',  # Green
                'mildly_concerning': '#FFEB3B',  # Yellow
                'moderately_concerning': '#FF9800',  # Orange
                'highly_concerning': '#F44336',  # Red
                'error': '#9E9E9E'  # Gray
            }
            
            for category, questions in questions_by_category.items():
                with st.expander(f"{category}", expanded=True):
                    st.write(f"**{category}**")
                    
                    for question in questions:
                        # Create a unique key for each question
                        question_key = f"q_{questions_by_category[category].index(question)}_{category}"
                        
                        # Display the question
                        st.markdown(f"**{question}**")
                        
                        # Save previous response if exists
                        previous_response = st.session_state.responses.get(question_key, "")
                        
                        # Handle different input methods
                        if input_method == "Text":
                            # Get response (use previous response if exists)
                            response = st.text_input(
                                "Type your answer",
                                value=previous_response,
                                key=f"text_{question_key}"
                            )
                            
                            # Check if a new response was entered (by comparing with previous)
                            if response != previous_response and response.strip():
                                # Classify the response immediately
                                if text_model and tokenizer:
                                    label, score, _ = classify_single_response(response, text_model, tokenizer, id_to_label)
                                    
                                    if label:
                                        # Store the classification
                                        st.session_state.response_classifications[question_key] = label
                                        
                                        # Display the classification with appropriate styling
                                        concern_color = concern_colors.get(label, '#9E9E9E')
                                        st.markdown(
                                            f"""
                                            <div style="padding: 10px; border-radius: 5px; background-color: {concern_color}; color: white;">
                                            <strong>Classification:</strong> {label.replace('_', ' ').title()} (Confidence: {score:.2f})
                                            </div>
                                            """, 
                                            unsafe_allow_html=True
                                        )
                        else:  # Voice input
                            # Create columns for voice input and display
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                st.write("Record your answer:")
                                
                                # Add audio recorder with a unique key for each question
                                audio_bytes = audio_recorder(
                                    key=f"audio_{question_key}",
                                    text="",
                                    recording_color="#e8b62c",
                                    neutral_color="#6aa36f",
                                    icon_name="microphone",
                                    icon_size="2x"
                                )
                                
                                if audio_bytes:
                                    # Display the recorded audio
                                    st.audio(audio_bytes, format="audio/wav")
                                    
                                    # Store audio for later reference
                                    st.session_state.audio_files[question_key] = audio_bytes
                                    
                                    # Transcribe audio
                                    with st.spinner("Transcribing your response..."):
                                        transcription = process_audio_to_text(audio_bytes)
                                    
                                    if transcription:
                                        # Store the transcribed text
                                        st.session_state.responses[question_key] = transcription
                                        
                                        # Classify the transcribed response
                                        if text_model and tokenizer:
                                            label, score, _ = classify_single_response(transcription, text_model, tokenizer, id_to_label)
                                            
                                            if label:
                                                # Store the classification
                                                st.session_state.response_classifications[question_key] = label
                                                
                                                # Display classification with appropriate styling
                                                concern_color = concern_colors.get(label, '#9E9E9E')
                                                st.markdown(
                                                    f"""
                                                    <div style="padding: 10px; border-radius: 5px; background-color: {concern_color}; color: white;">
                                                    <strong>Classification:</strong> {label.replace('_', ' ').title()} (Confidence: {score:.2f})
                                                    </div>
                                                    """, 
                                                    unsafe_allow_html=True
                                                )

                            with col2:
                                # Display previous response or transcribed text
                                response = st.session_state.responses.get(question_key, "")
                                
                                # Allow editing the transcribed text
                                new_response = st.text_input(
                                    "Verify or edit your answer",
                                    value=response,
                                    key=f"edit_{question_key}"
                                )
                                
                                # Check if response was edited
                                if new_response != response and new_response.strip():
                                    # Classify the edited response
                                    if text_model and tokenizer:
                                        label, score, _ = classify_single_response(new_response, text_model, tokenizer, id_to_label)
                                        
                                        if label:
                                            # Store the classification
                                            st.session_state.response_classifications[question_key] = label
                                            
                                            # Display classification with appropriate styling
                                            concern_color = concern_colors.get(label, '#9E9E9E')
                                            st.markdown(
                                                f"""
                                                <div style="padding: 10px; border-radius: 5px; background-color: {concern_color}; color: white;">
                                                <strong>Classification:</strong> {label.replace('_', ' ').title()} (Confidence: {score:.2f})
                                                </div>
                                                """, 
                                                unsafe_allow_html=True
                                            )
                                
                                response = new_response
                        
                        # Store response
                        st.session_state.responses[question_key] = response
                        
                        # Display existing classification if available
                        if question_key in st.session_state.response_classifications and not (response != previous_response and response.strip()):
                            label = st.session_state.response_classifications[question_key]
                            concern_color = concern_colors.get(label, '#9E9E9E')
                            st.markdown(
                                f"""
                                <div style="padding: 10px; border-radius: 5px; background-color: {concern_color}; color: white;">
                                <strong>Classification:</strong> {label.replace('_', ' ').title()}
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                        
                        # Check if this question is completed
                        if not response.strip():
                            all_completed = False
                        
                        # Add some space between questions
                        st.write("")
            
            # Continue button
            col1, col2 = st.columns([4, 1])
            with col2:
                continue_button = st.button(
                    "Continue to Image" if all_completed else "Continue to Image (some questions are empty)",
                    disabled=False,
                    type="primary" if all_completed else "secondary"
                )
                
                if continue_button:
                    # Proceed to image upload
                    st.session_state.step = 2
        
        # Step 2: Upload image
        elif st.session_state.step == 2:
            st.header("Upload Image of the Lesion")
            st.write("Please upload a clear image of the skin lesion:")
            
            # Image upload section
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            
            # Display the image if uploaded
            if uploaded_file is not None:
                st.session_state.uploaded_image = uploaded_file
                image = Image.open(uploaded_file).convert('RGB')
                
                # Create columns for original image and Grad-CAM visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Original Image**")
                    st.image(image, caption="Uploaded Image", width=300)
                
                # Generate Grad-CAM visualization if model is loaded
                if vision_model_loaded:
                    with col2:
                        st.write("**Feature Importance Visualization**")
                        with st.spinner("Generating heatmap visualization..."):
                            # Get custom Grad-CAM visualization
                            vis_image, _, pred_label, binary_result, binary_confidence = apply_custom_gradcam(
                                image, vision_model, id_to_label
                            )
                            
                            if vis_image:
                                st.image(vis_image, caption=f"Grad-CAM Visualization", width=300)
                                st.write(f"Initial Classification: **{binary_result}** (Confidence: {binary_confidence:.2f})")
                                st.info("Red/yellow areas highlight regions most influential for the model's prediction")
                            else:
                                st.warning("Could not generate visualization. Proceeding with regular analysis.")
                
                # Buttons for navigation
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    if st.button("← Back to Questionnaire"):
                        st.session_state.step = 1
                
                with col2:
                    if st.button("Perform Analysis", type="primary"):
                        # Process the image and questionnaire
                        st.session_state.step = 3
            else:
                # Only show back button if no image
                if st.button("← Back to Questionnaire"):
                    st.session_state.step = 1
                    
        
        # Step 3: Show results
        elif st.session_state.step == 3:
            st.header("Diagnostic Results")
            
            # Show loading spinner while processing
            with st.spinner("Analyzing data..."):
                # Process questionnaire responses if not already done
                if st.session_state.text_result is None and text_model_loaded:
                    # Combine all responses into one text
                    combined_response = " ".join(st.session_state.responses.values())
                    
                    # Classify text
                    text_label, text_score, text_probs = classify_text_response(
                        combined_response, text_model, tokenizer, id_to_label
                    )
                    
                    # Store result
                    st.session_state.text_result = {
                        "label": text_label,
                        "score": text_score,
                        "probs": text_probs
                    }
                
                # Process image if uploaded and not already done
                if st.session_state.image_result is None and vision_model_loaded and st.session_state.uploaded_image:
                    # Load image
                    image = Image.open(st.session_state.uploaded_image).convert('RGB')
                    vis_image, original_image, pred_label, binary_result, binary_confidence = apply_custom_gradcam(
                                image, vision_model, id_to_label
                            )
                    
                    # Resize both images to the same size for consistent display
                    resized_image = original_image.resize((300, 300))
                    resized_vis_image = vis_image.resize((300, 300))
                    
                    # Display the images
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(resized_image, caption="Analyzed Image", width=300)
                    with col2:
                        st.image(resized_vis_image, caption="Grad-CAM Visualization", width=300)

                    
                    # Classify image
                    image_label, image_score, image_probs = classify_image(
                        image, vision_model, id_to_label
                    )
                    
                    # Store result
                    st.session_state.image_result = {
                        "label": image_label,
                        "score": image_score,
                        "probs": image_probs
                    }
            
            # Show results
            if st.session_state.text_result or st.session_state.image_result:
                st.subheader("Analysis Results")
                
                # Handle the case where we have both text and image results
                if st.session_state.text_result and st.session_state.image_result:
                    text_label = st.session_state.text_result["label"]
                    text_score = st.session_state.text_result["score"]
                    text_probs = st.session_state.text_result["probs"]
                    
                    image_label = st.session_state.image_result["label"]
                    image_score = st.session_state.image_result["score"]
                    image_probs = st.session_state.image_result["probs"]
                    
                    # Show individual results first
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Questionnaire Analysis:**")
                        st.write(f"Classification: {text_label.replace('_', ' ').title()}")
                        st.write(f"Confidence: {text_score:.4f}")
                    
                    with col2:
                        st.write("**Image Analysis:**")
                        st.write(f"Classification: {image_label.replace('_', ' ').title()}")
                        st.write(f"Confidence: {image_score:.4f}")
                    
                    # Combine predictions
                    combined_label, combined_score, combined_probs = combine_predictions(
                        text_probs, image_probs, unique_labels, st.session_state.responses
                    )
                    
                    # Show combined results
                    st.markdown("---")
                    st.subheader("Final Diagnosis")
                    st.markdown(f"**Concern Level: {combined_label.replace('_', ' ').title()}**")
                    st.markdown(f"**Confidence: {combined_score:.4f}**")
                    
                    # Visualize all results
                    st.markdown("### Results Comparison")
                    visualize_results(
                        text_label, text_score, 
                        image_label, image_score, 
                        combined_label, combined_score
                    )
                    
                # Only text classification
                elif st.session_state.text_result:
                    text_label = st.session_state.text_result["label"]
                    text_score = st.session_state.text_result["score"]
                    
                    st.write("**Questionnaire Analysis:**")
                    st.write(f"Classification: {text_label.replace('_', ' ').title()}")
                    st.write(f"Confidence: {text_score:.4f}")
                    
                    st.markdown("---")
                    st.markdown("### Final Diagnosis")
                    st.markdown(f"**Concern Level: {text_label.replace('_', ' ').title()}**")
                    st.markdown(f"**Confidence: {text_score:.4f}**")
                    st.warning("This diagnosis is based solely on your responses. For a more accurate analysis, please upload an image of the lesion.")
                
                # Only image classification
                elif st.session_state.image_result:
                    image_label = st.session_state.image_result["label"]
                    image_score = st.session_state.image_result["score"]
                    
                    st.write("**Image Analysis:**")
                    st.write(f"Classification: {image_label.replace('_', ' ').title()}")
                    st.write(f"Confidence: {image_score:.4f}")
                    
                    st.markdown("---")
                    st.markdown("### Final Diagnosis")
                    st.markdown(f"**Concern Level: {image_label.replace('_', ' ').title()}**")
                    st.markdown(f"**Confidence: {image_score:.4f}**")
                    st.warning("This diagnosis is based solely on the image. For a more accurate analysis, please complete the questionnaire.")
                
                # Show detailed explanation based on the final diagnosis
                st.markdown("---")
                st.subheader("Results Interpretation")
                
                # Get the final label (combined, text, or image)
                final_label = None
                if st.session_state.text_result and st.session_state.image_result:
                    final_label = combined_label
                elif st.session_state.text_result:
                    final_label = text_label
                elif st.session_state.image_result:
                    final_label = image_label
                
                # Explanations for each concern level
                concern_explanations = {
                    'not_concerning': """
                    **Not Concerning**: No warning signs detected. 
                    However, it's important to regularly monitor any changes in the lesion.
                    
                    **Recommendations:**
                    - Perform regular self-examinations every 3 months
                    - Protect your skin from the sun with SPF 30+ sunscreen
                    - If you notice changes in the future, consult a dermatologist
                    """,
                    'mildly_concerning': """
                    **Mildly Concerning**: Some mild signs have been detected that warrant follow-up. 
                    It's recommended to periodically observe the lesion and consult a dermatologist
                    if additional changes are noticed.
                    
                    **Recommendations:**
                    - Take monthly photographs of the lesion to monitor changes
                    - Schedule a review with a dermatologist in the next 3 months
                    - Avoid prolonged sun exposure and use sunscreen
                    """,
                    'moderately_concerning': """
                    **Moderately Concerning**: Signs have been detected that require medical evaluation. 
                    It's recommended to schedule an appointment with a dermatologist in the coming weeks
                    for a professional assessment.
                    
                    **Recommendations:**
                    - Schedule a consultation with a dermatologist in the next 2-3 weeks
                    - Take photographs of the lesion to document its current state
                    - Avoid irritating or traumatizing the lesion
                    - Protect the area from sun exposure
                    """,
                    'highly_concerning': """
                    **Highly Concerning**: Serious signs have been detected that require immediate medical attention. 
                    It's recommended to consult a dermatologist as soon as possible for a complete evaluation
                    and possible biopsy.
                    
                    **Recommendations:**
                    - Seek urgent dermatological care (within the next few days)
                    - Do not attempt to treat the lesion on your own
                    - Take photographs from different angles for the dermatologist
                    - Prepare a history of when you noticed the lesion and how it has changed
                    """
                }
                
                # Show explanation for final diagnosis
                if final_label:
                    st.markdown(concern_explanations.get(final_label, ""))
                
                # Add note about model agreement if applicable
                if st.session_state.text_result and st.session_state.image_result:
                    if text_label == image_label:
                        st.success("Both analysis methods agree on the classification, increasing confidence in the diagnosis.")
                
                # Show warning about medical advice
                st.warning("""
                **Important note**: This classification is only an aid tool and does not substitute 
                professional medical diagnosis. Always consult a dermatologist for proper evaluation.
                """)
                
                # Navigation buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("← Back to Questionnaire"):
                        # Reset results to force recalculation
                        st.session_state.text_result = None
                        st.session_state.image_result = None
                        st.session_state.step = 1
                        
                
                with col2:
                    if st.button("← Back to Image"):
                        # Reset image result to force recalculation
                        st.session_state.image_result = None
                        st.session_state.step = 2
                        
            else:
                st.error("Could not generate a valid classification. Please check your inputs or try again.")
                
                # Back button
                if st.button("← Back to Start"):
                    st.session_state.step = 1
                    

    with tab2:
        st.header("Educational Information on Melanoma")
        
        st.subheader("🩺 DermaBot - Ask About Melanoma")
        st.markdown("""
            This chatbot provides educational information about melanomas.
            **Important**: This bot does not provide medical diagnoses. 
            Always consult a healthcare professional for specific cases.
        """)
            
        # Initialize chatbot session state
        if "chatbot_messages" not in st.session_state:
            st.session_state.chatbot_messages = [
                {"role": "assistant", "content": "Hi, I'm DermaBot, a virtual assistant specializing in melanoma information. How can I help you today?"}
            ]
            
        # Load LLM model
        with st.spinner("Loading medical AI model..."):
            llm_model = load_llm_model()
                
        if llm_model is None:
            st.error("Could not load the LLM model. Chat functionality is not available.")
            st.info("You can still view the educational information on the left.")
        else:
            # Display chat messages
            for message in st.session_state.chatbot_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Add voice input option for the chatbot
            with st.expander("Ask with voice"):
                st.write("Record your question:")
                
                audio_bytes = audio_recorder(
                    key="chatbot_audio",
                    text="",
                    recording_color="#e8b62c",
                    neutral_color="#6aa36f",
                    icon_name="microphone",
                    icon_size="2x"
                )
                
                if audio_bytes:
                    # Show the recorded audio
                    st.audio(audio_bytes, format="audio/wav")
                    
                    # Transcribe audio
                    with st.spinner("Transcribing your question..."):
                        voice_question = process_audio_to_text(audio_bytes)
                        
                    if voice_question:
                        # Show the transcribed question
                        st.write("**Transcribed question:**")
                        st.write(voice_question)
                        
                        if st.button("Send voice question"):
                            # Add user message to chat history
                            st.session_state.chatbot_messages.append({"role": "user", "content": voice_question})
                            
                            # Generate and display response
                            with st.spinner("Thinking..."):
                                response = generate_response(voice_question, llm_model)
                            
                            # Add assistant response to chat history
                            st.session_state.chatbot_messages.append({"role": "assistant", "content": response})
                            
                
            
            # Text input for chatbot
            if prompt := st.chat_input("Ask a question about melanomas..."):
                # Add user message to chat history
                st.session_state.chatbot_messages.append({"role": "user", "content": prompt})                    
                with st.chat_message("user"):
                    st.markdown(prompt)
                    
                # Generate and display response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = generate_response(prompt, llm_model)
                        st.markdown(response)

                # Add assistant response to chat history
                st.session_state.chatbot_messages.append({"role": "assistant", "content": response})

        # Add warning about medical advice
        st.warning("⚠️ Remember: The information provided is not a substitute for professional medical advice.")
    with tab3:
        st.header("Specific Information on Melanoma through Scientific Articles")
        st.write("This section provides information from scientific articles and recommend interesting links.")    



        
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
            
            # List of documents
            st.sidebar.header("📑 Processed Documents")
            for doc_name, metadata in st.session_state.melanoma_rag.doc_metadata.items():
                st.sidebar.markdown(f"**{doc_name}**")

        
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
                        
                        # Display the answer with a read-aloud option
                        answer_text = item['result']['answer']
                        st.write(answer_text)
                        
                        # Add text-to-speech button
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            if st.button("🔊 Read Aloud", key=f"read_rag_{i}"):
                                with st.spinner("Generating audio..."):
                                    # Detect language (you could add a language selector instead)
                                    language = 'en'  # Default to English
                                    audio_bytes = text_to_speech(answer_text, language)
                                    
                                    if audio_bytes:
                                        with col2:
                                            st.audio(audio_bytes, format="audio/mp3")
                        
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
            # Add warning about medical advice
            st.warning("⚠️ Remember: The information provided is not a substitute for professional medical advice.")
if __name__ == "__main__":
    main()