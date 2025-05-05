# Melanoma Detector: Multi-Modal Melanoma Detection and Information System

## Overview
Melanoma Detector is a comprehensive system designed to assist in melanoma risk assessment and provide educational information about melanoma. This university project integrates multiple AI technologies including Natural Language Processing, Computer Vision, and Large Language Models to create a holistic approach to melanoma awareness and early detection.

## Key Features

### 1. Multi-Modal Diagnostic System
- **Questionnaire Assessment:** Patient responses are analyzed using a fine-tuned BioBERT model
- **Image Analysis:** Skin lesion images are classified using an ensemble of advanced deep learning models (ViT-16 + SE-EfficientB0)
- **Combined Analysis:** Fusion of text and image analysis for more accurate assessment with logical agents
- **Visualized Results:** Clear presentation of risk levels with explanations and recommendations

### 2. Advanced Visualization Techniques
- **Multi-scale GradCAM:** Visual explanation of the classification process across different feature scales
- **Feature Importance:** Highlighting of critical regions in the image that influenced the diagnosis
- **Confidence Visualization:** Comparative visualization of confidence levels across analysis methods

### 3. Informational Systems
- **General Information Chatbot:** LLM-powered assistant providing educational information about melanomas
- **Scientific Knowledge Base:** RAG (Retrieval-Augmented Generation) system that extracts and contextualizes information from scientific literature
- **Voice Interaction:** Text-to-voice capabilities for the information retrieval.

## Technical Components

### Natural Language Processing questionnaire
- **Model:** Fine-tuned BioBERT model based on transformers architecture
- **Classification:** Four-level categorization (highly concerning, moderately concerning, mildly concerning, not concerning)
- **Features:** 
  - Analysis of 15 clinically relevant questions
  - Voice input with automatic transcription with Whisper.
  - Real-time classification feedback

### Computer Vision
- **Architecture:** Ensemble model combining Vision Transformer (ViT-B16) and Squeeze-and-Excitation EfficientNet-B0
- **Visualization:** Multi-scale GradCAM for interpretability
- **Features:**
  - Binary classification (benign/malignant)
  - Four-level concern classification
  - Analysis of critical visual features for melanoma detection

### Large Language Models
- **General Information:** LLM-based chatbot for educational purposes
- **Specific Information:** RAG system integrating:
  - Scientific document analysis
  - Context-aware retrieval
  - Term extraction and explanation
  - Source citation and recommendation

## Technology Stack
- **Core Framework:** Python, Streamlit
- **NLP:** Transformers, HuggingFace, Whisper (speech-to-text), LmStudio
- **Computer Vision:** PyTorch, torchvision, TensorFlow
- **LLM Integration:** llama-cpp-python
- **RAG Components:** FAISS, Sentence Transformers
- **Data Processing:** pandas, numpy, langchain

## Installation and Usage

### Prerequisites
- Python 3.9+
- Required models:
  - Fine-tuned text classification model
  - Ensemble vision model
  - LLM model (local GGUF format)
