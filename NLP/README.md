# Melanoma Diagnosis Chatbot

This project aims to develop a chatbot to acquire knowledge from patients about melanomas. The chatbot is designed to ask questions and classify responses to assist in identifying potential melanoma symptoms. The project progresses through different stages, starting with a basic model and advancing to more sophisticated models.

## Project Overview

### 1. **Basic Model with Lemmatization**
The first step in the project involves creating a basic chatbot using lemmatization to analyze user responses. This model is implemented in the script [`modelo_basico.ipynb`](modelo_basico.ipynb). 

- **Key Features**:
  - Utilizes the `spaCy` library for natural language processing.
  - Lemmatizes user responses to identify key words such as "yes" or "no."
  - Returns a simple classification of responses as "yes," "no," or "Sorry, I don't understand."

### 2. **Rule Based Model**

![rule based model](rules_based.png)

### 3. **Naive Bayes Model with Bigrams**
The second stage involves building a more advanced chatbot using a Naive Bayes classification model with bigram feature extraction. This model is implemented in the script [`modelo_naive_bayes.ipynb`](modelo_naive_bayes.ipynb).

- **Key Features**:
  - Uses the `nltk` library for tokenization, stopword removal, and classification.
  - Implements bigram extraction to capture word pairs, significantly improving context understanding.
  - Trains a Naive Bayes classifier on a dataset of labeled responses.
  - Asks a predefined set of questions about melanoma symptoms and classifies user responses as "positive," "negative," or "I don't understand."


  ![CHATBOT_NAIVE_BAYES](chatbot_naive_bayes.png)

  ![RESULTS_NAIVE_BAYES](results_naive_bayes.png)


To improve the acquisition of information we developed an improved model:
  ![CHATBOT_NAIVE_BAYES](chatbot_improved_naive_bayes.png)

### 4. **BioBERT Model for Enhanced Classification**
Further advancing our approach, we implemented a BioBERT-based classification system to achieve higher accuracy in patient response interpretation.

- **Key Features**:
  - Leverages the powerful BioBERT model (pretrained on biomedical literature) as a base.
  - Fine-tuned with a custom JSON dataset containing possible patient responses and their corresponding concern levels.
  - Classifies responses into four categories: highly concerning, moderately concerning, mildly concerning, and not concerning.
  - Significantly outperforms previous models in understanding complex and nuanced patient descriptions.

### 5. **Information Provision Systems**

#### Medical Information Chatbot
We developed a specialized chatbot to provide relevant medical information about melanomas:

- **Key Features**:
  - Powered by the Bio-Medical-Llama-3-8B-GGUF model, specifically adapted for medical domain knowledge.
  - Offers educational information about melanoma symptoms, risk factors, prevention, and treatment options.
  - Provides accurate, up-to-date information while maintaining appropriate medical disclaimers.

#### RAG-Based Specific Information System
For more detailed and scientific information, we implemented a Retrieval-Augmented Generation (RAG) system:

- **Key Features**:
  - Uses the DeepSeek-R1-Distill-Llama-8B model to process and contextualize information.
  - Extracts data from scientific articles uploaded as PDFs.
  - Performs semantic search to find the most relevant information for user queries.
  - Presents information with source references and context.
  - Extracts and explains medical terminology found in the documents.

## How to Use

1. **Basic Model**:
   - Open [`modelo_basico.ipynb`](modelo_basico.ipynb) in Jupyter Notebook.
   - Run the cells to load the `spaCy` model and test the lemmatization-based response classification.

2. **Naive Bayes Model**:
   - Open [`modelo_naive_bayes.ipynb`](modelo_naive_bayes.ipynb) in Jupyter Notebook.
   - Run the cells to train the Naive Bayes model and interact with the chatbot by answering the predefined questions.

3. **BioBERT Model**:
   - Use the script [`Biobert_comparing_model.py`](CHATBOT/Biobert_comparing_model.py) to test the classification capabilities.
   - Examine how different patient responses are categorized by both the pretrained and fine-tuned models.

4. **Information Systems**:
   - Launch the general information chatbot with [`provide_relevant_info.py`](CHATBOT/provide_info/provide_relevant_info.py).
   - For specific scientific information, use the RAG system in [`architecture_Rag.py`](RECOMENTATION_OF_PAPERS/architecture_Rag.py).
   - Upload scientific PDFs to the RAG system to enable document-based information retrieval.

## Future Work
- Integrate more advanced machine learning models for better response classification.
- Expand the training dataset to include a wider variety of responses.
- Deploy the chatbot as a web application for broader accessibility.
- Implement multilingual support for non-English speakers.
- Create a unified interface combining all components.

## Disclaimer
This chatbot is a prototype and should not be used as a substitute for professional medical advice. If you suspect you have melanoma or any other medical condition, please consult a qualified healthcare provider.