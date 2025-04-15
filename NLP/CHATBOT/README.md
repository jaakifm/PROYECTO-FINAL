# Melanoma Diagnosis Chatbot

This project aims to develop a chatbot to acquire knowledge from patients about melanomas. The chatbot is designed to ask questions and classify responses to assist in identifying potential melanoma symptoms. The project progresses through different stages, starting with a basic model and advancing to a more sophisticated models.

## Project Overview

### 1. **Basic Model with Lemmatization**
The first step in the project involves creating a basic chatbot using lemmatization to analyze user responses. This model is implemented in the script [`modelo_basico.ipynb`](modelo_basico.ipynb). 

- **Key Features**:
  - Utilizes the `spaCy` library for natural language processing.
  - Lemmatizes user responses to identify key words such as "yes" or "no."
  - Returns a simple classification of responses as "yes," "no," or "Sorry, I don't understand."

### 2. **Rule Based Model**



### 3. **Naive Bayes Model**
The second stage involves building a more advanced chatbot using a Naive Bayes classification model. This model is implemented in the script [`modelo_naive_bayes.ipynb`](modelo_naive_bayes.ipynb).

- **Key Features**:
  - Uses the `nltk` library for tokenization, stopword removal, and classification.
  - Trains a Naive Bayes classifier on a dataset of labeled responses.
  - Asks a predefined set of questions about melanoma symptoms and classifies user responses as "positive," "negative," or "I don't understand."


  ![CHATBOT_NAIVE_BAYES](chatbot_naive_bayes.png)



  

## How to Use

1. **Basic Model**:
   - Open [`modelo_basico.ipynb`](modelo_basico.ipynb) in Jupyter Notebook.
   - Run the cells to load the `spaCy` model and test the lemmatization-based response classification.

2. **Naive Bayes Model**:
   - Open [`modelo_naive_bayes.ipynb`](modelo_naive_bayes.ipynb) in Jupyter Notebook.
   - Run the cells to train the Naive Bayes model and interact with the chatbot by answering the predefined questions.

## Future Work
- Integrate more advanced machine learning models for better response classification.
- Expand the training dataset to include a wider variety of responses.
- Deploy the chatbot as a web application for broader accessibility.

## Disclaimer
This chatbot is a prototype and should not be used as a substitute for professional medical advice. If you suspect you have melanoma or any other medical condition, please consult a qualified healthcare provider.