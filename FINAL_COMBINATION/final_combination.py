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
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torchvision import models

# Page configuration
st.set_page_config(
    page_title="Melanoma Multi-Modal Classifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        model_path = "./melanoma_model_1_torch_RESNET50_harvard.pth"
        # Check if vision model exists
        if not os.path.exists(model_path):
            st.error(f"Vision model not found. Please make sure the model is available at '{model_path}'.")
            return None
        
        # Create a custom model handler class that deals with the size mismatch
        class CustomResNet(torch.nn.Module):
            def __init__(self, num_classes=1):
                super(CustomResNet, self).__init__()
                # Load the base ResNet50 model
                self.backbone = models.resnet50(pretrained=False)
                # Remove the final fully connected layer
                in_features = self.backbone.fc.in_features
                self.backbone.fc = torch.nn.Identity()
                # Add our own classifier layer
                self.classifier = torch.nn.Linear(in_features, num_classes)
                
            def forward(self, x):
                features = self.backbone(x)
                return self.classifier(features)
        
        # Create our custom model
        model = CustomResNet(num_classes=1)
        
        # Try to load the model
        try:
            # Option 1: Try to load directly (might work if it's just the state dict)
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            
            # Check if it's a complete model or just state_dict
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                # It's a complete model save
                state_dict = state_dict['state_dict']
            
            # Process the state dict to match our model
            # We need to map the original fc.weight to our custom classifier.weight
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('fc.'):
                    # Map fc.weight -> classifier.weight and fc.bias -> classifier.bias
                    new_key = key.replace('fc.', 'classifier.')
                    new_state_dict[new_key] = value
                else:
                    # Add backbone. prefix to other keys
                    new_key = 'backbone.' + key
                    new_state_dict[new_key] = value
            
            # Load the processed state dict
            model.load_state_dict(new_state_dict, strict=False)
            
        except Exception as e:
            st.error(f"Error loading model weights: {str(e)}")
            st.error(traceback.format_exc())
            
            # Fallback: Create a completely new model for demonstration
            model = CustomResNet(num_classes=1)
            st.warning("Using a simulated model for demonstration purposes")
        
        # Set to evaluation mode
        model.eval()
        
        return model
    except Exception as e:
        st.error(f"Error loading vision model: {str(e)}")
        st.error(traceback.format_exc())
        
        # Return a dummy model that always predicts 0.5 probability
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super(DummyModel, self).__init__()
            
            def forward(self, x):
                # Always return a middle probability (0.5)
                return torch.tensor([[0.5]])
        
        dummy_model = DummyModel()
        st.warning("Using a dummy model that returns fixed predictions for demonstration")
        return dummy_model

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
def classify_image(image, model, id_to_label):
    try:
        # Define image transformations (adjust according to your model's requirements)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Preprocess the image
        img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            # For binary model (sigmoid activation)
            if outputs.shape[1] == 1:
                # Apply sigmoid to get probability
                prob = torch.sigmoid(outputs)[0][0].item()
                
                # Map the probability to our classes based on threshold
                # Lower thresholds for higher sensitivity to concerning lesions
                if prob < 0.25:
                    pred_class = 0  # not_concerning
                    class_probs = {
                        id_to_label[0]: 1 - prob,  # not_concerning
                        id_to_label[1]: 0.0,       # mildly_concerning
                        id_to_label[2]: 0.0,       # moderately_concerning
                        id_to_label[3]: prob       # highly_concerning (treat as binary output)
                    }
                elif prob < 0.5:
                    pred_class = 1  # mildly_concerning
                    class_probs = {
                        id_to_label[0]: 0.7 - prob,       # not_concerning
                        id_to_label[1]: prob * 1.2,       # mildly_concerning
                        id_to_label[2]: 0.0,              # moderately_concerning
                        id_to_label[3]: 0.0               # highly_concerning
                    }
                elif prob < 0.75:
                    pred_class = 2  # moderately_concerning
                    class_probs = {
                        id_to_label[0]: 0.0,              # not_concerning
                        id_to_label[1]: 1.0 - prob,       # mildly_concerning
                        id_to_label[2]: prob,             # moderately_concerning
                        id_to_label[3]: 0.0               # highly_concerning
                    }
                else:
                    pred_class = 3  # highly_concerning
                    class_probs = {
                        id_to_label[0]: 0.0,              # not_concerning
                        id_to_label[1]: 0.0,              # mildly_concerning
                        id_to_label[2]: 1.0 - prob,       # moderately_concerning
                        id_to_label[3]: prob              # highly_concerning
                    }
                
                # Get label name
                label_name = id_to_label[pred_class]
                score = class_probs[label_name]
            else:
                # For multi-class model (softmax activation)
                probs = torch.softmax(outputs, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                score = probs[0, pred_class].item()
                
                # Get label name
                label_name = id_to_label[pred_class]
                
                # Get all probabilities
                all_probs = probs[0].tolist()
                class_probs = {id_to_label[i]: prob for i, prob in enumerate(all_probs)}
        
        return label_name, score, class_probs
    except Exception as e:
        st.error(f"Image classification error: {str(e)}")
        st.error(traceback.format_exc())
        return "error", 0.0, {}

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
            # Find highest probability class for each model
            text_max_class = max(text_probs, key=text_probs.get)
            image_max_class = max(image_probs, key=image_probs.get)
            
            # If both models agree, boost that class
            if text_max_class == image_max_class and text_max_class != "error":
                st.success(f"Both models agree on the classification: {text_max_class.replace('_', ' ').title()}")
                
                # Get average confidence
                avg_confidence = (text_probs[text_max_class] + image_probs[text_max_class]) / 2
                
                # Apply confidence boost
                for label in unique_labels:
                    if label == text_max_class:
                        # Boost the agreed class by 20%
                        text_probs[label] = min(1.0, text_probs[label] * 1.2)
                        image_probs[label] = min(1.0, image_probs[label] * 1.2)
        
        # Normalize weights to sum to 1
        total_weight = text_weight + image_weight
        text_weight = text_weight / total_weight
        image_weight = image_weight / total_weight
        
        # Combine probabilities for each class
        for label in unique_labels:
            text_prob = text_probs.get(label, 0.0)
            image_prob = image_probs.get(label, 0.0)
            combined_probs[label] = (text_prob * text_weight) + (image_prob * image_weight)
        
        # Log the weights used
        st.write(f"Ponderación del diagnóstico: Respuestas del paciente ({text_weight:.2f}), Análisis de imagen ({image_weight:.2f})")
        
        # Get the highest probability class
        max_label = max(combined_probs, key=combined_probs.get)
        max_score = combined_probs[max_label]
        
        return max_label, max_score, combined_probs
    except Exception as e:
        st.error(f"Error combining predictions: {str(e)}")
        st.error(traceback.format_exc())
        # Return the image prediction as fallback
        max_label = max(image_probs, key=image_probs.get) if image_probs else "error"
        max_score = image_probs.get(max_label, 0.0)
        return max_label, max_score, image_probs

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
    
    # Create data for visualization
    models = ['Resultados del Cuestionario', 'Análisis de Imagen', 'Diagnóstico Combinado']
    scores = [text_score, image_score, combined_score]
    labels = [text_label, image_label, combined_label]
    colors = [color_map.get(label, '#9E9E9E') for label in labels]
    
    # Create dataframe
    chart_data = pd.DataFrame({
        'Modelo': models,
        'Confianza': scores,
        'Clasificación': labels
    })
    
    # Create layout with 2 subplots: bar chart and pie chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Create barplot on first subplot
    bars = sns.barplot(x='Modelo', y='Confianza', data=chart_data, ax=ax1, 
                     palette=colors, hue='Modelo', legend=False)
    
    # Add classification labels
    for i, bar in enumerate(bars.patches):
        label_text = labels[i].replace('_', ' ').title()
        ax1.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            label_text,
            ha='center',
            va='bottom',
            fontsize=12
        )
    
    ax1.set_title('Confianza por método de diagnóstico', fontsize=14)
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel('Nivel de Confianza')
    ax1.set_xlabel('Método de Análisis')
    
    # Create pie chart for risk distribution on second subplot
    # Normalize and create risk data based on combined result
    risk_labels = ['No preocupante', 'Levemente preocupante', 'Moderadamente preocupante', 'Altamente preocupante']
    
    if combined_label == 'not_concerning':
        risk_values = [0.85, 0.15, 0.0, 0.0]
    elif combined_label == 'mildly_concerning':
        risk_values = [0.25, 0.65, 0.1, 0.0]
    elif combined_label == 'moderately_concerning':
        risk_values = [0.0, 0.2, 0.7, 0.1]
    elif combined_label == 'highly_concerning':
        risk_values = [0.0, 0.0, 0.3, 0.7]
    else:
        risk_values = [0.25, 0.25, 0.25, 0.25]  # Equal distribution for error
    
    risk_colors = ['#4CAF50', '#FFEB3B', '#FF9800', '#F44336']
    
    # Create pie chart
    wedges, texts, autotexts = ax2.pie(
        risk_values, 
        labels=risk_labels, 
        autopct='%1.1f%%',
        colors=risk_colors,
        startangle=90,
        explode=(0.05, 0.05, 0.05, 0.05),
        wedgeprops={'linewidth': 2, 'edgecolor': 'white'}
    )
    
    # Styling pie chart text
    for text in texts:
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_color('white')
    
    ax2.set_title('Distribución de Riesgo', fontsize=14)
    
    plt.tight_layout()
    
    # Show plot
    st.pyplot(fig)
    
    # Create a severity gauge chart
    fig2, ax = plt.subplots(figsize=(10, 4))
    
    # Define the gauge scale based on severity categories
    severity_scale = ['No preocupante', 'Levemente\npreocupante', 'Moderadamente\npreocupante', 'Altamente\npreocupante']
    severity_positions = [0, 0.33, 0.66, 1]
    severity_colors = ['#4CAF50', '#FFEB3B', '#FF9800', '#F44336']
    
    # Determine position based on combined label
    if combined_label == 'not_concerning':
        position = 0.16  # Middle of "not concerning" range
    elif combined_label == 'mildly_concerning':
        position = 0.49  # Middle of "mildly concerning" range
    elif combined_label == 'moderately_concerning':
        position = 0.82  # Middle of "moderately concerning" range
    elif combined_label == 'highly_concerning':
        position = 0.95  # High end of "highly concerning" range
    else:
        position = 0.5  # Middle for error
    
    # Create the gauge background
    for i in range(len(severity_positions)-1):
        ax.axvspan(severity_positions[i], severity_positions[i+1], facecolor=severity_colors[i], alpha=0.3)
    
    # Plot the gauge needle
    ax.arrow(position, 0.5, 0, -0.15, head_width=0.03, head_length=0.1, fc='black', ec='black', linewidth=2)
    ax.scatter(position, 0.5, s=300, color='#333333', zorder=5)
    
    # Add gauge labels
    for i, label in enumerate(severity_scale):
        ax.text(severity_positions[i] + 0.16, 0.25, label, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Set gauge title
    ax.text(0.5, 0.8, 'Indicador de Severidad', ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Format gauge plot
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add marker for current position
    label_text = combined_label.replace('_', ' ').title()
    ax.text(position, 0.6, f"{label_text}\n({combined_score:.2f})", ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Show gauge plot
    st.pyplot(fig2)

# Define the list of questions grouped by categories
questions_by_category = {
    "Crecimiento y Evolución": [
        "¿Ha crecido o cambiado de tamaño la lesión en los últimos meses?",
        "¿Ha notado algún cambio en su forma con el tiempo?",
        "¿Ha cambiado el color de la lesión recientemente?"
    ],
    "Apariencia": [
        "¿Es la lesión más grande que 6mm (aproximadamente el tamaño de una goma de borrar de lápiz)?",
        "¿Luce la lesión diferente a otros lunares o manchas en su cuerpo?"
    ],
    "Síntomas": [
        "¿Le pica la lesión?",
        "¿Sangra la lesión sin haber sido lesionada?",
        "¿Está el área alrededor de la lesión roja o inflamada?",
        "¿Siente dolor o sensibilidad en la lesión?",
        "¿Ha formado la lesión una costra que no sana?"
    ],
    "Factores de riesgo adicionales": [
        "¿Está la lesión expuesta al sol regularmente?",
        "¿Ha tenido quemaduras solares severas en el pasado, especialmente cuando era niño?",
        "¿Tiene antecedentes familiares de melanoma o cáncer de piel?",
        "¿Tiene muchos lunares (más de 50) en su cuerpo?",
        "¿Tiene piel clara que se quema fácilmente con el sol?"
    ]
}

# Flatten the questions list for other functions
all_questions = []
for category, questions in questions_by_category.items():
    all_questions.extend(questions)

# Main function
def main():
    # Title and description
    st.title("Sistema de Diagnóstico de Melanoma Multi-Modal")
    st.write("""
    Esta aplicación avanzada combina el análisis de sus respuestas con un modelo de visión por computadora 
    para proporcionar una evaluación más precisa del riesgo de melanoma. Complete el cuestionario y suba 
    una imagen de la lesión para un análisis completo.
    """)
    
    # Load data
    data = load_data()
    
    if not data or "data" not in data or not data["data"]:
        st.error("No hay datos disponibles para el análisis.")
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
        st.warning("Uno o más modelos no pudieron cargarse. Algunas funcionalidades pueden estar limitadas.")
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Cuestionario y Diagnóstico", "Historial", "Información"])
    
    with tab1:
        # Store the state of the app
        if 'step' not in st.session_state:
            st.session_state.step = 1  # 1: Questionnaire, 2: Image, 3: Results
        
        if 'responses' not in st.session_state:
            st.session_state.responses = {}
        
        if 'text_result' not in st.session_state:
            st.session_state.text_result = None
        
        if 'image_result' not in st.session_state:
            st.session_state.image_result = None
        
        if 'uploaded_image' not in st.session_state:
            st.session_state.uploaded_image = None
        
        # Step 1: Complete questionnaire
        if st.session_state.step == 1:
            st.header("Cuestionario de Evaluación de Lesiones en la Piel")
            st.write("Por favor, responda las siguientes preguntas sobre la lesión en su piel:")
            
            # Create a multi-step form for each category
            all_completed = True
            
            for category, questions in questions_by_category.items():
                with st.expander(f"{category}", expanded=True):
                    st.write(f"**{category}**")
                    
                    for question in questions:
                        # Create a unique key for each question
                        question_key = f"q_{questions_by_category[category].index(question)}_{category}"
                        
                        # Get response (use previous response if exists)
                        response = st.text_input(
                            question,
                            value=st.session_state.responses.get(question_key, ""),
                            key=question_key
                        )
                        
                        # Store response
                        st.session_state.responses[question_key] = response
                        
                        # Check if this question is completed
                        if not response.strip():
                            all_completed = False
            
            # Continue button
            col1, col2 = st.columns([4, 1])
            with col2:
                continue_button = st.button(
                    "Continuar con la imagen" if all_completed else "Continuar con la imagen (algunas preguntas están vacías)",
                    disabled=False,
                    type="primary" if all_completed else "secondary"
                )
                
                if continue_button:
                    # Proceed to image upload
                    st.session_state.step = 2
                    
        
        # Step 2: Upload image
        elif st.session_state.step == 2:
            st.header("Subir Imagen de la Lesión")
            st.write("Por favor, suba una imagen clara de la lesión en su piel:")
            
            # Image upload section
            uploaded_file = st.file_uploader("Elija una imagen...", type=["jpg", "jpeg", "png"])
            
            # Display the image if uploaded
            if uploaded_file is not None:
                st.session_state.uploaded_image = uploaded_file
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="Imagen subida", width=300)
                
                # Buttons for navigation
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    if st.button("← Volver al cuestionario"):
                        st.session_state.step = 1
                        
                
                with col2:
                    if st.button("Realizar análisis", type="primary"):
                        # Process the image and questionnaire
                        st.session_state.step = 3
                        
            else:
                # Only show back button if no image
                if st.button("← Volver al cuestionario"):
                    st.session_state.step = 1
                    
        
        # Step 3: Show results
        elif st.session_state.step == 3:
            st.header("Resultados del Diagnóstico")
            
            # Show loading spinner while processing
            with st.spinner("Analizando datos..."):
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
                    
                    # Display the image
                    st.image(image, caption="Imagen analizada", width=300)
                    
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
                st.subheader("Resultados del Análisis")
                
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
                        st.write("**Análisis del cuestionario:**")
                        st.write(f"Clasificación: {text_label.replace('_', ' ').title()}")
                        st.write(f"Confianza: {text_score:.4f}")
                    
                    with col2:
                        st.write("**Análisis de la imagen:**")
                        st.write(f"Clasificación: {image_label.replace('_', ' ').title()}")
                        st.write(f"Confianza: {image_score:.4f}")
                    
                    # Combine predictions
                    combined_label, combined_score, combined_probs = combine_predictions(
                        text_probs, image_probs, unique_labels, st.session_state.responses
                    )
                    
                    # Show combined results
                    st.markdown("---")
                    st.subheader("Diagnóstico Final")
                    st.markdown(f"**Nivel de preocupación: {combined_label.replace('_', ' ').title()}**")
                    st.markdown(f"**Confianza: {combined_score:.4f}**")
                    
                    # Visualize all results
                    st.markdown("### Comparación de Resultados")
                    visualize_results(
                        text_label, text_score, 
                        image_label, image_score, 
                        combined_label, combined_score
                    )
                    
                # Only text classification
                elif st.session_state.text_result:
                    text_label = st.session_state.text_result["label"]
                    text_score = st.session_state.text_result["score"]
                    
                    st.write("**Análisis del cuestionario:**")
                    st.write(f"Clasificación: {text_label.replace('_', ' ').title()}")
                    st.write(f"Confianza: {text_score:.4f}")
                    
                    st.markdown("---")
                    st.markdown("### Diagnóstico Final")
                    st.markdown(f"**Nivel de preocupación: {text_label.replace('_', ' ').title()}**")
                    st.markdown(f"**Confianza: {text_score:.4f}**")
                    st.warning("Este diagnóstico se basa únicamente en sus respuestas. Para un análisis más preciso, suba una imagen de la lesión.")
                
                # Only image classification
                elif st.session_state.image_result:
                    image_label = st.session_state.image_result["label"]
                    image_score = st.session_state.image_result["score"]
                    
                    st.write("**Análisis de la imagen:**")
                    st.write(f"Clasificación: {image_label.replace('_', ' ').title()}")
                    st.write(f"Confianza: {image_score:.4f}")
                    
                    st.markdown("---")
                    st.markdown("### Diagnóstico Final")
                    st.markdown(f"**Nivel de preocupación: {image_label.replace('_', ' ').title()}**")
                    st.markdown(f"**Confianza: {image_score:.4f}**")
                    st.warning("Este diagnóstico se basa únicamente en la imagen. Para un análisis más preciso, complete el cuestionario.")
                
                # Show detailed explanation based on the final diagnosis
                st.markdown("---")
                st.subheader("Interpretación de los Resultados")
                
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
                    **No preocupante**: No se detectaron signos de advertencia. 
                    Sin embargo, es importante monitorear regularmente cualquier cambio en la lesión.
                    
                    **Recomendaciones:**
                    - Realice autoexámenes regulares cada 3 meses
                    - Proteja su piel del sol con protector solar SPF 30+
                    - Si observa cambios en el futuro, consulte a un dermatólogo
                    """,
                    'mildly_concerning': """
                    **Levemente preocupante**: Se detectaron algunos signos leves que justifican seguimiento. 
                    Se recomienda observar periódicamente la lesión y consultar a un dermatólogo
                    si se notan cambios adicionales.
                    
                    **Recomendaciones:**
                    - Tome fotografías de la lesión mensualmente para monitorear cambios
                    - Programe una revisión con un dermatólogo en los próximos 3 meses
                    - Evite la exposición prolongada al sol y use protector solar
                    """,
                    'moderately_concerning': """
                    **Moderadamente preocupante**: Se detectaron signos que requieren evaluación médica. 
                    Se recomienda programar una cita con un dermatólogo en las próximas semanas
                    para una evaluación profesional.
                    
                    **Recomendaciones:**
                    - Programe una consulta con un dermatólogo en las próximas 2-3 semanas
                    - Tome fotografías de la lesión para documentar su estado actual
                    - Evite irritar o traumatizar la lesión
                    - Proteja la zona de la exposición solar
                    """,
                    'highly_concerning': """
                    **Altamente preocupante**: Se detectaron signos serios que requieren atención médica inmediata. 
                    Se recomienda consultar a un dermatólogo lo antes posible para una evaluación completa
                    y posible biopsia.
                    
                    **Recomendaciones:**
                    - Busque atención dermatológica urgente (en los próximos días)
                    - No intente tratar la lesión por su cuenta
                    - Tome fotografías desde diferentes ángulos para el dermatólogo
                    - Prepare un historial de cuándo notó la lesión y cómo ha cambiado
                    """
                }
                
                # Show explanation for final diagnosis
                if final_label:
                    st.markdown(concern_explanations.get(final_label, ""))
                
                # Add note about model agreement if applicable
                if st.session_state.text_result and st.session_state.image_result:
                    if text_label == image_label:
                        st.success("Ambos métodos de análisis coinciden en la clasificación, lo que aumenta la confianza en el diagnóstico.")
                
                # Show warning about medical advice
                st.warning("""
                **Nota importante**: Esta clasificación es solo una herramienta de ayuda y no sustituye 
                el diagnóstico médico profesional. Siempre consulte a un dermatólogo para una evaluación adecuada.
                """)
                
                # Navigation buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("← Volver al cuestionario"):
                        # Reset results to force recalculation
                        st.session_state.text_result = None
                        st.session_state.image_result = None
                        st.session_state.step = 1
                        
                
                with col2:
                    if st.button("← Volver a la imagen"):
                        # Reset image result to force recalculation
                        st.session_state.image_result = None
                        st.session_state.step = 2
                        
            else:
                st.error("No se pudo generar una clasificación válida. Por favor revise sus entradas o intente nuevamente.")
                
                # Back button
                if st.button("← Volver al inicio"):
                    st.session_state.step = 1
                    

if __name__ == "__main__":
    main()