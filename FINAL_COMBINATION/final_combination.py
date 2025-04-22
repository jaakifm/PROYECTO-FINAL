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

# Page configuration
st.set_page_config(
    page_title="Melanoma Multi-Modal Classifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to transcribe audio with Whisper - load model only when needed
def transcribe_with_whisper(audio_path):
    """Load whisper model and transcribe audio file"""
    # Import whisper here to avoid early initialization issues
    import whisper
    
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
        st.write(f"Diagnostic weighting: Patient responses ({text_weight:.2f}), Image analysis ({image_weight:.2f})")
        
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
    
    # Ensure all labels are strings and handle None values
    text_label = str(text_label) if text_label is not None else "error"
    image_label = str(image_label) if image_label is not None else "error"
    combined_label = str(combined_label) if combined_label is not None else "error"
    
    # Create data for visualization
    models = ['Questionnaire Results', 'Image Analysis', 'Combined Diagnosis']
    scores = [text_score, image_score, combined_score]
    labels = [text_label, image_label, combined_label]
    colors = [color_map.get(label, '#9E9E9E') for label in labels]
    
    # Create dataframe
    chart_data = pd.DataFrame({
        'Model': models,
        'Confidence': scores,
        'Classification': labels
    })
    
    # Create a bar chart for confidence levels
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create barplot
    bars = sns.barplot(x='Model', y='Confidence', data=chart_data, ax=ax, palette=colors, hue='Classification')
    
    # Add classification labels - with safety checks
    for i, bar in enumerate(bars.patches):
        if i < len(labels):  # Ensure we don't go out of bounds
            try:
                label_text = labels[i].replace('_', ' ').title()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    label_text,
                    ha='center',
                    va='bottom',
                    fontsize=12
                )
            except (AttributeError, IndexError):
                # If there's any issue with label formatting, use a safe default
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    "Unknown",
                    ha='center',
                    va='bottom',
                    fontsize=12
                )
    
    ax.set_title('Confidence by Diagnostic Method', fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Confidence Level')
    ax.set_xlabel('Analysis Method')
    
    plt.tight_layout()
    
    # Show plot
    st.pyplot(fig)
    
    # Create a severity gauge chart
    fig2, ax = plt.subplots(figsize=(10, 4))
    
    # Define the gauge scale based on severity categories
    severity_scale = ['Not Concerning', 'Mildly\nConcerning', 'Moderately\nConcerning', 'Highly\nConcerning']
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
    tab1, tab2 = st.tabs(["Questionnaire & Diagnosis", "Results"])
    
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
        
        # Step 1: Complete questionnaire
        if st.session_state.step == 1:
            st.header("Skin Lesion Assessment Questionnaire")
            st.write("Please answer the following questions about the skin lesion. You can type or record your answers:")
            
            # Toggle for input method selection
            input_method = st.radio("Select your preferred input method:", ["Text", "Voice"], horizontal=True)
            
            # Create a multi-step form for each category
            all_completed = True
            
            for category, questions in questions_by_category.items():
                with st.expander(f"{category}", expanded=True):
                    st.write(f"**{category}**")
                    
                    for question in questions:
                        # Create a unique key for each question
                        question_key = f"q_{questions_by_category[category].index(question)}_{category}"
                        
                        # Display the question
                        st.markdown(f"**{question}**")
                        
                        # Handle different input methods
                        if input_method == "Text":
                            # Get response (use previous response if exists)
                            response = st.text_input(
                                "Type your answer",
                                value=st.session_state.responses.get(question_key, ""),
                                key=f"text_{question_key}"
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
                                        st.success(f"Transcribed: {transcription}")
                            
                            with col2:
                                # Display previous response or transcribed text
                                response = st.session_state.responses.get(question_key, "")
                                
                                # Allow editing the transcribed text
                                response = st.text_input(
                                    "Verify or edit your answer",
                                    value=response,
                                    key=f"edit_{question_key}"
                                )
                        
                        # Store response
                        st.session_state.responses[question_key] = response
                        
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
                st.image(image, caption="Uploaded Image", width=300)
                
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
                    
                    # Display the image
                    st.image(image, caption="Analyzed Image", width=300)
                    
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
                            
                            # Rerun to update UI
                            st.experimental_rerun()
            
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
if __name__ == "__main__":
    main()