import streamlit as st
import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import os
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Page configuration
st.set_page_config(
    page_title="Melanoma Diagnosis System",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS styles
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .high-risk {
        background-color: rgba(255, 0, 0, 0.1);
        border: 2px solid red;
    }
    .medium-risk {
        background-color: rgba(255, 165, 0, 0.1);
        border: 2px solid orange;
    }
    .low-risk {
        background-color: rgba(0, 128, 0, 0.1);
        border: 2px solid green;
    }
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("""
    <div class="header-container">
        <h1>Integrated Melanoma Diagnosis System</h1>
    </div>
    <p>This application combines textual analysis and computer vision to assess melanoma risk.</p>
""", unsafe_allow_html=True)

# Class for the computer vision model
class MelanomaModel(nn.Module):
    def __init__(self, num_classes=2):
        super(MelanomaModel, self).__init__()
        # We use a pretrained model as the base (ResNet-50)
        self.base_model = models.resnet50(pretrained=False)
        # Modify the last layer for our binary classification problem
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.base_model(x)

# Function to determine device (GPU or CPU)
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_vision_model():
    try:
        device = get_device()
        model = MelanomaModel(num_classes=2)
        
        # Convert to absolute path with correct separators
        model_path = os.path.abspath(os.path.join( "best_model.pth"))
        
        # Check if file exists
        if not os.path.isfile(model_path):
            st.error(f"Model file not found: {model_path}")
            return None, None
            
        st.info(f"Attempting to load vision model from: {model_path}")
        
        # Load model to the appropriate device
        if device.type == 'cuda':
            model.load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            
        model = model.to(device)
        model.eval()
        
        st.sidebar.success(f"Vision model loaded on {device.type.upper()}")
        return model, device
    except Exception as e:
        st.error(f"Error loading vision model: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None

# Function to load the transformers model
@st.cache_resource
def load_text_model():
    try:
        device = get_device()
        
        # Define label mapping (needed for model loading)
        num_labels = 4
        id2label = {0: "not_concerning", 1: "mildly_concerning", 2: "moderately_concerning", 3: "higly_concerning"}
        label2id = {0: "not_concerning", 1: "mildly_concerning", 2: "moderately_concerning", 3: "higly_concerning"}
        
        # Path alternatives to try
        paths_to_try = [
            os.path.abspath(os.path.join( "finetuned_model")),
            os.path.abspath("./CHATBOT/finetuned_model")
        ]
        
        # Try each path
        model = None
        tokenizer = None
        
        for path in paths_to_try:
            st.info(f"Attempting to load model from: {path}")
            
            if os.path.exists(path):
                st.success(f"✅ Path exists: {path}")
                
                try:
                    # Load tokenizer first
                    tokenizer = AutoTokenizer.from_pretrained(
                        path,
                        local_files_only=True
                    )
                    
                    # Load model with label configuration
                    model = AutoModelForSequenceClassification.from_pretrained(
                        path,
                        num_labels=num_labels,
                        id2label=id2label,
                        label2id=label2id,
                        local_files_only=True
                    )
                    
                    st.success(f"✅ Successfully loaded model from: {path}")
                    break
                    
                except Exception as e:
                    st.warning(f"Could not load model from {path}: {str(e)}")
                    continue
            else:
                st.warning(f"❌ Path does not exist: {path}")
        
        if model is None or tokenizer is None:
            st.error("Failed to load model from any of the attempted paths")
            return None, None, None
        
        model = model.to(device)
        model.eval()
        
        st.sidebar.success(f"Text model loaded on {device.type.upper()}")
        return tokenizer, model, device
        
    except Exception as e:
        st.error(f"Error loading text model: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None

# Transformations for the images
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# ABCDE questions and other risk factors
questions = [
    # Growth and Evolution (E in ABCDE)
    "Has the lesion grown or changed in size in recent months?",
    "Have you noticed any change in its shape over time?",
    "Has the color of the lesion changed recently?",
    
    # Appearance (A, B, C, D in ABCDE)
    "Is the lesion larger than 6mm (about the size of a pencil eraser)?",
    "Does the lesion look different from other moles or spots on your body?",
    
    # Symptoms
    "Is the lesion itchy?",
    "Does the lesion bleed without being injured?",
    "Is the area around the lesion red or swollen?",
    "Do you feel pain or tenderness in the lesion?",
    "Has the lesion formed a scab or crust that doesn't heal?",
    
    # Additional risk factors
    "Is the lesion exposed to the sun regularly?",
    "Have you had severe sunburns in the past, especially as a child?",
    "Do you have a family history of melanoma or skin cancer?",
    "Do you have many moles (more than 50) on your body?",
    "Do you have fair skin that burns easily in the sun?"
]

# Function to process responses with the text model
def analyze_responses(responses, tokenizer, model, device):
    try:
        # Combine all responses into a single text
        full_text = " ".join([f"Question: {q} Answer: {r}" for q, r in zip(questions, responses)])
        
        # Tokenize the text
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Process with the model
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
            
        # We assume the model returns the risk probability (0: low, 1: high)
        risk_score = predictions[0][1].item()  # Probability of high risk
        
        return risk_score
    except Exception as e:
        st.error(f"Error analyzing responses: {e}")
        return 0.5  # Neutral value in case of error

# Function to process the image with the vision model
def analyze_image(image, model, transform, device):
    try:
        # Preprocess the image
        image_tensor = transform(image).unsqueeze(0)
        
        # Move tensor to the same device as the model
        image_tensor = image_tensor.to(device)
        
        # Process with the model
        with torch.no_grad():
            outputs = model(image_tensor)
            predictions = torch.softmax(outputs, dim=1)
            
        # We assume the model returns the melanoma probability (0: benign, 1: malignant)
        melanoma_probability = predictions[0][1].item()
        
        return melanoma_probability
    except Exception as e:
        st.error(f"Error analyzing image: {e}")
        return 0.5  # Neutral value in case of error

# Function to combine results and get final diagnosis
def get_combined_diagnosis(text_score, image_score):
    # We weight both models (weights can be adjusted according to confidence in each model)
    text_weight = 0.4
    image_weight = 0.6
    
    combined_score = (text_score * text_weight) + (image_score * image_weight)
    
    if combined_score >= 0.7:
        risk_level = "High"
        recommendation = "It is recommended to consult a dermatologist immediately."
        css_class = "high-risk"
    elif combined_score >= 0.4:
        risk_level = "Medium"
        recommendation = "It is recommended to schedule an appointment with a dermatologist in the coming weeks."
        css_class = "medium-risk"
    else:
        risk_level = "Low"
        recommendation = "Continue monitoring the lesion and consult a dermatologist if you notice changes."
        css_class = "low-risk"
    
    return {
        "text_score": text_score,
        "image_score": image_score,
        "combined_score": combined_score,
        "risk_level": risk_level,
        "recommendation": recommendation,
        "css_class": css_class
    }

# Main function
def main():
    # Add GPU info in sidebar
    st.sidebar.title("System Information")
    if torch.cuda.is_available():
        gpu_info = f"GPU: {torch.cuda.get_device_name(0)}"
        st.sidebar.success(f"✅ CUDA available: {gpu_info}")
    else:
        st.sidebar.warning("⚠️ CUDA not available. Using CPU.")
    
    # Load models
    vision_model, vision_device = load_vision_model()
    tokenizer, text_model, text_device = load_text_model()
    transform = get_transforms()
    
    # Verify that the models loaded correctly
    models_loaded = vision_model is not None and text_model is not None and tokenizer is not None
    
    if not models_loaded:
        st.warning("Could not load models. Please check the paths and try again.")
        return
    
    # Create tabs for the different sections
    tab1, tab2, tab3 = st.tabs(["Questionnaire", "Image Analysis", "Results"])
    
    # Variables to store responses and image
    if 'responses' not in st.session_state:
        st.session_state.responses = [""] * len(questions)
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    # Questionnaire tab
    with tab1:
        st.subheader("Skin Lesion Questionnaire")
        st.write("Please answer the following questions about the lesion:")
        
        # Display questions
        for i, question in enumerate(questions):
            st.session_state.responses[i] = st.text_input(
                question, 
                value=st.session_state.responses[i],
                key=f"question_{i}"
            )
        
        # Button to analyze responses
        if st.button("Save Responses"):
            st.success("Responses saved. Please continue with the image analysis.")
    
    # Image analysis tab
    with tab2:
        st.subheader("Upload Lesion Image")
        st.write("Please upload a clear image of the skin lesion:")
        
        uploaded_file = st.file_uploader("Select an image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded image", width=300)
            st.session_state.uploaded_image = image
            
            st.success("Image successfully uploaded. Proceed to view the results.")
    
    # Results tab
    with tab3:
        st.subheader("Diagnosis Results")
        
        # Verify that we have data to analyze
        all_questions_answered = all(st.session_state.responses)
        image_uploaded = st.session_state.uploaded_image is not None
        
        if not all_questions_answered:
            st.warning("Please answer all questions in the Questionnaire tab.")
        
        if not image_uploaded:
            st.warning("Please upload an image in the Image Analysis tab.")
        
        # Button to generate diagnosis
        if st.button("Generate Diagnosis") and all_questions_answered and image_uploaded:
            with st.spinner("Analyzing data..."):
                # Analyze textual responses
                start_time = time.time()
                text_score = analyze_responses(st.session_state.responses, tokenizer, text_model, text_device)
                text_time = time.time() - start_time
                
                # Analyze image
                start_time = time.time()
                image_score = analyze_image(st.session_state.uploaded_image, vision_model, transform, vision_device)
                image_time = time.time() - start_time
                
                # Combine results
                st.session_state.results = get_combined_diagnosis(text_score, image_score)
                st.session_state.results["text_time"] = text_time
                st.session_state.results["image_time"] = image_time
        
        # Display results if available
        if st.session_state.results:
            results = st.session_state.results
            
            # Display results in a container with style according to risk level
            st.markdown(f"""
                <div class="result-box {results['css_class']}">
                    <h2>Risk level: {results['risk_level']}</h2>
                    <h3>Recommendation: {results['recommendation']}</h3>
                    <p>Textual analysis result: {results['text_score']:.2f} (processed in {results['text_time']:.2f} seconds)</p>
                    <p>Image analysis result: {results['image_score']:.2f} (processed in {results['image_time']:.2f} seconds)</p>
                    <p>Combined score: {results['combined_score']:.2f}</p>
                    <p><b>Note:</b> This system is only an assistive tool and does not replace professional diagnosis.</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Display risk chart
            st.subheader("Risk Distribution")
            
            # Data for the chart
            chart_data = {
                'Category': ['Text analysis', 'Image analysis', 'Combined score'],
                'Value': [results['text_score'], results['image_score'], results['combined_score']]
            }
            
            # Create bar chart
            st.bar_chart(
                chart_data, 
                x='Category', 
                y='Value',
                color=['#FF9999', '#99CCFF', '#99FF99']
            )
            
            # Important warning
            st.warning("""
                **IMPORTANT**: This system is only an assistive tool and does not replace 
                professional clinical evaluation. Always consult with a dermatologist for a definitive diagnosis.
            """)

if __name__ == "__main__":
    main()