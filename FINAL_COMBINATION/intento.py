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
        # Check if vision model exists
        if not os.path.exists("./melanoma_model_1_torch_RESNET50_harvard.pth"):
            st.error("Vision model not found. Please make sure the model is available at './melanoma_vision_model.pth'.")
            return None
        
        # Create a ResNet model (or another architecture that matches your saved model)
        # You'll need to adjust this to match the architecture of your saved model
        model = models.resnet50(pretrained=False)
        
        # Modify the final layer to match your classification problem
        # Example: For the 4 melanoma severity classes
        num_classes = 1  # adjust based on your model
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        
        # Load the saved weights
        model.load_state_dict(torch.load("./melanoma_model_1_torch_RESNET50_harvard.pth", map_location=torch.device('cpu')))
        
        # Set to evaluation mode
        model.eval()
        
        return model
    except Exception as e:
        st.error(f"Error loading vision model: {str(e)}")
        st.error(traceback.format_exc())
        return None

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
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            score = probs[0, pred_class].item()
        
        # Get label
        label_name = id_to_label[pred_class]
        
        # Return full probabilities for all classes
        all_probs = probs[0].tolist()
        class_probs = {id_to_label[i]: prob for i, prob in enumerate(all_probs)}
        
        return label_name, score, class_probs
    except Exception as e:
        st.error(f"Image classification error: {str(e)}")
        st.error(traceback.format_exc())
        return "error", 0.0, {}

# Logic-based fusion of text and image predictions
def combine_predictions(text_probs, image_probs, unique_labels):
    try:
        # Initialize combined scores
        combined_probs = {}
        
        # Combination weights (can be adjusted based on model performance)
        text_weight = 0.4
        image_weight = 0.6
        
        # Combine probabilities for each class
        for label in unique_labels:
            text_prob = text_probs.get(label, 0.0)
            image_prob = image_probs.get(label, 0.0)
            combined_probs[label] = (text_prob * text_weight) + (image_prob * image_weight)
        
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

# Function to visualize classification results
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
    models = ['Text Model', 'Image Model', 'Combined Model']
    scores = [text_score, image_score, combined_score]
    labels = [text_label, image_label, combined_label]
    colors = [color_map.get(label, '#9E9E9E') for label in labels]
    
    # Create dataframe
    chart_data = pd.DataFrame({
        'Model': models,
        'Confidence': scores,
        'Classification': labels
    })
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create barplot
    bars = sns.barplot(x='Model', y='Confidence', data=chart_data, ax=ax, 
                     palette=colors, hue='Model', legend=False)
    
    # Add classification labels
    for i, bar in enumerate(bars.patches):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            labels[i].replace('_', ' ').title(),
            ha='center',
            va='bottom',
            fontsize=12
        )
    
    plt.title('Comparison of confidence and classification between models', fontsize=14)
    plt.ylim(0, 1.1)
    plt.ylabel('Confidence')
    plt.xlabel('Model')
    
    # Show plot
    st.pyplot(fig)

# List of questions
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

# Main function
def main():
    # Title and description
    st.title("Multi-Modal Melanoma Classifier")
    st.write("""
    This advanced application combines a text-based classifier and a computer vision model to provide
    more accurate melanoma risk assessment. Upload an image of the skin lesion and answer questions 
    about its characteristics for a comprehensive analysis.
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
    
    # Sidebar with dataset information
    st.sidebar.title("Dataset Information")
    st.sidebar.write(f"Dataset name: {data.get('name', 'Unknown')}")
    st.sidebar.write(f"Total examples: {len(data.get('data', []))}")
    
    # Show label distribution
    label_counts = {}
    for item in data.get("data", []):
        if "label" in item:
            if item["label"] not in label_counts:
                label_counts[item["label"]] = 0
            label_counts[item["label"]] += 1
    
    if label_counts:
        st.sidebar.subheader("Label Distribution")
        
        # Create DataFrame for distribution
        dist_data = pd.DataFrame({
            'Label': list(label_counts.keys()),
            'Count': list(label_counts.values())
        })
        
        # Sort labels by concern level
        order = ['not_concerning', 'mildly_concerning', 'moderately_concerning', 'highly_concerning']
        if all(label in order for label in dist_data['Label']):
            dist_data['Label'] = pd.Categorical(dist_data['Label'], categories=order, ordered=True)
            dist_data = dist_data.sort_values('Label')
        
        # Define colors for labels
        colors = {
            'not_concerning': '#4CAF50',  # Green
            'mildly_concerning': '#FFEB3B',  # Yellow
            'moderately_concerning': '#FF9800',  # Orange
            'highly_concerning': '#F44336'   # Red
        }
        
        bar_colors = [colors.get(label, '#9E9E9E') for label in dist_data['Label']]
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Updated barplot version
        bars = sns.barplot(x='Label', y='Count', data=dist_data, palette=bar_colors, ax=ax, legend=False)
        
        # Set ticks before changing labels
        plt.xticks(range(len(dist_data)))
        ax.set_xticklabels([label.replace('_', ' ').title() for label in dist_data['Label']])
        
        # Rotate labels
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        st.sidebar.pyplot(fig)
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["Multi-Modal Classification", "Dataset Examples"])
    
    with tab1:
        st.header("Combined Image and Text Analysis")
        
        # Image upload section
        st.subheader("Upload Image of Skin Lesion")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        # Text response section
        st.subheader("Answer Questions About the Lesion")
        
        # Select a question
        selected_question = st.selectbox("Select a question:", questions)
        
        # Response input
        user_response = st.text_input("Your response:", "")
        
        # Submit button
        submit_button = st.button("Analyze")
        
        if submit_button:
            # Check if we have the necessary inputs
            if not user_response and uploaded_file is None:
                st.error("Please provide a text response and/or upload an image.")
            else:
                # Initialize variables to store classification results
                text_label = None
                text_score = 0.0
                text_probs = {}
                image_label = None
                image_score = 0.0
                image_probs = {}
                
                # Text analysis (if text model is loaded and response is provided)
                if text_model_loaded and user_response:
                    text_label, text_score, text_probs = classify_text_response(
                        user_response, text_model, tokenizer, id_to_label
                    )
                
                # Image analysis (if vision model is loaded and image is uploaded)
                if vision_model_loaded and uploaded_file is not None:
                    # Read and process image
                    image = Image.open(uploaded_file).convert('RGB')
                    
                    # Display the uploaded image
                    st.image(image, caption="Uploaded Image", width=300)
                    
                    # Classify image
                    image_label, image_score, image_probs = classify_image(
                        image, vision_model, id_to_label
                    )
                
                # Check if we have at least one valid prediction
                if (text_label is not None and text_label != "error") or (image_label is not None and image_label != "error"):
                    st.subheader("Classification Results")
                    
                    # If we only have text prediction
                    if (text_label is not None and text_label != "error") and (image_label is None or image_label == "error"):
                        st.write("**Text-based classification:**")
                        st.write(f"Classification: {text_label.replace('_', ' ').title()}")
                        st.write(f"Confidence: {text_score:.4f}")
                        
                        combined_label = text_label
                        combined_score = text_score
                        combined_probs = text_probs
                    
                    # If we only have image prediction
                    elif (image_label is not None and image_label != "error") and (text_label is None or text_label == "error"):
                        st.write("**Image-based classification:**")
                        st.write(f"Classification: {image_label.replace('_', ' ').title()}")
                        st.write(f"Confidence: {image_score:.4f}")
                        
                        combined_label = image_label
                        combined_score = image_score
                        combined_probs = image_probs
                    
                    # If we have both predictions
                    else:
                        st.write("**Text-based classification:**")
                        st.write(f"Classification: {text_label.replace('_', ' ').title()}")
                        st.write(f"Confidence: {text_score:.4f}")
                        
                        st.write("**Image-based classification:**")
                        st.write(f"Classification: {image_label.replace('_', ' ').title()}")
                        st.write(f"Confidence: {image_score:.4f}")
                        
                        # Combine predictions
                        combined_label, combined_score, combined_probs = combine_predictions(
                            text_probs, image_probs, unique_labels
                        )
                        
                        st.write("**Combined classification:**")
                        st.write(f"Classification: {combined_label.replace('_', ' ').title()}")
                        st.write(f"Confidence: {combined_score:.4f}")
                    
                    # Visualize results (if we have both predictions)
                    if text_label is not None and image_label is not None:
                        visualize_results(
                            text_label, text_score, 
                            image_label, image_score, 
                            combined_label, combined_score
                        )
                    
                    # Classification explanation
                    st.subheader("Results Interpretation")
                    
                    concern_explanations = {
                        'not_concerning': """
                        **Not concerning**: No warning signs detected. 
                        However, it's important to regularly monitor any changes in the lesion.
                        """,
                        'mildly_concerning': """
                        **Mildly concerning**: Some mild signs detected that warrant follow-up. 
                        It's recommended to periodically observe the lesion and consult a dermatologist 
                        if additional changes are noticed.
                        """,
                        'moderately_concerning': """
                        **Moderately concerning**: Signs detected that require medical evaluation. 
                        It's recommended to schedule an appointment with a dermatologist in the coming weeks
                        for a professional assessment.
                        """,
                        'highly_concerning': """
                        **Highly concerning**: Serious signs detected that require immediate medical attention. 
                        It's recommended to consult a dermatologist as soon as possible for a complete evaluation
                        and possible biopsy.
                        """
                    }
                    
                    # Show explanation for combined or available model
                    st.markdown(concern_explanations.get(combined_label, ""))
                    
                    # Recommendation confidence
                    if text_label == image_label:
                        st.success("Both models agree on the classification, increasing confidence in the assessment.")
                    
                    # Show warning
                    st.warning("""
                    **Important note**: This classification is only an aid tool and does not substitute 
                    professional medical diagnosis. Always consult a dermatologist for proper evaluation.
                    """)
                else:
                    st.error("Could not generate a valid classification. Please check your inputs or try again.")
    
    with tab2:
        st.header("Dataset Examples")
        
        # Check if there's data to show
        if data and "data" in data and data["data"]:
            # Show examples for each category
            st.subheader("Examples by Category")
            
            # Create dictionary of examples by label
            examples_by_label = {}
            for item in data["data"]:
                if "label" in item and "text" in item:
                    if item["label"] not in examples_by_label:
                        examples_by_label[item["label"]] = []
                    examples_by_label[item["label"]].append(item["text"])
            
            # Show examples by category
            for label in unique_labels:
                # Get examples for this label
                examples = examples_by_label.get(label, [])
                # Show up to 5 examples
                example_list = examples[:5]
                
                # Create expander for each category
                with st.expander(f"{label.replace('_', ' ').title()} ({len(examples)} examples)"):
                    if example_list:
                        for i, example in enumerate(example_list):
                            st.write(f"{i+1}. \"{example}\"")
                    else:
                        st.write("No examples available for this category.")
        else:
            st.warning("No examples available in the dataset.")

if __name__ == "__main__":
    main()