import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import torch
import os
import traceback
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Page configuration
st.set_page_config(
    page_title="Melanoma Classifier",
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

# Function to load model and tokenizer
@st.cache_resource
def load_model(label_to_id, id_to_label):
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("rjac/biobert-ICD10-L3-mimic")
        
        # Correct label configuration
        num_labels = len(label_to_id)
        id2label = {str(i): label for i, label in id_to_label.items()}
        label2id = {label: str(i) for label, i in label_to_id.items()}
        
        # Check if fine-tuned model exists
        if not os.path.exists("./finetuned_model"):
            st.error("Fine-tuned model not found. Please make sure the model is available at './finetuned_model'.")
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
            st.error(f"Could not load fine-tuned model: {str(e)}")
            st.error(traceback.format_exc())
            return None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error(traceback.format_exc())
        return None, None

# Function to classify a response
def classify_response(response, model, tokenizer, id_to_label):
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
        
        return label_name, score
    except Exception as e:
        st.error(f"Classification error: {str(e)}")
        st.error(traceback.format_exc())
        return "error", 0.0

# Function to visualize results
def visualize_results(label, score):
    # Define color mapping
    color_map = {
        'not_concerning': '#4CAF50',  # Green
        'mildly_concerning': '#FFEB3B',  # Yellow
        'moderately_concerning': '#FF9800',  # Orange
        'highly_concerning': '#F44336',   # Red
        'error': '#9E9E9E'  # Gray for errors
    }
    
    # Create data for visualization
    chart_data = pd.DataFrame({
        'Model': ['Fine-tuned Model'],
        'Confidence': [score],
        'Classification': [label]
    })
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Updated barplot version
    bars = sns.barplot(x='Model', y='Confidence', data=chart_data, ax=ax, 
                      color=color_map.get(label, '#9E9E9E'), legend=False)
    
    # Add classification label
    ax.text(
        bars.patches[0].get_x() + bars.patches[0].get_width()/2,
        bars.patches[0].get_height() + 0.02,
        label.replace('_', ' ').title(),
        ha='center',
        va='bottom',
        fontsize=12
    )
    
    plt.title('Classification confidence', fontsize=14)
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
    st.title("Melanoma Severity Classifier")
    st.write("""
    This application uses a fine-tuned transformer model to classify responses about skin lesions
    and evaluate potential melanoma risk.
    """)
    
    # Load data
    data = load_data()
    
    if not data or "data" not in data or not data["data"]:
        st.error("No data available for analysis.")
        return
    
    unique_labels, label_to_id, id_to_label = get_labels(data)
    
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
    
    # Load model
    tokenizer, model = load_model(label_to_id, id_to_label)
    
    if tokenizer and model:
        # Create tabs for different sections
        tab1, tab2 = st.tabs(["Classification", "Dataset Examples"])
        
        with tab1:
            st.header("Response Classification")
            
            # Select a question
            selected_question = st.selectbox("Select a question:", questions)
            
            # Response input
            user_response = st.text_input("Your response:", "")
            
            if st.button("Classify response") and user_response:
                # Classify with model
                label, score = classify_response(
                    user_response, model, tokenizer, id_to_label
                )
                
                # Show results
                st.subheader("Classification Results")
                st.write(f"Classification: {label.replace('_', ' ').title()}")
                st.write(f"Confidence: {score:.4f}")
                
                # Visualize results
                visualize_results(label, score)
                
                # Classification explanation
                st.subheader("Results Interpretation")
                
                concern_explanations = {
                    'not_concerning': """
                    **Not concerning**: No warning signs detected in this response. 
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
                
                # Show explanation
                st.markdown(concern_explanations.get(label, ""))
                
                # Show warning
                st.warning("""
                **Important note**: This classification is only an aid tool and does not substitute 
                professional medical diagnosis. Always consult a dermatologist for proper evaluation.
                """)
        
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
    else:
        st.error("Could not load the fine-tuned model. Please verify installation.")

if __name__ == "__main__":
    main()