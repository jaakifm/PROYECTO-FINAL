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

# Function to load models and tokenizer
@st.cache_resource
def load_models(label_to_id, id_to_label):
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("rjac/biobert-ICD10-L3-mimic")
        
        # Correct label configuration
        num_labels = len(label_to_id)
        id2label = {str(i): label for i, label in id_to_label.items()}
        label2id = {label: str(i) for label, i in label_to_id.items()}
        
        # Load pretrained model
        pretrained_model = AutoModelForSequenceClassification.from_pretrained(
            "rjac/biobert-ICD10-L3-mimic", 
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True  # Ignore mismatched layer sizes
        )
        
        # Load fine-tuned model (if exists)
        finetuned_model = None
        if os.path.exists("./finetuned_model"):
            try:
                # Try to load model with same label configuration
                finetuned_model = AutoModelForSequenceClassification.from_pretrained(
                    "./finetuned_model",
                    num_labels=num_labels,
                    id2label=id2label,
                    label2id=label2id
                )
            except Exception as e:
                st.warning(f"Could not load fine-tuned model: {str(e)}")
                finetuned_model = None
        
        return tokenizer, pretrained_model, finetuned_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.error(traceback.format_exc())
        return None, None, None

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
def visualize_results(pretrained_label, pretrained_score, finetuned_label=None, finetuned_score=None):
    # Define color mapping
    color_map = {
        'not_concerning': '#4CAF50',  # Green
        'mildly_concerning': '#FFEB3B',  # Yellow
        'moderately_concerning': '#FF9800',  # Orange
        'highly_concerning': '#F44336',   # Red
        'error': '#9E9E9E'  # Gray for errors
    }
    
    # Create data for visualization
    models = ['Pretrained']
    scores = [pretrained_score]
    labels = [pretrained_label]
    colors = [color_map.get(pretrained_label, '#9E9E9E')]
    
    if finetuned_label is not None and finetuned_score is not None:
        models.append('Fine-tuned')
        scores.append(finetuned_score)
        labels.append(finetuned_label)
        colors.append(color_map.get(finetuned_label, '#9E9E9E'))
    
    # Create dataframe
    chart_data = pd.DataFrame({
        'Model': models,
        'Confidence': scores,
        'Classification': labels
    })
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Updated barplot version
    bars = sns.barplot(x='Model', y='Confidence', hue='Model', data=chart_data, ax=ax, palette=colors, legend=False)
    
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
    st.title("Melanoma Severity Classifier")
    st.write("""
    This application uses transformer models to classify responses about skin lesions
    and evaluate potential melanoma risk. It compares a pretrained model (BioBERT) with a
    fine-tuned model specifically for this use case.
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
        bars = sns.barplot(x='Label', y='Count', hue='Label', data=dist_data, palette=bar_colors, ax=ax, legend=False)
        
        # Set ticks before changing labels
        plt.xticks(range(len(dist_data)))
        ax.set_xticklabels([label.replace('_', ' ').title() for label in dist_data['Label']])
        
        # Rotate labels
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        st.sidebar.pyplot(fig)
    
    # Load models
    tokenizer, pretrained_model, finetuned_model = load_models(label_to_id, id_to_label)
    
    if tokenizer and pretrained_model:
        # Create tabs for different sections
        tab1, tab2, tab3 = st.tabs(["Single Classification", "Model Comparison", "Dataset Examples"])
        
        with tab1:
            st.header("Response Classification")
            
            # Select a question
            selected_question = st.selectbox("Select a question:", questions)
            
            # Response input
            user_response = st.text_input("Your response:", "")
            
            if st.button("Classify response") and user_response:
                # Classify with pretrained model
                pretrained_label, pretrained_score = classify_response(
                    user_response, pretrained_model, tokenizer, id_to_label
                )
                
                # Show results
                st.subheader("Classification Results")
                
                # Pretrained model
                st.write("**Pretrained model (BioBERT):**")
                st.write(f"Classification: {pretrained_label.replace('_', ' ').title()}")
                st.write(f"Confidence: {pretrained_score:.4f}")
                
                # If fine-tuned model exists
                if finetuned_model:
                    # Classify with fine-tuned model
                    finetuned_label, finetuned_score = classify_response(
                        user_response, finetuned_model, tokenizer, id_to_label
                    )
                    
                    # Fine-tuned model
                    st.write("**Fine-tuned model:**")
                    st.write(f"Classification: {finetuned_label.replace('_', ' ').title()}")
                    st.write(f"Confidence: {finetuned_score:.4f}")
                    
                    # Visualize results
                    visualize_results(pretrained_label, pretrained_score, finetuned_label, finetuned_score)
                else:
                    # Only visualize pretrained model
                    visualize_results(pretrained_label, pretrained_score)
                
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
                
                # Show explanation for pretrained model
                st.markdown(concern_explanations.get(pretrained_label, ""))
                
                # Show warning
                st.warning("""
                **Important note**: This classification is only an aid tool and does not substitute 
                professional medical diagnosis. Always consult a dermatologist for proper evaluation.
                """)
        
        with tab2:
            st.header("Model Comparison")
            
            # Load comparison metrics if they exist
            comparison_file = 'model_comparison_results.json'
            if os.path.exists(comparison_file):
                try:
                    with open(comparison_file, 'r') as f:
                        metrics_data = json.load(f)
                    
                    # Comparative metrics
                    if finetuned_model:
                        metrics = {
                            'Accuracy': [metrics_data['pretrained']['accuracy'], metrics_data['finetuned']['accuracy']],
                            'Precision': [metrics_data['pretrained']['precision'], metrics_data['finetuned']['precision']],
                            'Recall': [metrics_data['pretrained']['recall'], metrics_data['finetuned']['recall']],
                            'F1-Score': [metrics_data['pretrained']['f1'], metrics_data['finetuned']['f1']]
                        }
                        
                        metrics_df = pd.DataFrame(metrics, index=['Pretrained model', 'Fine-tuned model'])
                        
                        # Show metrics table
                        st.subheader("Performance Metrics")
                        st.table(metrics_df)
                        
                        # Metrics visualization
                        st.subheader("Metrics Visualization:")
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        metrics_df.T.plot(kind='bar', ax=ax)
                        plt.title('Model metrics comparison')
                        plt.ylim(0, 1)
                        plt.ylabel('Score')
                        plt.xlabel('Metric')
                        plt.legend(title='Model')
                        
                        st.pyplot(fig)
                        
                        # Improvements explanation
                        st.subheader("Improvement Analysis")
                        
                        # Calculate improvement percentages
                        accuracy_improvement = (metrics_data['finetuned']['accuracy'] - metrics_data['pretrained']['accuracy']) * 100
                        precision_improvement = (metrics_data['finetuned']['precision'] - metrics_data['pretrained']['precision']) * 100
                        recall_improvement = (metrics_data['finetuned']['recall'] - metrics_data['pretrained']['recall']) * 100
                        f1_improvement = (metrics_data['finetuned']['f1'] - metrics_data['pretrained']['f1']) * 100
                        
                        st.write(f"""
                        The fine-tuned model shows improvements in all evaluated metrics:
                        
                        - **Accuracy**: {accuracy_improvement:.1f}% increase in correctly classifying all responses.
                        - **Precision**: {precision_improvement:.1f}% improvement in accurately identifying positive cases.
                        - **Recall**: {recall_improvement:.1f}% increase in ability to identify all positive cases.
                        - **F1-Score**: {f1_improvement:.1f}% improvement in harmonic mean between precision and recall.
                        
                        These improvements demonstrate the value of domain-specific fine-tuning for melanoma
                        compared to the base BioBERT pretrained model.
                        """)
                    else:
                        st.info("Fine-tuned model not available for comparison.")
                except Exception as e:
                    st.error(f"Error loading comparison metrics: {str(e)}")
                    st.error(traceback.format_exc())
            else:
                st.info("No comparative data available. Run model training and evaluation first.")
        
        with tab3:
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
        st.error("Could not load models. Please verify installation.")

if __name__ == "__main__":
    main()