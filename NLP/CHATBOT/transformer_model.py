#CHATBOT BASED ON A TRANSFORMER MODEL


import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Page configuration
st.set_page_config(
    page_title="Melanoma Risk Assessment System",
    page_icon="🔬",
    layout="wide"
)

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("pritamdeka/BioBert-PubMed200kRCT")
    model = AutoModelForSequenceClassification.from_pretrained("pritamdeka/BioBert-PubMed200kRCT")
    return tokenizer, model

tokenizer, model=load_model()

# Define severity and certainty categories
SEVERITY_CATEGORIES = ["not concerning", "slightly concerning", "moderately concerning", "highly concerning"]
CERTAINTY_CATEGORIES = ["unknown", "very uncertain", "somewhat uncertain", "quite certain", "completely certain"]

# Define questions by category
questions = {
    "Evolution": [
        {"id": "size_change", "text": "Have you noticed changes in the size of the mole or lesion?", "weight": 0.85},
        {"id": "color_change", "text": "Have you noticed changes in the color of the mole or lesion?", "weight": 0.90},
        {"id": "shape_change", "text": "Have you noticed changes in the shape of the mole or lesion?", "weight": 0.80},
        {"id": "rapid_growth", "text": "Has it grown rapidly in the last few months?", "weight": 0.75}
    ],
    "Appearance": [
        {"id": "asymmetry", "text": "Is the mole or lesion asymmetric (different from one side to the other)?", "weight": 0.75},
        {"id": "border", "text": "Does it have irregular or poorly defined borders?", "weight": 0.70},
        {"id": "color_variety", "text": "Does it present various colors or shades?", "weight": 0.80},
        {"id": "diameter", "text": "Does it have a diameter larger than 6mm (size of a pencil eraser)?", "weight": 0.65}
    ],
    "Symptoms": [
        {"id": "bleeding", "text": "Has it bled without apparent reason?", "weight": 0.85},
        {"id": "itching", "text": "Do you feel itching in the area?", "weight": 0.60},
        {"id": "pain", "text": "Do you feel pain in the area?", "weight": 0.65},
        {"id": "ulceration", "text": "Has it ulcerated or formed a crust?", "weight": 0.75}
    ],
    "Risk Factors": [
        {"id": "sun_exposure", "text": "Have you had intense sun exposure or sunburns?", "weight": 0.50},
        {"id": "family_history", "text": "Do you have a family history of melanoma?", "weight": 0.55},
        {"id": "previous_skin_cancer", "text": "Have you had skin cancer previously?", "weight": 0.60},
        {"id": "many_moles", "text": "Do you have many moles (more than 50)?", "weight": 0.45}
    ]
}

# Function to classify the response
def classify_response(response, model, tokenizer):
    # If the response is empty or irrelevant
    if not response or len(response.strip()) < 3:
        return {"severity_idx": 0, "severity": SEVERITY_CATEGORIES[0], 
                "certainty_idx": 0, "certainty": CERTAINTY_CATEGORIES[0]}
    
    # Preprocess the response
    inputs = tokenizer(response, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # Get the prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Simulate classification in two dimensions
    # In a real case, you would need a model specifically trained for these classifications
    # Here we are simulating the output for demonstration
    
    # Probabilities for severity (using softmax on the first 4 dimensions)
    severity_probs = torch.nn.functional.softmax(logits[0, :4], dim=0).numpy()
    severity_idx = np.argmax(severity_probs)
    
    # Probabilities for certainty (using softmax on the next 5 dimensions)
    # In a real model, this would be part of the training
    certainty_seed = sum([ord(c) for c in response]) % 100  # Simulation for demo
    certainty_probs = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
    if "sure" in response.lower() or "definitely" in response.lower():
        certainty_probs = np.array([0.05, 0.05, 0.1, 0.3, 0.5])
    elif "think" in response.lower() or "maybe" in response.lower():
        certainty_probs = np.array([0.05, 0.1, 0.5, 0.25, 0.1])
    elif "don't know" in response.lower() or "uncertain" in response.lower():
        certainty_probs = np.array([0.2, 0.4, 0.3, 0.05, 0.05])
    
    certainty_idx = np.argmax(certainty_probs)
    
    return {
        "severity_idx": int(severity_idx),
        "severity": SEVERITY_CATEGORIES[severity_idx],
        "certainty_idx": int(certainty_idx),
        "certainty": CERTAINTY_CATEGORIES[certainty_idx]
    }

# Calculate weighted score
def calculate_weighted_score(responses, questions_data):
    total_score = 0
    max_possible_score = 0
    
    for q in questions_data:
        if q["id"] in responses:
            response = responses[q["id"]]
            # Get severity and certainty values (0-3 for severity, 0-4 for certainty)
            severity_factor = response["severity_idx"] / (len(SEVERITY_CATEGORIES) - 1)
            certainty_factor = response["certainty_idx"] / (len(CERTAINTY_CATEGORIES) - 1)
            
            # Calculate weighted score
            question_score = q["weight"] * severity_factor * (0.5 + 0.5 * certainty_factor)
            total_score += question_score
            
        # Maximum possible score if all responses were maximum severity and certainty
        max_possible_score += q["weight"]
    
    # Normalize to 0-100% scale
    if max_possible_score > 0:
        normalized_score = (total_score / max_possible_score) * 100
    else:
        normalized_score = 0
        
    return normalized_score

# Determine risk level and recommendation
def get_risk_level(score):
    if score >= 70:
        return {
            "level": "High risk",
            "color": "red",
            "recommendation": "Consult a dermatologist immediately. Your responses indicate several warning signs that require urgent professional evaluation."
        }
    elif score >= 40:
        return {
            "level": "Moderate risk",
            "color": "orange",
            "recommendation": "Schedule an appointment with a dermatologist in the next few weeks. There are signs that deserve medical attention."
        }
    elif score >= 20:
        return {
            "level": "Low-moderate risk",
            "color": "yellow",
            "recommendation": "Consider consulting with a doctor during your next regular check-up. Keep observing any changes."
        }
    else:
        return {
            "level": "Low risk",
            "color": "green",
            "recommendation": "Continue monitoring any changes. Perform regular skin self-examinations and protect yourself from the sun."
        }

# User interface with Streamlit
st.title("Melanoma Risk Assessment System")

st.markdown("""
This system uses artificial intelligence to assess the possible risk of melanoma 
based on your answers. Complete the questionnaire with the greatest possible accuracy.

**IMPORTANT**: This tool does NOT replace professional medical diagnosis. 
If you have concerns about your health, consult a medical professional.
""")

# Create tabs for each category
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Evolution", "Appearance", "Symptoms", "Risk Factors", "Results"])

# Initialize the response dictionary in the session state
if 'responses' not in st.session_state:
    st.session_state.responses = {}

# Store all questions for analysis
all_questions = []
for category, category_questions in questions.items():
    all_questions.extend(category_questions)

# Function to reset all responses
def reset_all():
    st.session_state.responses = {}
    
# Create form for each category
with tab1:
    st.header("Evolution")
    for q in questions["Evolution"]:
        response = st.text_area(f"{q['text']}", key=f"input_{q['id']}", height=100)
        if response and (q['id'] not in st.session_state.responses or st.session_state.responses[q['id']]['response'] != response):
            classification = classify_response(response, model, tokenizer)
            st.session_state.responses[q['id']] = {**classification, "response": response}

with tab2:
    st.header("Appearance")
    for q in questions["Appearance"]:
        response = st.text_area(f"{q['text']}", key=f"input_{q['id']}", height=100)
        if response and (q['id'] not in st.session_state.responses or st.session_state.responses[q['id']]['response'] != response):
            classification = classify_response(response, model, tokenizer)
            st.session_state.responses[q['id']] = {**classification, "response": response}

with tab3:
    st.header("Symptoms")
    for q in questions["Symptoms"]:
        response = st.text_area(f"{q['text']}", key=f"input_{q['id']}", height=100)
        if response and (q['id'] not in st.session_state.responses or st.session_state.responses[q['id']]['response'] != response):
            classification = classify_response(response, model, tokenizer)
            st.session_state.responses[q['id']] = {**classification, "response": response}

with tab4:
    st.header("Risk Factors")
    for q in questions["Risk Factors"]:
        response = st.text_area(f"{q['text']}", key=f"input_{q['id']}", height=100)
        if response and (q['id'] not in st.session_state.responses or st.session_state.responses[q['id']]['response'] != response):
            classification = classify_response(response, model, tokenizer)
            st.session_state.responses[q['id']] = {**classification, "response": response}
    
    st.button("Reset all responses", on_click=reset_all)

# Show results
with tab5:
    st.header("Risk Assessment")
    
    if len(st.session_state.responses) == 0:
        st.warning("You haven't entered any responses yet. Please complete the questions in the other tabs.")
    else:
        # Calculate scores by category
        category_scores = {}
        for category, category_questions in questions.items():
            category_score = calculate_weighted_score({k: v for k, v in st.session_state.responses.items() if k in [q["id"] for q in category_questions]}, category_questions)
            category_scores[category] = category_score
        
        # Calculate total score
        total_score = calculate_weighted_score(st.session_state.responses, all_questions)
        risk_assessment = get_risk_level(total_score)
        
        # Show results
        st.subheader("Overall risk score")
        st.markdown(f"""
        <div style="background-color: {risk_assessment['color']}; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h3 style="color: white;">{risk_assessment['level']}: {total_score:.1f}%</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Score by category")
        cols = st.columns(4)
        for i, (category, score) in enumerate(category_scores.items()):
            with cols[i]:
                st.metric(label=category, value=f"{score:.1f}%")
        
        st.subheader("Recommendation")
        st.info(risk_assessment['recommendation'])
        
        # Response details for medical review
        st.subheader("Response details")
        st.write("This information may be useful to share with your doctor")
        
        for category, category_questions in questions.items():
            st.write(f"**{category}**")
            for q in category_questions:
                if q["id"] in st.session_state.responses:
                    response = st.session_state.responses[q["id"]]
                    st.markdown(f"""
                    - **Question**: {q['text']}
                    - **Response**: {response['response']}
                    - **Assessment**: {response['severity']} (Certainty: {response['certainty']})
                    """)
                else:
                    st.write(f"- {q['text']}: No response")
            st.write("---")

# Add footer
st.markdown("""
---
**Disclaimer**: This system is for educational and informational purposes only. 
It is not intended to provide medical diagnosis and does not substitute consultation with a healthcare professional.
""")