import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Configure the page
st.set_page_config(
    page_title="Melanoma Diagnosis Chatbot based on NAIVE BAYES",
)

# Create enhanced training set for severity classification
def get_severity_training_set():
    return [
        # Highly Concerning responses
        ("it's grown a lot", "highly_concerning"),
        ("it's changed a lot", "highly_concerning"),
        ("always", "highly_concerning"),
        ("it's growing rapidly", "highly_concerning"),
        ("it's significantly larger now", "highly_concerning"),
        ("it's expanding quickly", "highly_concerning"),
        ("it's growing fast", "highly_concerning"),
        ("it's much bigger than before", "highly_concerning"),
        ("it's changed shape dramatically", "highly_concerning"),
        ("the color has changed significantly", "highly_concerning"),
        ("it's turned very dark", "highly_concerning"),
        ("it now has multiple very different colors", "highly_concerning"),
        ("it bleeds frequently", "highly_concerning"),
        ("it bleeds a lot", "highly_concerning"),
        ("it's always painful", "highly_concerning"),
        ("it hurts all the time", "highly_concerning"),
        ("it's extremely itchy", "highly_concerning"),
        ("it looks very different from all my other moles", "highly_concerning"),
        
        # Moderately Concerning responses
        ("it has grown somewhat", "moderately_concerning"),
        ("it's a bit bigger now", "moderately_concerning"),
        ("it's changed shape a little", "moderately_concerning"),
        ("the color has darkened", "moderately_concerning"),
        ("it has a couple different colors", "moderately_concerning"),
        ("the border has become more irregular", "moderately_concerning"),
        ("it bleeds occasionally", "moderately_concerning"),
        ("it bleeds when I touch it", "moderately_concerning"),
        ("it's sometimes itchy", "moderately_concerning"),
        ("it itches now and then", "moderately_concerning"),
        ("it can be a bit tender", "moderately_concerning"),
        ("it's somewhat painful sometimes", "moderately_concerning"),
        ("it doesn't look quite like my other moles", "moderately_concerning"),
        
        # Mildly Concerning responses
        ("I think it might be slightly larger", "mildly_concerning"),
        ("it could have grown a tiny bit", "mildly_concerning"),
        ("perhaps the shape has changed slightly", "mildly_concerning"),
        ("the color might be a bit different", "mildly_concerning"),
        ("it's itched once or twice", "mildly_concerning"),
        ("it bled one time", "mildly_concerning"),
        ("it was tender once", "mildly_concerning"),
        ("it's a little different from my other moles", "mildly_concerning"),
        ("it might be a little asymmetrical", "mildly_concerning"),
        
        # Not Concerning responses
        ("no"," not_concerning"),
        ("nothing", "not_concerning"),
        ("it hasn't grown at all", "not_concerning"),
        ("it's still the same size", "not_concerning"),
        ("no change in size", "not_concerning"),
        ("same as always", "not_concerning"),
        ("it hasn't changed shape", "not_concerning"),
        ("the shape is constant", "not_concerning"),
        ("the color hasn't changed", "not_concerning"),
        ("it never bleeds", "not_concerning"),
        ("no bleeding at all", "not_concerning"),
        ("it doesn't itch", "not_concerning"),
        ("no itching", "not_concerning"),
        ("it's not painful", "not_concerning"),
        ("no pain or tenderness", "not_concerning"),
        ("it looks like all my other moles", "not_concerning"),
        ("it's symmetrical", "not_concerning"),
        ("the edges are smooth", "not_concerning"),
        ("it's a single color", "not_concerning"),
    ]

# Create training set for certainty classification
def get_certainty_training_set():
    return [
        # Certain responses
        ("definitely yes", "certain"),
        ("absolutely", "certain"),
        ("certainly", "certain"),
        ("100 percent", "certain"),
        ("without a doubt", "certain"),
        ("I'm sure", "certain"),
        ("I'm certain", "certain"),
        ("definitely not", "certain"),
        ("absolutely not", "certain"),
        ("certainly not", "certain"),
        
        # Probable responses
        ("probably", "probable"),
        ("most likely", "probable"),
        ("I think so", "probable"),
        ("it seems like it", "probable"),
        ("I believe so", "probable"),
        ("it appears to be", "probable"),
        ("likely yes", "probable"),
        ("I don't think so", "probable"),
        ("probably not", "probable"),
        
        # Possible responses
        ("possibly", "possible"),
        ("maybe", "possible"),
        ("perhaps", "possible"),
        ("it could be", "possible"),
        ("it might be", "possible"),
        ("somewhat", "possible"),
        ("it's possible", "possible"),
        
        # Unlikely responses
        ("unlikely", "unlikely"),
        ("I doubt it", "unlikely"),
        ("not really", "unlikely"),
        ("I don't believe so", "unlikely"),
        ("probably not", "unlikely"),
        
        # Unknown responses
        ("I don't know", "unknown"),
        ("I'm not sure", "unknown"),
        ("I haven't checked", "unknown"),
        ("I haven't noticed", "unknown"),
        ("hard to say", "unknown"),
        ("I can't tell", "unknown"),
        ("not certain", "unknown"),
        ("I don't remember", "unknown"),
        ("I haven't paid attention", "unknown"),
    ]

def extract_advanced_features(text):
    text = text.lower()
    tokens = word_tokenize(text)
    
    features = {}
    
    # Basic word presence features
    for token in tokens:
        if token not in stopwords.words("english"):
            features[f"contains({token})"] = True
    
    # Add bigram features
    bigrams = list(nltk.bigrams(tokens))
    for bigram in bigrams:
        features[f"bigram({bigram[0]}_{bigram[1]})"] = True
    
    # Check for negation patterns
    negations = [ "no","not", "never", "doesn't", "don't", "didn't", "hasn't", "haven't", "isn't", "aren't", "wasn't"]
    for i, token in enumerate(tokens):
        if token in negations and i+1 < len(tokens):
            # Mark the next few words as negated
            for j in range(1, min(4, len(tokens)-i)):
                if tokens[i+j] not in stopwords.words("english"):
                    features[f"negated({tokens[i+j]})"] = True
    
    # Add severity indicators
    severity_terms = {
        "high": ["very", "extremely", "significantly", "a lot", "much", "greatly", "heavily", "severely", "yes"],
        "moderate": ["somewhat", "moderately", "fairly", "quite", "rather"],
        "mild": ["slightly", "a bit", "a little", "mildly", "somewhat"]
    }
    
    for level, terms in severity_terms.items():
        for term in terms:
            if term in text or any(term == t for t in tokens):
                features[f"severity({level})"] = True
    
    # Add certainty indicators
    certainty_terms = {
        "high": ["definitely", "certainly", "absolutely", "sure", "clearly", "undoubtedly", "without a doubt"],
        "moderate": ["probably", "likely", "most likely", "think", "believe", "seems"],
        "low": ["maybe", "perhaps", "possibly", "might", "could", "may"]
    }
    
    for level, terms in certainty_terms.items():
        for term in terms:
            if term in text or any(term == t for t in tokens):
                features[f"certainty({level})"] = True
    
    # Add frequency indicators
    frequency_terms = {
        "high": ["always", "constantly", "frequently", "regularly", "often", "every day"],
        "moderate": ["sometimes", "occasionally", "now and then", "periodically"],
        "low": ["rarely", "seldom", "once", "once or twice", "almost never"]
    }
    
    for level, terms in frequency_terms.items():
        for term in terms:
            if term in text or any(term == t for t in tokens):
                features[f"frequency({level})"] = True
    
    # Check for size-related terms
    size_increase = ["bigger", "larger", "grown", "expanded", "increased", "growing"]
    size_stable = ["same", "unchanged", "stable", "constant", "consistent"]
    size_small = ["small", "tiny", "little"]
    
    for term in size_increase:
        if term in tokens:
            features["size_increase"] = True
    for term in size_stable:
        if term in tokens:
            features["size_stable"] = True
    for term in size_small:
        if term in tokens:
            features["size_small"] = True
    
    # Check for color-related terms
    color_change = ["darkened", "darker", "changed color", "different color"]
    color_multiple = ["colors", "multi-colored", "different shades", "varied"]
    
    for term in color_change:
        if term in text:
            features["color_change"] = True
    for term in color_multiple:
        if term in text:
            features["color_multiple"] = True
    
    # Check for symptom terms
    symptoms = {
        "bleeding": ["bleed", "bleeding", "bled", "blood"],
        "itching": ["itch", "itches", "itchy", "itching"],
        "pain": ["pain", "painful", "hurts", "hurt", "tender", "sore"],
        "scab": ["scab", "crust", "non-healing", "not healing"],
        "redness": ["red", "redness", "swollen", "swelling"],
        "swelling": ["swollen", "swelling"],
        "irregular": ["irregular", "ragged", "uneven", "asymmetric", "asymmetrical", "not round"]
    }
    
    for symptom, terms in symptoms.items():
        for term in terms:
            if term in tokens or term in text:
                features[f"symptom({symptom})"] = True
    
    return features
def extract_word(text):
    tokenized_text = word_tokenize(text.lower())  
    return {word: True for word in tokenized_text}
# Define questions
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

question_weights = {
    0: 1.0,  # Growth in size (very important)
    1: 0.9,  # Change in shape (very important)
    2: 0.9,  # Change in color (very important)
    3: 0.7,  # Size > 6mm (important)
    4: 0.8,  # Different from other moles (important)
    5: 0.6,  # Itchiness (moderately important)
    6: 0.8,  # Bleeding (important)
    7: 0.7,  # Redness/swelling (important)
    8: 0.6,  # Pain/tenderness (moderately important)
    9: 0.8,  # Non-healing scab (important)
    10: 0.5, # Sun exposure (less important)
    11: 0.6, # History of sunburns (less important) 
    12: 0.7, # Family history (important)
    13: 0.3, # Many moles (less important)
    14: 0.3, # Fair skin (less important)
}

# Initialize models
@st.cache_resource
def initialize_models():
    # Train severity model
    severity_training_set = get_severity_training_set()
    severity_training_data = [(extract_advanced_features(text), label) for text, label in severity_training_set]
    severity_model = NaiveBayesClassifier.train(severity_training_data)
    
    # Train certainty model
    certainty_training_set = get_certainty_training_set()
    certainty_training_data = [(extract_advanced_features(text), label) for text, label in certainty_training_set]
    certainty_model = NaiveBayesClassifier.train(certainty_training_data)
    
    return severity_model, certainty_model

# Classify answers
def classify_answer(answer, severity_model, certainty_model):
    features = extract_advanced_features(answer)
    severity = severity_model.classify(features)
    certainty = certainty_model.classify(features)
    return severity, certainty

# Calculate risk score based on responses
def calculate_risk_score(responses, weights):
    severity_values = {
        "highly_concerning": 1.0,
        "moderately_concerning": 0.7,
        "mildly_concerning": 0.3,
        "not_concerning": 0.0
    }
    
    certainty_multipliers = {
        "certain": 1.0,
        "probable": 0.8,
        "possible": 0.6,
        "unlikely": 0.3,
        "unknown": 0.5
    }
    
    total_score = 0
    max_possible = 0
    
    for question_idx, (severity, certainty) in enumerate(responses):
        question_weight = weights[question_idx]
        max_possible += question_weight
        
        response_score = (severity_values[severity] * 
                         certainty_multipliers[certainty] * 
                         question_weight)
        
        total_score += response_score
    
    # Return normalized score (0-100%)
    return (total_score / max_possible) * 100 if max_possible > 0 else 0

# Define severity descriptions
severity_descriptions = {
    "highly_concerning": "highly concerning",
    "moderately_concerning": "moderately concerning",
    "mildly_concerning": "mildly concerning",
    "not_concerning": "not concerning"
}

# Define certainty descriptions
certainty_descriptions = {
    "certain": "with high certainty",
    "probable": "with moderate certainty",
    "possible": "with some uncertainty",
    "unlikely": "with low likelihood",
    "unknown": "with unclear certainty"
}

# Define severity colors
severity_colors = {
    "highly_concerning": "🔴",
    "moderately_concerning": "🟠",
    "mildly_concerning": "🟡",
    "not_concerning": "🟢"
}

# Create title and description
st.title("Advanced Melanoma Diagnosis Chatbot")
st.markdown("### Answer the questions with natural phrases to help evaluate your skin lesion.")
st.markdown("This system uses a granular classification approach to better understand your responses.")

# Initialize session state variables
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
    st.session_state.answers = []
    st.session_state.classifications = []  # Now stores (severity, certainty) tuples
    st.session_state.finished = False
    st.session_state.answer_submitted = False
    st.session_state.current_answer = ""
    st.session_state.current_classification = ()  # Now a tuple

severity_model, certainty_model = initialize_models()

# Display current question and process the response
if not st.session_state.finished and st.session_state.current_question < len(questions):
    st.subheader(f"Question {st.session_state.current_question + 1}/{len(questions)}:")
    st.markdown(f"**{questions[st.session_state.current_question]}**")
    
    # Create input field for answer
    answer = st.text_input("Your answer:", key=f"q_{st.session_state.current_question}")
    
    # Create columns for buttons
    col1, col2 = st.columns(2)
    
    # Submit button
    with col1:
        if st.button("Submit Answer"):
            if answer:
                # Get both severity and certainty classifications
                severity, certainty = classify_answer(answer, severity_model, certainty_model)
                
                # Store current answer and classification in session state
                st.session_state.current_answer = answer
                st.session_state.current_classification = (severity, certainty)
                st.session_state.answer_submitted = True
                
                # Show interpretation with severity and certainty
                severity_text = severity_descriptions[severity]
                certainty_text = certainty_descriptions[certainty]
                severity_icon = severity_colors[severity]
                
                st.markdown(f"""
                **Interpretation:** {severity_icon} I understand this as **{severity_text}** {certainty_text}.
                """)
                
                # Show detailed explanation based on classification
                if severity == "highly_concerning":
                    st.error("This response suggests a significant symptom that should be evaluated by a healthcare professional.")
                elif severity == "moderately_concerning":
                    st.warning("This response indicates a moderate level of concern that merits attention.")
                elif severity == "mildly_concerning":
                    st.info("This response shows a mild level of concern. Continue to monitor for changes.")
                else:  # not_concerning
                    st.success("This response does not indicate a concerning symptom for this question.")
            else:
                st.error("Please provide an answer before submitting.")
    
    # Next question button
    with col2:
        next_disabled = not st.session_state.answer_submitted
        if st.button("Next Question", disabled=next_disabled):
            if st.session_state.answer_submitted:
                # Save answer and classification
                st.session_state.answers.append(st.session_state.current_answer)
                st.session_state.classifications.append(st.session_state.current_classification)
                
                # Reset for next question
                st.session_state.answer_submitted = False
                st.session_state.current_answer = ""
                st.session_state.current_classification = ()
                
                # Move to next question
                st.session_state.current_question += 1
                
                # Check if all questions have been answered
                if st.session_state.current_question >= len(questions):
                    st.session_state.finished = True
                
                

# Show summary when all questions have been answered
if st.session_state.finished:
    st.subheader("Summary of your responses:")
    
    # Create columns for organizing the summary
    col1, col2 = st.columns([2, 1])
    
    with col1:
        for i, (question, answer, classification) in enumerate(zip(questions, st.session_state.answers, st.session_state.classifications)):
            severity, certainty = classification
            severity_text = severity_descriptions[severity]
            certainty_text = certainty_descriptions[certainty]
            severity_icon = severity_colors[severity]
            
            st.markdown(f"**Question {i+1}:** {question}")
            st.markdown(f"**Your answer:** {answer}")
            st.markdown(f"**Interpretation:** {severity_icon} {severity_text.capitalize()} {certainty_text}")
            st.markdown("---")
    
    with col2:
        # Calculate risk score
        risk_score = calculate_risk_score(st.session_state.classifications, question_weights)
        
        st.markdown("### Risk Assessment")
        
        # Display risk meter
        st.markdown(f"**Overall Risk Score:** {risk_score:.1f}%")
        
        # Create a visual risk meter
        risk_color = "green"
        risk_message = "Low Risk"
        
        if risk_score >= 70:
            risk_color = "red"
            risk_message = "High Risk"
        elif risk_score >= 40:
            risk_color = "orange"
            risk_message = "Moderate Risk"
        elif risk_score >= 20:
            risk_color = "yellow"
            risk_message = "Low-Moderate Risk"
        
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(to right, green, yellow, orange, red);
                height: 20px;
                border-radius: 10px;
                position: relative;
                margin: 10px 0;
            ">
                <div style="
                    position: absolute;
                    left: {risk_score}%;
                    top: -15px;
                    transform: translateX(-50%);
                    font-size: 24px;
                ">▼</div>
            </div>
            <div style="
                color: {risk_color};
                font-weight: bold;
                font-size: 18px;
                text-align: center;
            ">{risk_message}</div>
            """,
            unsafe_allow_html=True
        )
        
        # Display symptom categories
        st.markdown("### Symptom Categories")
        
        # Group questions by category
        categories = {
            "Evolution (Change)": [0, 1, 2],
            "Appearance": [3, 4],
            "Symptoms": [5, 6, 7, 8, 9],
            "Risk Factors": [10, 11, 12, 13, 14]
        }
        
        for category, question_indices in categories.items():
            category_responses = [st.session_state.classifications[i] for i in question_indices if i < len(st.session_state.classifications)]
            if category_responses:
                # Count concerning responses
                concerning_count = sum(1 for severity, _ in category_responses if severity in ["highly_concerning", "moderately_concerning"])
                total_count = len(category_responses)
                
                if concerning_count == 0:
                    status = "✅"
                elif concerning_count < total_count / 2:
                    status = "⚠️"
                else:
                    status = "❌"
                
                st.markdown(f"{status} **{category}:** {concerning_count}/{total_count} concerning")
    
    # Recommendation based on risk score
    st.subheader("Recommendation:")
    
    if risk_score >= 70:
        st.error("""
        **Based on your responses, it is strongly recommended that you consult with a dermatologist or healthcare provider as soon as possible.**
        
        Several concerning features have been identified that warrant professional evaluation.
        """)
    elif risk_score >= 40:
        st.warning("""
        **Based on your responses, it is recommended that you schedule an appointment with a healthcare provider for evaluation.**
        
        Some concerning features have been identified that should be professionally assessed.
        """)
    elif risk_score >= 20:
        st.info("""
        **Based on your responses, consider having this lesion checked during your next regular healthcare visit.**
        
        While there are some mildly concerning features, immediate action may not be necessary unless you notice changes.
        """)
    else:
        st.success("""
        **Based on your responses, there appear to be no immediate concerning features.**
        
        Continue to monitor the lesion for any changes. If you notice changes in size, shape, color, or if it begins to
        itch, bleed, or cause discomfort, please seek medical advice.
        """)
    
    st.markdown("### Thank you for your answers.")
    
    # Add disclaimer
    st.warning("""
    **IMPORTANT:** This application is for educational purposes only and is not a substitute for professional medical advice. 
    The granular classification system provides more detailed analysis but is still experimental.
    Always consult a dermatologist or healthcare provider for any concerns about skin lesions.
    """)
    
    # Button to restart
    if st.button("Start Over"):
        st.session_state.current_question = 0
        st.session_state.answers = []
        st.session_state.classifications = []
        st.session_state.finished = False
        st.session_state.answer_submitted = False
        st.session_state.current_answer = ""
        st.session_state.current_classification = ()
        

# Add information in the sidebar
with st.sidebar:
    st.header("Information about Melanoma")
    st.markdown("""
    ### The ABCDE Rule

    To detect potential melanomas, dermatologists use the ABCDE rule:

    - **A**symmetry: One half doesn't match the other
    - **B**order: Irregular, ragged, notched, or blurred edges
    - **C**olor: Variety of colors or uneven distribution
    - **D**iameter: Larger than 6mm (pencil eraser)
    - **E**volving: Changing in size, shape, or color
    """)
    
    st.markdown("### About Multi-Class Classification")
    st.markdown("""
    This enhanced version uses a multi-class classification system that evaluates:
    
    **Severity Levels:**
    - 🔴 **Highly Concerning**: Significant symptoms that warrant medical attention
    - 🟠 **Moderately Concerning**: Notable symptoms that suggest further evaluation
    - 🟡 **Mildly Concerning**: Minor symptoms that should be monitored
    - 🟢 **Not Concerning**: No significant symptoms for this question
    
    **Certainty Levels:**
    - **High Certainty**: Clear, definitive responses
    - **Moderate Certainty**: Probable but not definitive responses
    - **Some Uncertainty**: Possible but unclear responses
    - **Low Likelihood**: Responses suggesting the symptom is unlikely
    - **Unclear Certainty**: Responses where certainty cannot be determined
    
    This approach provides a more nuanced assessment than simple yes/no answers.
    """)