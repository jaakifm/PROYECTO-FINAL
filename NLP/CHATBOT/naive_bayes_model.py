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
        ("it hasn't grown at all", "not_concerning"),
        ("it's still the same size", "not_concerning"),
        ("no change in size", "not_concerning"),
        ("same as always", "not_concerning"),
        ("it hasn't changed shape", "not_concerning"),
        ("the shape is constant", "not_concerning"),
        ("the color hasn't changed", "not_concerning"),
        ("it's always been this color", "not_concerning"),
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
    negations = ["no", "not", "never", "doesn't", "don't", "didn't", "hasn't", "haven't", "isn't", "aren't", "wasn't"]
    for i, token in enumerate(tokens):
        if token in negations and i+1 < len(tokens):
            # Mark the next few words as negated
            for j in range(1, min(4, len(tokens)-i)):
                if tokens[i+j] not in stopwords.words("english"):
                    features[f"negated({tokens[i+j]})"] = True
    
    # Add severity indicators
    severity_terms = {
        "high": ["very", "extremely", "significantly", "a lot", "much", "greatly", "heavily", "severely"],
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
        "irregular": ["irregular", "ragged", "uneven", "asymmetric", "asymmetrical", "not round"]
    }
    
    for symptom, terms in symptoms.items():
        for term in terms:
            if term in tokens or term in text:
                features[f"symptom({symptom})"] = True
    
    return features

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

# Initialize model
@st.cache_resource
def initialize_model():
    training_set = get_training_set()
    training_data = []
    for text, label in training_set:
        training_data.append((extract_word(text), label))
    model = NaiveBayesClassifier.train(training_data)
    return model

# Classify answers
def answer_classification(answer, model):
    features = extract_word(answer)
    return model.classify(features)

# Create title and description
st.title("Melanoma Diagnosis Chatbot")
st.markdown("### Answer the questions with natural phrases to help evaluate your skin lesion.")

# Initialize session state variables
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
    st.session_state.answers = []
    st.session_state.classifications = []
    st.session_state.pos = 0
    st.session_state.neg = 0
    st.session_state.finished = False
    st.session_state.answer_submitted = False
    st.session_state.current_answer = ""
    st.session_state.current_classification = ""

model = initialize_model()

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
                classification = answer_classification(answer, model)
                
                # Store current answer and classification in session state
                st.session_state.current_answer = answer
                st.session_state.current_classification = classification
                st.session_state.answer_submitted = True
                
                # Show interpretation
                if classification == "positive":
                    st.success("I understand your answer as: 'Yes'")
                elif classification == "negative":
                    st.info("I understand your answer as: 'No'")
                else:
                    st.warning("I didn't understand your answer clearly. I'll consider it as partially positive.")
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
                
                # Count positive and negative responses
                if st.session_state.current_classification == "positive":
                    st.session_state.pos += 1
                elif st.session_state.current_classification == "negative":
                    st.session_state.neg += 1
                else:
                    st.session_state.pos += 0.5
                
                # Reset for next question
                st.session_state.answer_submitted = False
                st.session_state.current_answer = ""
                st.session_state.current_classification = ""
                
                # Move to next question
                st.session_state.current_question += 1
                
                # Check if all questions have been answered
                if st.session_state.current_question >= len(questions):
                    st.session_state.finished = True
                


# Show summary when all questions have been answered
if st.session_state.finished:
    st.subheader("Summary of your responses:")
    
    for i, (question, answer, classification) in enumerate(zip(questions, st.session_state.answers, st.session_state.classifications)):
        st.markdown(f"**Question {i+1}:** {question}")
        st.markdown(f"**Your answer:** {answer}")
        
        if classification == "positive":
            st.markdown("**Interpretation:** Yes ✓")
        elif classification == "negative":
            st.markdown("**Interpretation:** No ✗")
        else:
            st.markdown("**Interpretation:** Unclear ⚠️")
        
        st.markdown("---")
    
    st.subheader("Results:")
    st.markdown(f"**Positive responses:** {st.session_state.pos}")
    st.markdown(f"**Negative responses:** {st.session_state.neg}")
    
    # Simple assessment
    if st.session_state.pos > 0.75 * len(questions):
        st.error("Based on your responses, I recommend you see a doctor for further evaluation.")
    else:
        st.success("Based on your responses, it seems there is no immediate concern. However, if you have any doubts, please consult a doctor.")
    
    st.markdown("### Thank you for your answers.")
    
    # Add disclaimer
    st.warning("""
    **IMPORTANT:** This application is for educational purposes only and is not a substitute for professional medical advice. 
    Always consult a dermatologist or healthcare provider for any concerns about skin lesions.
    """) 
    
    # Button to restart
    if st.button("Start Over"):
        st.session_state.current_question = 0
        st.session_state.answers = []
        st.session_state.classifications = []
        st.session_state.pos = 0
        st.session_state.neg = 0
        st.session_state.finished = False
        st.session_state.answer_submitted = False
        st.session_state.current_answer = ""
        st.session_state.current_classification = ""
        st.experimental_rerun()

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
    
    ### Additional Symptoms
    
    - Itching or pain
    - Bleeding or crusting
    - Inflammation or redness
    - Changes in sensitivity
    """)
    
    st.markdown("""
    ### Risk Factors
    
    - Excessive sun exposure
    - History of severe sunburns
    - Fair skin, light hair
    - Family history of skin cancer
    - Many moles (more than 50)
    - Weakened immune system
    """)