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

# Create training set
def get_training_set():
    return [
        # Positive responses - Growth/Change
        ("yes, the lesion has increased", "positive"),
        ("Sure, it's getting bigger", "positive"),
        ("It's definitely changed in size", "positive"),
        ("I think it's grown", "positive"),
        ("it seems larger now", "positive"),
        ("it has grown a bit", "positive"),
        ("it's expanding", "positive"),
        ("it's changed shape", "positive"),
        ("it used to be round but now it's not", "positive"),
        ("the color has darkened", "positive"),
        ("it's darker than before", "positive"),
        ("the color is changing", "positive"),
        ("yes it's different now", "positive"),
        
        # Positive responses - Appearance
        ("yes, it's asymmetrical", "positive"),
        ("one side is different", "positive"),
        ("it's not even on both sides", "positive"),
        ("the edges are ragged", "positive"),
        ("the border isn't smooth", "positive"),
        ("the edges look jagged", "positive"),
        ("yes, there are multiple colors", "positive"),
        ("it has different shades", "positive"),
        ("there's black and brown in it", "positive"),
        ("it's bigger than a pencil eraser", "positive"),
        ("it's larger than 6mm", "positive"),
        ("yes it's quite big", "positive"),
        ("it looks different from my other moles", "positive"),
        ("this one stands out", "positive"),
        ("it doesn't match my other spots", "positive"),
        
        # Positive responses - Symptoms
        ("yes it itches sometimes", "positive"),
        ("it's itchy", "positive"),
        ("I feel the need to scratch it", "positive"),
        ("yes it bleeds occasionally", "positive"),
        ("it bleeds when I touch it", "positive"),
        ("it's bled a few times", "positive"),
        ("the area is red", "positive"),
        ("there's redness around it", "positive"),
        ("it looks inflamed", "positive"),
        ("yes it hurts", "positive"),
        ("it's tender to touch", "positive"),
        ("I feel pain there", "positive"),
        ("it forms scabs", "positive"),
        ("it crusts over", "positive"),
        ("it doesn't seem to heal properly", "positive"),
        
        # Positive responses - Risk factors
        ("yes it's in a sunny spot", "positive"),
        ("it gets a lot of sun", "positive"),
        ("it's on my face which is always exposed", "positive"),
        ("I've had bad sunburns", "positive"),
        ("I got sunburned badly as a kid", "positive"),
        ("I've had blistering sunburns", "positive"),
        ("yes, my parent had melanoma", "positive"),
        ("there's skin cancer in my family", "positive"),
        ("my sibling had melanoma", "positive"),
        ("I have lots of moles", "positive"),
        ("I have more than 50 moles", "positive"),
        ("my body is covered in moles", "positive"),
        ("I have fair skin", "positive"),
        ("I burn easily in the sun", "positive"),
        ("I'm very pale and never tan", "positive"),
        ("it's possible", "positive"),
        
        # Negative responses
        ("No, it's still the same", "negative"),
        ("No, it hasn't changed in months", "negative"),
        ("Not at all, it's always been the same size", "negative"),
        ("Nothing", "negative"),
        ("no change", "negative"),
        ("same as always", "negative"),
        ("it hasn't grown", "negative"),
        ("it's the same shape as before", "negative"),
        ("no changes in shape", "negative"),
        ("the color is consistent", "negative"),
        ("it's always been this color", "negative"),
        ("no, the color is stable", "negative"),
        
        ("no, it's symmetrical", "negative"),
        ("both sides look the same", "negative"),
        ("it's even all around", "negative"),
        ("the edges are smooth", "negative"),
        ("the border is regular", "negative"),
        ("no, the edges are well-defined", "negative"),
        ("it's just one color", "negative"),
        ("it's uniformly brown", "negative"),
        ("no variation in color", "negative"),
        ("it's smaller than that", "negative"),
        ("no, it's tiny", "negative"),
        ("it's very small", "negative"),
        ("it looks just like my other moles", "negative"),
        ("all my moles look similar", "negative"),
        ("it's like the others", "negative"),
        
        ("no itching", "negative"),
        ("it doesn't itch", "negative"),
        ("I feel no itch", "negative"),
        ("it's never bled", "negative"),
        ("no bleeding at all", "negative"),
        ("doesn't bleed", "negative"),
        ("there's no redness", "negative"),
        ("skin around it looks normal", "negative"),
        ("no inflammation", "negative"),
        ("it doesn't hurt", "negative"),
        ("no pain or tenderness", "negative"),
        ("painless", "negative"),
        ("it heals normally", "negative"),
        ("no crusting", "negative"),
        ("no scabbing", "negative"),
        
        ("it's usually covered", "negative"),
        ("it doesn't get sun exposure", "negative"),
        ("it's in a protected area", "negative"),
        ("I haven't had severe sunburns", "negative"),
        ("no bad sunburns in my history", "negative"),
        ("I usually don't burn", "negative"),
        ("no family history of skin cancer", "negative"),
        ("no melanoma in my family", "negative"),
        ("no one in my family had skin cancer", "negative"),
        ("I only have a few moles", "negative"),
        ("not many moles", "negative"),
        ("just a couple of moles", "negative"),
        ("I have darker skin", "negative"),
        ("my skin tans easily", "negative"),
        ("I rarely burn in the sun", "negative"),
        
        # Unclear responses
        ("I don't know", "I don't understand"),
        ("What does that mean?", "I don't understand"),
        ("I'm not sure", "I don't understand"),
        ("could you clarify?", "I don't understand"),
        ("can't tell", "I don't understand"),
        ("hard to say", "I don't understand"),
        ("maybe", "I don't understand"),
        ("possibly", "I don't understand"),
        ("sometimes", "I don't understand"),
        ("I haven't checked", "I don't understand"),
        ("I haven't noticed", "I don't understand"),
        ("I haven't paid attention to that", "I don't understand"),
        ("not certain", "I don't understand"),
        ("50/50", "I don't understand"),
        ("I need to look more closely", "I don't understand"),
        ("I don't think so", "I don't understand"),
        ("I don't remember", "I don't understand"),
    ]

# Function to extract words
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