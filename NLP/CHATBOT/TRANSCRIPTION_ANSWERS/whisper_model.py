import streamlit as st
import tempfile
import os
import time
import pandas as pd
import numpy as np
from audio_recorder_streamlit import audio_recorder

# Set page configuration
st.set_page_config(page_title="Skin Lesion Assessment", layout="wide")

# Initialize Whisper model - we'll load it only when needed to avoid conflicts with Streamlit
# This helps prevent the PyTorch runtime errors with Streamlit
def transcribe_with_whisper(audio_path):
    """Load whisper model and transcribe audio file"""
    # Import whisper here to avoid early initialization issues
    import whisper
    
    # Load model when function is called
    model = whisper.load_model("base")  # Options: "tiny", "base", "small", "medium", "large"
    
    # Transcribe
    result = model.transcribe(audio_path)
    return result["text"].strip()

# Define questionnaire
questionnaire = {
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

# App title
st.title("Skin Lesion Assessment Questionnaire")
st.markdown("Please answer each question by recording your voice. Speak clearly after clicking the 'Record' button.")

# Initialize session state to track progress
if 'current_category' not in st.session_state:
    st.session_state.current_category = list(questionnaire.keys())[0]
    
if 'current_question_index' not in st.session_state:
    st.session_state.current_question_index = 0
    
if 'responses' not in st.session_state:
    st.session_state.responses = {}
    
if 'audio_files' not in st.session_state:
    st.session_state.audio_files = {}

def transcribe_audio(audio_bytes):
    """Transcribe audio using Whisper model"""
    # Save audio bytes to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio_path = temp_audio.name
    
    # Transcribe with Whisper
    try:
        # Call our function that loads whisper only when needed
        transcription = transcribe_with_whisper(temp_audio_path)
    except Exception as e:
        st.error(f"Transcription error: {e}")
        transcription = "Error in transcription"
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            
    return transcription

# Function to progress to next question
def next_question():
    categories = list(questionnaire.keys())
    current_cat_index = categories.index(st.session_state.current_category)
    current_cat_questions = questionnaire[st.session_state.current_category]
    
    # Move to next question in current category
    if st.session_state.current_question_index < len(current_cat_questions) - 1:
        st.session_state.current_question_index += 1
    # Move to next category
    elif current_cat_index < len(categories) - 1:
        st.session_state.current_category = categories[current_cat_index + 1]
        st.session_state.current_question_index = 0
    # Completed all questions
    else:
        st.session_state.current_category = "Completed"

# Display current question
if st.session_state.current_category != "Completed":
    # Display progress
    categories = list(questionnaire.keys())
    total_questions = sum(len(questions) for questions in questionnaire.values())
    completed_questions = sum(1 for cat in categories[:categories.index(st.session_state.current_category)] 
                             for _ in questionnaire[cat])
    completed_questions += st.session_state.current_question_index
    
    progress = completed_questions / total_questions
    st.progress(progress)
    
    # Display category and question
    st.header(st.session_state.current_category)
    current_question = questionnaire[st.session_state.current_category][st.session_state.current_question_index]
    st.subheader(f"Question {completed_questions + 1}/{total_questions}:")
    st.write(current_question)
    
    # Create a unique key for the current question
    question_key = f"{st.session_state.current_category}_{st.session_state.current_question_index}"
    
    # Audio recording
    col1, col2 = st.columns([3, 1])
    
    with col1:
        audio_bytes = audio_recorder(
            text="",
            recording_color="#e8b62c",
            neutral_color="#6aa36f",
            icon_name="microphone",
            icon_size="2x"
        )
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            
            # Store audio for later reference
            st.session_state.audio_files[question_key] = audio_bytes
            
            # Transcribe audio
            with st.spinner("Transcribing your response..."):
                transcription = transcribe_audio(audio_bytes)
            
            # Display and store transcription
            st.info(f"Transcribed: {transcription}")
            st.session_state.responses[question_key] = transcription
    
    with col2:
        if audio_bytes or question_key in st.session_state.responses:
            if st.button("Next Question", key=f"next_{question_key}"):
                next_question()
                
else:
    # Show summary of responses
    st.header("Assessment Complete")
    st.success("Thank you for completing the skin lesion assessment questionnaire.")
    
    # Create DataFrame from responses
    data = []
    
    for category in questionnaire:
        for i, question in enumerate(questionnaire[category]):
            key = f"{category}_{i}"
            response = st.session_state.responses.get(key, "No response recorded")
            data.append({
                "Category": category,
                "Question": question,
                "Response": response
            })
    
    response_df = pd.DataFrame(data)
    
    # Display responses
    st.subheader("Your Responses:")
    for category in questionnaire:
        st.write(f"**{category}**")
        category_responses = response_df[response_df["Category"] == category]
        for _, row in category_responses.iterrows():
            st.write(f"- {row['Question']}")
            st.write(f"  *{row['Response']}*")
        st.write("")
    
    # Option to download responses as CSV
    csv = response_df.to_csv(index=False)
    st.download_button(
        label="Download Responses as CSV",
        data=csv,
        file_name="skin_lesion_assessment.csv",
        mime="text/csv"
    )
    
    # Reset questionnaire button
    if st.button("Start New Assessment"):
        for key in ['current_category', 'current_question_index', 'responses', 'audio_files']:
            if key in st.session_state:
                del st.session_state[key]
        