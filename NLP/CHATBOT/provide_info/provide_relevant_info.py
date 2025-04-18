import streamlit as st
from llama_cpp import Llama
import os

MODEL_PATH = r"C:\Users\jakif\.lmstudio\models\QuantFactory\Bio-Medical-Llama-3-8B-GGUF\Bio-Medical-Llama-3-8B.Q4_K_S.gguf" # Actualiza esto con la ruta real a tu modelo

# Función para cargar el modelo
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found {MODEL_PATH}. Please verify it")
        st.stop()
    
    model = Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,           # Tamaño del contexto
        n_gpu_layers=-1,      # Utilizar tantas capas en GPU como sea posible (-1)
        n_batch=512,          # Tamaño del batch para la inferencia
        verbose=False         # Silenciar logs
    )
    return model

def generate_response(prompt, model):
    # Context dermatlog specialist
    system_prompt = """
    
You act as a dermatologist specializing in melanomas. Your goal is to provide accurate and educational medical information about melanomas, including:

- Identifying warning signs
- Risk factors
- Prevention methods
- Diagnosis and treatment options
- When to seek medical attention

Remember that you are not diagnosing specific cases and should always recommend consulting a physician for individual cases. Base your answers on up-to-date scientific evidence.
    """
    # Formato para el modelo LLaMA
    full_prompt = f"""<|system|>
{system_prompt}
<|user|>
{prompt}
<|assistant|>"""
    
    # Generar respuesta
    response = model.create_completion(
        full_prompt,
        max_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        stop=["<|user|>", "<|system|>"],
        echo=False
    )
    
    # Extraer texto de la respuesta
    return response['choices'][0]['text'].strip()


# UI principal
def main():
    # Título y descripción
    st.title("🩺 DermaBot - Asistence for information about melanomas")
    st.markdown("""
    This chatbot uses a biomedical AI model to provide educational information about melanomas.
**Important**: This bot does not provide medical diagnoses. Always consult a healthcare professional for specific cases.""")
    
    # Sidebar con información adicional
    with st.sidebar:
        st.header("About Dermabot")
        st.markdown("""
        This chatbot is powered by the Bio-Medical-Llama-3-8B-GGUF model from QuantFactory
        and is designed to provide educational information about melanomas.

        ### Helpful Resources:
        - [Asociación Americana de Dermatología](https://www.aad.org/)
        - [Skin Cancer Foundation](https://www.skincancer.org/)
        - [Mayo Clinic - Melanoma](https://www.mayoclinic.org/diseases-conditions/melanoma/symptoms-causes/syc-20374884)
        """)
        
        st.warning("⚠️ Remember: The information provided is not a substitute for professional medical advice.")
    

    # Cargar el modelo
    try:
        with st.spinner("charging the model..."):
            model = load_model()
        st.success("Model loaded! You can now ask questions about melanomas.")
    except Exception as e:
        st.error(f"Error charging the model: {e}")
        st.stop()
    
    # Inicializar historial de chat si no existe
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi, I'm DermaBot, a virtual assistant specializing in melanoma information. How can I help you today?"}
        ]
    
    # Mostrar mensajes del chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input del usuario
    if prompt := st.chat_input("Write your question about melanomas..."):
        # Agregar mensaje del usuario al historial
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generar respuesta
        with st.chat_message("assistant"):
            with st.spinner("thinking..."):
                response = generate_response(prompt, model)
                st.markdown(response)
        
        # Agregar respuesta al historial
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()