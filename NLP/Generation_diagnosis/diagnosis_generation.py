import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import pandas as pd
import glob
import PyPDF2
import streamlit as st

# ---- 1. FUNCIONES PARA CARGAR Y PROCESAR PAPERS ACADÉMICOS ----

def extract_text_from_pdf(pdf_path):
    """Extraer texto de un archivo PDF."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error al procesar {pdf_path}: {e}")
    return text

def load_papers_from_directory(papers_dir):
    """Cargar todos los papers PDF de un directorio."""
    all_papers = []
    
    # Buscar todos los PDFs en el directorio
    pdf_files = glob.glob(os.path.join(papers_dir, "**/*.pdf"), recursive=True)
    
    for pdf_path in pdf_files:
        paper_text = extract_text_from_pdf(pdf_path)
        if paper_text.strip():  # Solo agregar si se extrajo texto
            all_papers.append({
                "text": paper_text,
                "source": os.path.basename(pdf_path)
            })
    
    return pd.DataFrame(all_papers)

def prepare_dataset(papers_df, tokenizer, max_length=1024):
    """Preparar el dataset para entrenamiento."""
    
    # Convertir a formato HuggingFace Dataset
    dataset = Dataset.from_pandas(papers_df)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "source"]
    )
    
    return tokenized_dataset

# ---- 2. FUNCIONES PARA FINE-TUNING ----

def load_model_and_tokenizer(model_name):
    """Cargar el modelo y tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Asegurar que el tokenizer tenga los tokens especiales para instrucciones
    special_tokens = {
        "pad_token": "[PAD]",
        "eos_token": "</s>",
        "bos_token": "<s>"
    }
    tokenizer.add_special_tokens(special_tokens)
    
    # Cargar el modelo con soporte para LoRA para optimizar el fine-tuning
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,  # Cuantización para reducir uso de memoria
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Resize el embeddings del modelo para acomodar los nuevos tokens especiales
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer

def fine_tune_model(model, tokenizer, train_dataset, output_dir):
    """Realizar el fine-tuning del modelo."""
    
    # Configuración del entrenamiento
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=True,  # Usar precisión mixta para acelerar y reducir memoria
        report_to="none",  # Evitar dependencias externas
        disable_tqdm=False
    )
    
    # Preparar el Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # No usar masked language modeling para LLaMA
    )
    
    # Configurar y ejecutar el entrenamiento
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Entrenar el modelo
    trainer.train()
    
    # Guardar el modelo entrenado
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    return output_dir

# ---- 3. APLICACION STREAMLIT ----

def main():

    st.title("Fine-tuning de LLaMA 3.1 con Papers Académicos")
    
    # Sidebar para configuración
    st.sidebar.header("Configuración")
    
    # Selección del directorio de papers
    papers_dir = st.sidebar.text_input(
        "Directorio de Papers (ruta local)",
        value=r"C:\Users\jakif\CODE\PROYECTO-FINAL\dataset_papers"
    )
    
    # Modelo a utilizar
    model_path = st.sidebar.text_input(
        "Ruta del modelo LLaMA descargado",
        value="deepseek-ai/DeepSeek-V3-0324"
    )
    
    # Directorio de salida para el modelo fine-tuned
    output_dir = st.sidebar.text_input(
        "Directorio para guardar el modelo",
        value=r"C:\Users\jakif\CODE\PROYECTO-FINAL\NLP\Generation_diagnosis\output"
    )
    
    # Parámetros adicionales
    max_length = st.sidebar.slider("Longitud máxima de secuencia", 512, 2048, 1024)
    
    # Pestañas para diferentes funcionalidades
    tab1, tab2, tab3 = st.tabs(["Cargar Papers", "Fine-tuning", "Inferencia"])
    
    # 1. Pestaña de carga de papers
    with tab1:
        st.header("Cargar y Preparar Papers")
        
        if st.button("Cargar y Mostrar Papers"):
            if os.path.exists(papers_dir):
                with st.spinner("Cargando papers..."):
                    papers_df = load_papers_from_directory(papers_dir)
                    st.session_state.papers_df = papers_df
                    
                st.success(f"Se cargaron {len(papers_df)} papers")
                st.dataframe(papers_df[["source"]].head())
                
                # Mostrar una muestra del texto extraído
                if not papers_df.empty:
                    st.subheader("Muestra de texto extraído")
                    sample_text = papers_df.iloc[0]["text"]
                    st.text_area("Texto del primer paper", sample_text[:1000] + "...", height=200)
            else:
                st.error(f"El directorio {papers_dir} no existe")
    
    # 2. Pestaña de fine-tuning
    with tab2:
        st.header("Fine-tuning del Modelo")
        
        if st.button("Iniciar Fine-tuning"):
            if not hasattr(st.session_state, "papers_df") or st.session_state.papers_df.empty:
                st.error("Primero debes cargar los papers en la pestaña anterior")
            else:
                try:
                    with st.spinner("Cargando modelo y tokenizer..."):
                        model, tokenizer = load_model_and_tokenizer(model_path)
                        st.success("Modelo y tokenizer cargados correctamente")
                    
                    with st.spinner("Preparando dataset..."):
                        train_dataset = prepare_dataset(st.session_state.papers_df, tokenizer, max_length)
                        st.success(f"Dataset preparado con {len(train_dataset)} ejemplos")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Crear el directorio de salida si no existe
                    os.makedirs(output_dir, exist_ok=True)
                    
                    status_text.text("Iniciando fine-tuning...")
                    
                    # Aquí simularemos las actualizaciones del entrenamiento
                    # En una implementación real, estas actualizaciones vendrían del callback de Trainer
                    fine_tune_model(model, tokenizer, train_dataset, output_dir)
                    
                    progress_bar.progress(100)
                    status_text.text("¡Fine-tuning completado!")
                    
                    st.session_state.trained_model_path = output_dir
                    st.success(f"Modelo guardado en {output_dir}")
                    
                except Exception as e:
                    st.error(f"Error durante el fine-tuning: {str(e)}")
    
    # 3. Pestaña de inferencia
    with tab3:
        st.header("Inferencia con el Modelo Fine-tuned")
        
        model_path_for_inference = st.text_input(
            "Ruta del modelo fine-tuned",
            value=st.session_state.get("trained_model_path", output_dir)
        )
        
        user_input = st.text_area("Introduce tu consulta:", height=150)
        
        if st.button("Generar Respuesta"):
            if os.path.exists(model_path_for_inference):
                try:
                    with st.spinner("Cargando modelo para inferencia..."):
                        inference_model, inference_tokenizer = load_model_and_tokenizer(model_path_for_inference)
                    
                    with st.spinner("Generando respuesta..."):
                        inputs = inference_tokenizer(
                            user_input, 
                            return_tensors="pt", 
                            padding=True, 
                            truncation=True,
                            max_length=max_length
                        ).to(inference_model.device)
                        
                        generated_ids = inference_model.generate(
                            inputs["input_ids"],
                            max_length=1024,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            num_return_sequences=1
                        )
                        
                        response = inference_tokenizer.decode(
                            generated_ids[0], 
                            skip_special_tokens=True
                        )
                        
                        st.text_area("Respuesta:", response, height=300)
                        
                except Exception as e:
                    st.error(f"Error durante la inferencia: {str(e)}")
            else:
                st.error(f"El modelo no se encuentra en {model_path_for_inference}")

if __name__ == "__main__":
    main()