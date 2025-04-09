import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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
    
    pdf_files = glob.glob(os.path.join(papers_dir, "**/*.pdf"), recursive=True)
    
    for pdf_path in pdf_files:
        paper_text = extract_text_from_pdf(pdf_path)
        if paper_text.strip():
            all_papers.append({
                "text": paper_text,
                "source": os.path.basename(pdf_path)
            })
    
    return pd.DataFrame(all_papers)

def prepare_dataset(papers_df, tokenizer, max_length=1024):
    """Preparar el dataset para entrenamiento."""
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

# ---- 2. FUNCIONES PARA FINE-TUNING CON PEFT ----

def load_model_and_tokenizer(model_name):
    """Cargar el modelo cuantizado y tokenizer con configuración LoRA."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Configurar tokens especiales
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configuración de cuantización 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Cargar modelo cuantizado
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Preparar modelo para entrenamiento con k-bit
    model = prepare_model_for_kbit_training(model)
    
    # Configuración LoRA
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Aplicar LoRA al modelo
    model = get_peft_model(model, peft_config)
    
    # Imprimir parámetros entrenables
    model.print_trainable_parameters()
    
    return model, tokenizer

def fine_tune_model(model, tokenizer, train_dataset, output_dir):
    """Realizar el fine-tuning del modelo con LoRA."""
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        disable_tqdm=False,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    trainer.train()
    
    # Guardar solo los adaptadores LoRA
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return output_dir

# ---- 3. FUNCIONES DE INFERENCIA ----

def generate_response(model, tokenizer, prompt, max_length=1024):
    """Generar respuesta con un modelo dado."""
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=max_length
    ).to(model.device)
    
    generated_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1
    )
    
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# ---- 4. INTERFAZ STREAMLIT ----

def main():
    st.title("Fine-tuning de Modelos Cuantizados con LoRA")
    
    st.sidebar.header("Configuración")
    papers_dir = st.sidebar.text_input(
        "Directorio de Papers",
        value=r"C:\Users\jakif\CODE\PROYECTO-FINAL\dataset_papers"
    )
    
    model_path = st.sidebar.text_input(
        "Modelo Base",
        value="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    )
    
    output_dir = st.sidebar.text_input(
        "Directorio de Salida",
        value=r"C:\Users\jakif\CODE\PROYECTO-FINAL\NLP\Generation_diagnosis\output"
    )
    
    max_length = st.sidebar.slider("Longitud máxima", 512, 2048, 1024)
    
    tab1, tab2, tab3 = st.tabs(["Cargar Datos", "Fine-tuning", "Inferencia"])
    
    with tab1:
        st.header("Cargar Papers")
        if st.button("Cargar Papers"):
            if os.path.exists(papers_dir):
                with st.spinner("Procesando PDFs..."):
                    papers_df = load_papers_from_directory(papers_dir)
                    st.session_state.papers_df = papers_df
                st.success(f"{len(papers_df)} papers cargados")
                st.dataframe(papers_df.head())
            else:
                st.error("Directorio no encontrado")
    
    with tab2:
        st.header("Fine-tuning con LoRA")
        if st.button("Iniciar Entrenamiento"):
            if not hasattr(st.session_state, "papers_df"):
                st.error("Primero carga los datos")
            else:
                try:
                    with st.spinner("Cargando modelo..."):
                        model, tokenizer = load_model_and_tokenizer(model_path)
                    
                    with st.spinner("Preparando dataset..."):
                        train_dataset = prepare_dataset(st.session_state.papers_df, tokenizer, max_length)
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    os.makedirs(output_dir, exist_ok=True)
                    status_text.text("Entrenando...")
                    
                    fine_tune_model(model, tokenizer, train_dataset, output_dir)
                    
                    progress_bar.progress(100)
                    status_text.text("¡Entrenamiento completado!")
                    st.session_state.trained_model_path = output_dir
                    st.success(f"Modelo guardado en {output_dir}")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with tab3:
        st.header("Probar Modelo")
        prompt = st.text_area("Prompt:", height=150)
        
        if st.button("Generar Respuesta"):
            if hasattr(st.session_state, "trained_model_path"):
                try:
                    with st.spinner("Cargando modelo..."):
                        model, tokenizer = load_model_and_tokenizer(st.session_state.trained_model_path)
                    
                    with st.spinner("Generando..."):
                        response = generate_response(model, tokenizer, prompt)
                        st.text_area("Respuesta:", response, height=300)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.error("Primero entrena el modelo")

if __name__ == "__main__":
    main()