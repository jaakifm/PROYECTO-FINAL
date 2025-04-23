import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import LlamaCpp
import pandas as pd

# Configuración de la página
st.set_page_config(page_title="RAG - Artículos Científicos", layout="wide")
st.title("Sistema RAG para Artículos Científicos")

# Inicialización de variables de sesión
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'documents_df' not in st.session_state:
    st.session_state.documents_df = pd.DataFrame(columns=['Nombre', 'Tipo', 'Ruta'])
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()

# Sidebar para configuración y carga de archivos
with st.sidebar:
    st.header("Configuración")
    
    # Configuración del modelo local
    st.subheader("Modelo LLM")
    model_path = st.text_input(
        "Ruta del modelo GGUF", 
        value="C:\\Users\\jakif\\.lmstudio\\models\\lmstudio-community\\DeepSeek-R1-Distill-Llama-8B-GGUF\\DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
    )
    
    # Parámetros del modelo
    n_ctx = st.slider("Contexto máximo (n_ctx)", min_value=512, max_value=8192, value=4096, step=512)
    n_gpu_layers = st.slider("Capas en GPU", min_value=0, max_value=100, value=0)
    temperature = st.slider("Temperatura", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
    # Selección de modelo de embeddings
    st.subheader("Embeddings")
    embedding_model = st.selectbox(
        "Modelo de Embeddings",
        ["all-MiniLM-L6-v2", "multi-qa-mpnet-base-dot-v1"]
    )
    
    # Carga de archivos
    st.header("Carga de Artículos")
    uploaded_files = st.file_uploader("Cargar artículos científicos", 
                                     type=["pdf", "txt"], 
                                     accept_multiple_files=True)
    
    if uploaded_files and st.button("Procesar Archivos"):
        with st.spinner("Procesando archivos..."):
            documents = []
            file_details = []
            
            for file in uploaded_files:
                # Guardar el archivo en un directorio temporal
                file_path = os.path.join(st.session_state.temp_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                
                # Cargar documentos según el tipo de archivo
                try:
                    if file.name.endswith('.pdf'):
                        loader = PyPDFLoader(file_path)
                        file_type = "PDF"
                    else:
                        loader = TextLoader(file_path)
                        file_type = "Texto"
                    
                    file_docs = loader.load()
                    
                    # Agregar metadatos más detallados
                    for doc in file_docs:
                        doc.metadata["source"] = file.name
                        doc.metadata["file_type"] = file_type
                        doc.metadata["path"] = file_path
                        # Para PDFs, agregar número de página si está disponible
                        if "page" in doc.metadata:
                            doc.metadata["reference"] = f"{file.name} (p. {doc.metadata['page']})"
                        else:
                            doc.metadata["reference"] = file.name
                    
                    documents.extend(file_docs)
                    file_details.append({
                        'Nombre': file.name,
                        'Tipo': file_type,
                        'Ruta': file_path
                    })
                except Exception as e:
                    st.error(f"Error al procesar {file.name}: {str(e)}")
            
            # Actualizar DataFrame de documentos
            new_df = pd.DataFrame(file_details)
            st.session_state.documents_df = pd.concat([st.session_state.documents_df, new_df], ignore_index=True)
            
            # Dividir documentos en chunks con metadatos mejorados
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)
            
            # Mejorar metadatos de los chunks para rastreo de fuentes
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_id"] = i
                if "page" in chunk.metadata:
                    chunk.metadata["source_detail"] = f"{chunk.metadata['source']} (Pág. {chunk.metadata['page']})"
                else:
                    chunk.metadata["source_detail"] = chunk.metadata['source']
            
            # Crear embeddings y vectorstore
            embeddings = HuggingFaceEmbeddings(model_name=f"sentence-transformers/{embedding_model}")
            st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
            
            st.success(f"Procesados {len(documents)} documentos y creados {len(chunks)} chunks.")

# Mostrar los documentos cargados en una tabla
if not st.session_state.documents_df.empty:
    st.header("Artículos Cargados")
    st.dataframe(st.session_state.documents_df[['Nombre', 'Tipo']], use_container_width=True)

# Área de consulta
st.header("Consulta de Información")

    # Verificar si hay vectorstore disponible
if st.session_state.vectorstore is None:
    st.info("Por favor, carga y procesa algunos artículos científicos para comenzar.")
else:
    query = st.text_area("Ingresa tu consulta sobre los artículos:", height=100)
    
    num_results = st.slider("Número de resultados", min_value=1, max_value=10, value=3)
    
    if st.button("Buscar") and query:
        with st.spinner("Buscando información relevante..."):
            try:
                # Verificar que el modelo existe
                if not os.path.exists(model_path):
                    st.error(f"El archivo del modelo no existe en la ruta: {model_path}")
                else:
                    # Crear modelo LLM local
                    llm = LlamaCpp(
                        model_path=model_path,
                        n_ctx=n_ctx,
                        n_gpu_layers=n_gpu_layers,
                        temperature=temperature,
                        verbose=True,
                        streaming=True
                    )
                    
                    # Crear cadena de retrieval con un prompt personalizado que incluya referencias
                    from langchain.prompts import PromptTemplate
                    
                    # Prompt personalizado que instruye al modelo a proporcionar referencias claras
                    template = """
                    Usa los siguientes fragmentos de artículos científicos para responder a la consulta del usuario.
                    
                    INSTRUCCIONES IMPORTANTES:
                    1. Para cada dato o afirmación en tu respuesta, DEBES indicar de qué artículo obtuviste la información usando el formato [Artículo: nombre_del_archivo].
                    2. Si distintos artículos proporcionan información sobre el mismo tema, sintetiza la información e indica TODAS las fuentes.
                    3. Si encuentras información contradictoria, presenta las diferentes perspectivas indicando claramente la fuente de cada una.
                    4. Si no puedes responder algún aspecto de la consulta con la información proporcionada, indícalo explícitamente.
                    5. Organiza tu respuesta por temas o conceptos clave, no por artículo.
                    6. Prioriza la precisión sobre la exhaustividad.
                    7. NO inventes información ni cites artículos que no estén en los fragmentos proporcionados.
                    
                    Fragmentos de artículos:
                    {context}
                    
                    Consulta: {question}
                    
                    Respuesta (con referencias claras a los artículos):
                    """
                    
                    QA_PROMPT = PromptTemplate(
                        template=template,
                        input_variables=["context", "question"]
                    )
                    
                    # Crear cadena de retrieval con el prompt personalizado
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": num_results}),
                        chain_type_kwargs={"prompt": QA_PROMPT}
                    )
                    
                    # Obtener respuesta
                    with st.spinner("Generando respuesta con el modelo local..."):
                        response = qa_chain.invoke(query)
                    
                    # Mostrar resultado con formato mejorado
                    st.subheader("Respuesta:")
                    result = response["result"]
                    
                    # Resaltar las referencias en el texto
                    import re
                    
                    # Buscar patrones como [Artículo: nombre_del_archivo]
                    highlighted_result = re.sub(
                        r'\[Artículo: ([^\]]+)\]',
                        r'<span style="background-color: #e6f7ff; padding: 2px 4px; border-radius: 3px; font-weight: bold;">[Artículo: \1]</span>',
                        result
                    )
                    
                    st.markdown(highlighted_result, unsafe_allow_html=True)
                    
                    # Sección de resumen de fuentes
                    st.subheader("Resumen de Fuentes Utilizadas:")
                    
                    # Extraer todas las referencias únicas
                    references = set(re.findall(r'\[Artículo: ([^\]]+)\]', result))
                    
                    if references:
                        for ref in references:
                            # Encontrar la ruta del archivo original
                            file_info = st.session_state.documents_df[
                                st.session_state.documents_df['Nombre'] == ref
                            ]
                            
                            if not file_info.empty:
                                file_path = file_info.iloc[0]['Ruta']
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.markdown(f"**{ref}**")
                                with col2:
                                    st.download_button(
                                        label="Descargar",
                                        data=open(file_path, "rb"),
                                        file_name=ref,
                                        mime="application/octet-stream"
                                    )
                    else:
                        st.info("No se detectaron referencias específicas en la respuesta.")
                    
                    # Mostrar los documentos originales
                    st.subheader("Fuentes Relevantes:")
                    docs = st.session_state.vectorstore.similarity_search(query, k=num_results)
                    
                    for i, doc in enumerate(docs):
                        with st.expander(f"Documento {i+1}: {doc.metadata['source']}"):
                            st.markdown("**Extracto:**")
                            st.write(doc.page_content)
                            st.markdown("**Fuente:** " + doc.metadata['source'])
                            
                            # Encontrar la ruta del archivo original
                            file_info = st.session_state.documents_df[
                                st.session_state.documents_df['Nombre'] == doc.metadata['source']
                            ]
                            
                            if not file_info.empty:
                                file_path = file_info.iloc[0]['Ruta']
                                st.download_button(
                                    label="Descargar artículo completo",
                                    data=open(file_path, "rb"),
                                    file_name=doc.metadata['source'],
                                    mime="application/octet-stream"
                                )
            except Exception as e:
                st.error(f"Error al procesar la consulta: {str(e)}")
                st.error("Detalles del error para depuración:")
                st.code(str(e))