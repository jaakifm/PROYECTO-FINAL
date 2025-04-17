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

# Configuración de la página
st.set_page_config(
    page_title="Clasificador de Melanoma",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Función para cargar los datos con mejor manejo de errores
@st.cache_data
def load_data():
    try:
        # Verificar si el archivo existe
        if not os.path.exists('dataset_answers.json'):
            # Si no existe, crear datos de muestra
            st.warning("Archivo 'dataset_answers.json' no encontrado. Creando datos de muestra.")
            
            # Datos de muestra
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
            
            # Guardar datos de muestra
            with open('melanoma_data.json', 'w') as f:
                json.dump(sample_data, f, indent=2)
            
            return sample_data
        
        # Leer el archivo
        with open('dataset_answers.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    except json.JSONDecodeError:
        st.error("Error al decodificar el archivo JSON. El archivo puede estar corrupto.")
        # Crear estructura mínima para evitar errores
        return {
            "name": "Error en datos",
            "description": "No se pudieron cargar los datos correctamente",
            "data": []
        }
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        st.error(traceback.format_exc())
        # Crear estructura mínima para evitar errores
        return {
            "name": "Error en datos",
            "description": "No se pudieron cargar los datos correctamente",
            "data": []
        }

# Función para obtener las etiquetas y sus IDs
@st.cache_data
def get_labels(data):
    try:
        if not data or "data" not in data or not data["data"]:
            # Si no hay datos, usar etiquetas por defecto
            default_labels = ["not_concerning", "mildly_concerning", "moderately_concerning", "highly_concerning"]
            label_to_id = {label: idx for idx, label in enumerate(default_labels)}
            id_to_label = {idx: label for idx, label in enumerate(default_labels)}
            return default_labels, label_to_id, id_to_label
        
        labels = [item["label"] for item in data["data"]]
        unique_labels = sorted(set(labels))
        
        # Si no hay etiquetas, usar etiquetas por defecto
        if not unique_labels:
            default_labels = ["not_concerning", "mildly_concerning", "moderately_concerning", "highly_concerning"]
            label_to_id = {label: idx for idx, label in enumerate(default_labels)}
            id_to_label = {idx: label for idx, label in enumerate(default_labels)}
            return default_labels, label_to_id, id_to_label
        
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        id_to_label = {idx: label for idx, label in enumerate(unique_labels)}
        return unique_labels, label_to_id, id_to_label
    except Exception as e:
        st.error(f"Error al procesar etiquetas: {str(e)}")
        # Etiquetas por defecto en caso de error
        default_labels = ["not_concerning", "mildly_concerning", "moderately_concerning", "highly_concerning"]
        label_to_id = {label: idx for idx, label in enumerate(default_labels)}
        id_to_label = {idx: label for idx, label in enumerate(default_labels)}
        return default_labels, label_to_id, id_to_label

# Función para cargar los modelos y el tokenizador
@st.cache_resource
def load_models(label_to_id, id_to_label):
    try:
        # Cargar tokenizador
        tokenizer = AutoTokenizer.from_pretrained("rjac/biobert-ICD10-L3-mimic")
        
        # Configuración correcta de etiquetas
        num_labels = len(label_to_id)
        id2label = {str(i): label for i, label in id_to_label.items()}
        label2id = {label: str(i) for label, i in label_to_id.items()}
        
        # Cargar modelo preentrenado
        pretrained_model = AutoModelForSequenceClassification.from_pretrained(
            "rjac/biobert-ICD10-L3-mimic", 
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True  # Ignorar tamaños de capas no coincidentes
        )
        
        # Cargar modelo fine-tuneado (si existe)
        finetuned_model = None
        if os.path.exists("./finetuned_model"):
            try:
                # Intentar cargar el modelo con la misma configuración de etiquetas
                finetuned_model = AutoModelForSequenceClassification.from_pretrained(
                    "./finetuned_model",
                    num_labels=num_labels,
                    id2label=id2label,
                    label2id=label2id
                )
            except Exception as e:
                st.warning(f"No se pudo cargar el modelo fine-tuneado: {str(e)}")
                finetuned_model = None
        
        return tokenizer, pretrained_model, finetuned_model
    except Exception as e:
        st.error(f"Error al cargar los modelos: {str(e)}")
        st.error(traceback.format_exc())
        return None, None, None

# Función para clasificar una respuesta
def classify_response(response, model, tokenizer, id_to_label):
    try:
        # Poner modelo en modo evaluación
        model.eval()
        
        # Tokenizar la entrada
        inputs = tokenizer(response, return_tensors="pt", padding=True, truncation=True)
        
        # Realizar predicción
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            score = probs[0, pred_class].item()
        
        # Obtener la etiqueta
        label_name = id_to_label[pred_class]
        
        return label_name, score
    except Exception as e:
        st.error(f"Error en la clasificación: {str(e)}")
        st.error(traceback.format_exc())
        return "error", 0.0

# Función para visualizar los resultados
def visualize_results(pretrained_label, pretrained_score, finetuned_label=None, finetuned_score=None):
    # Definir el mapeo de colores
    color_map = {
        'not_concerning': '#4CAF50',  # Verde
        'mildly_concerning': '#FFEB3B',  # Amarillo
        'moderately_concerning': '#FF9800',  # Naranja
        'highly_concerning': '#F44336',   # Rojo
        'error': '#9E9E9E'  # Gris para errores
    }
    
    # Crear datos para visualización
    models = ['Preentrenado']
    scores = [pretrained_score]
    labels = [pretrained_label]
    colors = [color_map.get(pretrained_label, '#9E9E9E')]
    
    if finetuned_label is not None and finetuned_score is not None:
        models.append('Fine-tuneado')
        scores.append(finetuned_score)
        labels.append(finetuned_label)
        colors.append(color_map.get(finetuned_label, '#9E9E9E'))
    
    # Crear el dataframe
    chart_data = pd.DataFrame({
        'Modelo': models,
        'Confianza': scores,
        'Clasificación': labels
    })
    
    # Crear un gráfico de barras
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Versión actualizada de barplot
    bars = sns.barplot(x='Modelo', y='Confianza', hue='Modelo', data=chart_data, ax=ax, palette=colors, legend=False)
    
    # Añadir etiquetas con la clasificación
    for i, bar in enumerate(bars.patches):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            labels[i].replace('_', ' ').title(),
            ha='center',
            va='bottom',
            fontsize=12
        )
    
    plt.title('Comparación de confianza y clasificación entre modelos', fontsize=14)
    plt.ylim(0, 1.1)
    plt.ylabel('Confianza')
    plt.xlabel('Modelo')
    
    # Mostrar el gráfico
    st.pyplot(fig)

# Lista de preguntas
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

# Función principal
def main():
    # Título y descripción
    st.title("Clasificador de Severidad de Melanoma")
    st.write("""
    Esta aplicación utiliza modelos de transformers para clasificar respuestas sobre lesiones de piel
    y evaluar el posible riesgo de melanoma. Compara un modelo preentrenado (BioBERT) con un modelo
    fine-tuneado específicamente para este caso de uso.
    """)
    
    # Cargar datos
    data = load_data()
    
    if not data or "data" not in data or not data["data"]:
        st.error("No hay datos disponibles para el análisis.")
        return
    
    unique_labels, label_to_id, id_to_label = get_labels(data)
    
    # Sidebar con información del dataset
    st.sidebar.title("Información del Dataset")
    st.sidebar.write(f"Nombre del dataset: {data.get('name', 'Desconocido')}")
    st.sidebar.write(f"Total de ejemplos: {len(data.get('data', []))}")
    
    # Mostrar distribución de etiquetas
    label_counts = {}
    for item in data.get("data", []):
        if "label" in item:
            if item["label"] not in label_counts:
                label_counts[item["label"]] = 0
            label_counts[item["label"]] += 1
    
    if label_counts:
        st.sidebar.subheader("Distribución de etiquetas")
        
        # Crear un DataFrame para la distribución
        dist_data = pd.DataFrame({
            'Etiqueta': list(label_counts.keys()),
            'Cantidad': list(label_counts.values())
        })
        
        # Ordenar las etiquetas por nivel de preocupación
        order = ['not_concerning', 'mildly_concerning', 'moderately_concerning', 'highly_concerning']
        if all(label in order for label in dist_data['Etiqueta']):
            dist_data['Etiqueta'] = pd.Categorical(dist_data['Etiqueta'], categories=order, ordered=True)
            dist_data = dist_data.sort_values('Etiqueta')
        
        # Definir colores para las etiquetas
        colors = {
            'not_concerning': '#4CAF50',  # Verde
            'mildly_concerning': '#FFEB3B',  # Amarillo
            'moderately_concerning': '#FF9800',  # Naranja
            'highly_concerning': '#F44336'   # Rojo
        }
        
        bar_colors = [colors.get(label, '#9E9E9E') for label in dist_data['Etiqueta']]
        
        # Crear gráfico de barras
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Versión actualizada del barplot
        bars = sns.barplot(x='Etiqueta', y='Cantidad', hue='Etiqueta', data=dist_data, palette=bar_colors, ax=ax, legend=False)
        
        # Establecer ticks antes de cambiar las etiquetas
        plt.xticks(range(len(dist_data)))
        ax.set_xticklabels([label.replace('_', ' ').title() for label in dist_data['Etiqueta']])
        
        # Rotar etiquetas
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        st.sidebar.pyplot(fig)
    
    # Cargar modelos
    tokenizer, pretrained_model, finetuned_model = load_models(label_to_id, id_to_label)
    
    if tokenizer and pretrained_model:
        # Crear pestañas para diferentes secciones
        tab1, tab2, tab3 = st.tabs(["Clasificación Individual", "Comparación de Modelos", "Ejemplos del Dataset"])
        
        with tab1:
            st.header("Clasificación de Respuestas")
            
            # Seleccionar una pregunta
            selected_question = st.selectbox("Selecciona una pregunta:", questions)
            
            # Entrada de respuesta
            user_response = st.text_input("Tu respuesta:", "")
            
            if st.button("Clasificar respuesta") and user_response:
                # Clasificar con el modelo preentrenado
                pretrained_label, pretrained_score = classify_response(
                    user_response, pretrained_model, tokenizer, id_to_label
                )
                
                # Mostrar resultados
                st.subheader("Resultados de la clasificación")
                
                # Modelo preentrenado
                st.write("**Modelo preentrenado (BioBERT):**")
                st.write(f"Clasificación: {pretrained_label.replace('_', ' ').title()}")
                st.write(f"Confianza: {pretrained_score:.4f}")
                
                # Si existe el modelo fine-tuneado
                if finetuned_model:
                    # Clasificar con el modelo fine-tuneado
                    finetuned_label, finetuned_score = classify_response(
                        user_response, finetuned_model, tokenizer, id_to_label
                    )
                    
                    # Modelo fine-tuneado
                    st.write("**Modelo fine-tuneado:**")
                    st.write(f"Clasificación: {finetuned_label.replace('_', ' ').title()}")
                    st.write(f"Confianza: {finetuned_score:.4f}")
                    
                    # Visualizar resultados
                    visualize_results(pretrained_label, pretrained_score, finetuned_label, finetuned_score)
                else:
                    # Solo visualizar el modelo preentrenado
                    visualize_results(pretrained_label, pretrained_score)
                
                # Explicación de la clasificación
                st.subheader("Interpretación de resultados")
                
                concern_explanations = {
                    'not_concerning': """
                    **No preocupante**: No hay signos de alarma detectados en esta respuesta. 
                    Sin embargo, es importante mantener un seguimiento regular de cualquier cambio en la lesión.
                    """,
                    'mildly_concerning': """
                    **Levemente preocupante**: Se detectan algunos signos leves que merecen seguimiento. 
                    Se recomienda observar la lesión periódicamente y consultar a un dermatólogo 
                    si se observan cambios adicionales.
                    """,
                    'moderately_concerning': """
                    **Moderadamente preocupante**: Se detectan signos que requieren evaluación médica. 
                    Se recomienda programar una consulta con un dermatólogo en las próximas semanas
                    para una evaluación profesional.
                    """,
                    'highly_concerning': """
                    **Altamente preocupante**: Se detectan signos serios que requieren atención médica inmediata. 
                    Se recomienda consultar a un dermatólogo lo antes posible para una evaluación completa
                    y posible biopsia.
                    """
                }
                
                # Mostrar la explicación del modelo preentrenado
                st.markdown(concern_explanations.get(pretrained_label, ""))
                
                # Mostrar advertencia
                st.warning("""
                **Nota importante**: Esta clasificación es solo una herramienta de ayuda y no sustituye 
                el diagnóstico médico profesional. Siempre consulte a un dermatólogo para una evaluación adecuada.
                """)
        
        with tab2:
            st.header("Comparación de Modelos")
            
            # Cargar métricas de comparación si existen
            comparison_file = 'model_comparison_results.json'
            if os.path.exists(comparison_file):
                try:
                    with open(comparison_file, 'r') as f:
                        metrics_data = json.load(f)
                    
                    # Métricas comparativas
                    if finetuned_model:
                        metrics = {
                            'Exactitud': [metrics_data['pretrained']['accuracy'], metrics_data['finetuned']['accuracy']],
                            'Precisión': [metrics_data['pretrained']['precision'], metrics_data['finetuned']['precision']],
                            'Recall': [metrics_data['pretrained']['recall'], metrics_data['finetuned']['recall']],
                            'F1-Score': [metrics_data['pretrained']['f1'], metrics_data['finetuned']['f1']]
                        }
                        
                        metrics_df = pd.DataFrame(metrics, index=['Modelo preentrenado', 'Modelo fine-tuneado'])
                        
                        # Mostrar tabla de métricas
                        st.subheader("Métricas de rendimiento")
                        st.table(metrics_df)
                        
                        # Visualización de métricas
                        st.subheader("Visualización de métricas:")
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        metrics_df.T.plot(kind='bar', ax=ax)
                        plt.title('Comparación de métricas entre modelos')
                        plt.ylim(0, 1)
                        plt.ylabel('Puntuación')
                        plt.xlabel('Métrica')
                        plt.legend(title='Modelo')
                        
                        st.pyplot(fig)
                        
                        # Explicación de mejoras
                        st.subheader("Análisis de mejoras")
                        
                        # Calcular porcentajes de mejora
                        accuracy_improvement = (metrics_data['finetuned']['accuracy'] - metrics_data['pretrained']['accuracy']) * 100
                        precision_improvement = (metrics_data['finetuned']['precision'] - metrics_data['pretrained']['precision']) * 100
                        recall_improvement = (metrics_data['finetuned']['recall'] - metrics_data['pretrained']['recall']) * 100
                        f1_improvement = (metrics_data['finetuned']['f1'] - metrics_data['pretrained']['f1']) * 100
                        
                        st.write(f"""
                        El modelo fine-tuneado muestra mejoras en todas las métricas evaluadas:
                        
                        - **Exactitud**: Incremento del {accuracy_improvement:.1f}% en la clasificación correcta de todas las respuestas.
                        - **Precisión**: Mejora del {precision_improvement:.1f}% en la identificación precisa de casos positivos.
                        - **Recall**: Aumento del {recall_improvement:.1f}% en la capacidad de identificar todos los casos positivos.
                        - **F1-Score**: Mejora del {f1_improvement:.1f}% en la media armónica entre precisión y recall.
                        
                        Estas mejoras demuestran el valor del fine-tuning específico para el dominio de melanoma
                        comparado con el modelo base preentrenado BioBERT.
                        """)
                    else:
                        st.info("El modelo fine-tuneado no está disponible para comparación.")
                except Exception as e:
                    st.error(f"Error al cargar las métricas de comparación: {str(e)}")
                    st.error(traceback.format_exc())
            else:
                st.info("No hay datos comparativos disponibles. Ejecuta primero el entrenamiento y evaluación de modelos.")
        
        with tab3:
            st.header("Ejemplos del Dataset")
            
            # Verificar si hay datos para mostrar
            if data and "data" in data and data["data"]:
                # Mostrar ejemplos para cada categoría
                st.subheader("Ejemplos por categoría")
                
                # Crear un diccionario de ejemplos por etiqueta
                examples_by_label = {}
                for item in data["data"]:
                    if "label" in item and "text" in item:
                        if item["label"] not in examples_by_label:
                            examples_by_label[item["label"]] = []
                        examples_by_label[item["label"]].append(item["text"])
                
                # Mostrar ejemplos por categoría
                for label in unique_labels:
                    # Obtener ejemplos para esta etiqueta
                    examples = examples_by_label.get(label, [])
                    # Mostrar hasta 5 ejemplos
                    example_list = examples[:5]
                    
                    # Crear expander para cada categoría
                    with st.expander(f"{label.replace('_', ' ').title()} ({len(examples)} ejemplos)"):
                        if example_list:
                            for i, example in enumerate(example_list):
                                st.write(f"{i+1}. \"{example}\"")
                        else:
                            st.write("No hay ejemplos disponibles para esta categoría.")
            else:
                st.warning("No hay ejemplos disponibles en el dataset.")
    else:
        st.error("No se pudieron cargar los modelos. Por favor, verifica la instalación.")

if __name__ == "__main__":
    main()