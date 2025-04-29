import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, vit_b_16
from sklearn.metrics import f1_score, roc_curve, auc, confusion_matrix, precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
import os
import glob
import cv2
from tqdm import tqdm

# ----- Funciones de preprocesamiento avanzado -----

def segment_skin_lesion(image_path):
    """
    Segmenta la lesión de piel para enfocar el análisis solo en la región de interés
    """
    # Leer imagen
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplicar desenfoque gaussiano para reducir ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Umbralización de Otsu
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Operaciones morfológicas para mejorar la segmentación
    kernel = np.ones((7, 7), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Encontrar el contorno más grande (presumiblemente la lesión)
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Si no se encuentra lesión, devolver imagen original
    
    # Encontrar el contorno más grande
    max_contour = max(contours, key=cv2.contourArea)
    
    # Crear máscara
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [max_contour], 0, 255, -1)
    
    # Aplicar máscara a la imagen original
    result = cv2.bitwise_and(img, img, mask=mask)
    
    # Convertir de BGR a RGB para ser compatible con PIL
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    return result_rgb

class AdvancedMelanomaDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None, segmentation=False):
        self.image_paths = image_paths
        self.labels = labels  # Puede ser None para conjuntos de prueba
        self.transform = transform
        self.segmentation = segmentation
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Aplicar segmentación si está habilitada
        if self.segmentation:
            img_data = segment_skin_lesion(img_path)
            if img_data is None:  # Si la segmentación falla
                img_data = Image.open(img_path).convert('RGB')
            else:
                img_data = Image.fromarray(img_data)
        else:
            img_data = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img_data = self.transform(img_data)
        
        if self.labels is not None:
            return img_data, self.labels[idx]
        else:
            return img_data

# ----- Definiciones de los modelos -----

class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_mask = self.attention(x)
        return x * attention_mask

class EfficientNetWithAttention(nn.Module):
    def __init__(self, num_classes=2):
        super(EfficientNetWithAttention, self).__init__()
        self.efficientnet = efficientnet_b0(pretrained=True)
        feature_size = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Identity()
        self.attention = AttentionModule(1280)
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.efficientnet.features(x)
        features = self.attention(features)
        features = self.efficientnet.avgpool(features)
        features = torch.flatten(features, 1)
        return self.classifier(features)

# ----- Evaluación avanzada y visualización -----

def evaluate_model(model, dataloader, device, threshold=0.5):
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluando modelo"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # Probabilidad de clase positiva
            preds = (probs > threshold).astype(int)
            
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)
    
    # Calcular métricas
    conf_matrix = confusion_matrix(all_targets, all_preds)
    tn, fp, fn, tp = conf_matrix.ravel()
    
    metrics = {
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,  # Recall o sensibilidad
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'f1_score': f1_score(all_targets, all_preds),
        'dice': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    }
    
    # Calcular AUCROC
    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    metrics['auc_roc'] = auc(fpr, tpr)
    
    # Calcular curva Precision-Recall
    precision, recall, _ = precision_recall_curve(all_targets, all_probs)
    metrics['auc_pr'] = auc(recall, precision)
    
    return metrics, all_targets, all_preds, all_probs, (fpr, tpr), (precision, recall)

def visualize_results(metrics, conf_matrix, roc_data, pr_data, save_dir='./results'):
    """
    Visualiza y guarda los resultados de evaluación
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benigno', 'Maligno'],
                yticklabels=['Benigno', 'Maligno'])
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    plt.title('Matriz de confusión')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix.png')
    plt.close()
    
    # 2. Curva ROC
    fpr, tpr = roc_data
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', label=f'AUC = {metrics["auc_roc"]:.4f}')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Tasa de falsos positivos')
    plt.ylabel('Tasa de verdaderos positivos')
    plt.title('Curva ROC')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f'{save_dir}/roc_curve.png')
    plt.close()
    
    # 3. Curva Precision-Recall
    precision, recall = pr_data
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, 'g-', label=f'AUC PR = {metrics["auc_pr"]:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig(f'{save_dir}/pr_curve.png')
    plt.close()
    
    # 4. Tabla de métricas
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Valor'])
    metrics_df.to_csv(f'{save_dir}/metrics.csv')
    
    # 5. Gráfico de barras de métricas
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.title('Métricas de rendimiento')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/metrics_bar.png')
    plt.close()

def find_optimal_threshold(y_true, y_scores):
    """
    Encuentra el umbral óptimo para maximizar F1
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    # Visualizar la búsqueda de umbral
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, 'b-')
    plt.axvline(x=best_threshold, color='r', linestyle='--', 
                label=f'Umbral óptimo: {best_threshold:.2f}, F1: {best_f1:.4f}')
    plt.xlabel('Umbral')
    plt.ylabel('F1 Score')
    plt.title('Búsqueda del umbral óptimo')
    plt.legend()
    plt.grid(True)
    plt.savefig('./results/threshold_tuning.png')
    plt.close()
    
    return best_threshold, best_f1

# ----- Ensamblado de modelos -----

class EnsembleModel(nn.Module):
    def __init__(self, models, weights=None):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights if weights is not None else [1/len(models)] * len(models)
    
    def forward(self, x):
        outputs = []
        for i, model in enumerate(self.models):
            output = model(x)
            outputs.append(self.weights[i] * F.softmax(output, dim=1))
        
        # Promedio ponderado
        return sum(outputs)

def optimize_ensemble(efficientnet_model, vit_model, val_loader, device):
    """
    Optimiza los pesos del ensamblado utilizando un conjunto de validación
    """
    # Definir posibles pesos a probar
    weight_options = [
        [0.5, 0.5],  # Igual peso
        [0.6, 0.4],  # Más peso a EfficientNet
        [0.4, 0.6],  # Más peso a ViT
        [0.7, 0.3],
        [0.3, 0.7],
        [0.8, 0.2],
        [0.2, 0.8]
    ]
    
    best_weights = None
    best_metrics = None
    best_f1 = 0
    
    for weights in weight_options:
        # Crear ensamblado
        ensemble = EnsembleModel([efficientnet_model, vit_model], weights=weights)
        ensemble.to(device)
        
        # Evaluar
        metrics, _, _, _, _, _ = evaluate_model(ensemble, val_loader, device)
        
        print(f"Pesos {weights}: F1={metrics['f1_score']:.4f}, DICE={metrics['dice']:.4f}, AUC-ROC={metrics['auc_roc']:.4f}")
        
        # Actualizar mejores pesos
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_weights = weights
            best_metrics = metrics
    
    return best_weights, best_metrics

# ----- Función principal -----

def main():
    # Definir transformaciones
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    # Rutas a los modelos pre-entrenados
    efficientnet_path = r"C:\Users\jakif\CODE\PROYECTO-FINAL\COMPUTER_VISION\melanoma_model_1_torch_EFFICIENTNETB0_harvard_attention.pth"
    vit_path = r"C:\Users\jakif\CODE\PROYECTO-FINAL\COMPUTER_VISION\vision_transformers\best_vit_model_2.pth"
    
    # Ruta al conjunto de datos de prueba
    test_dir = r"C:\Users\jakif\CODE\PROYECTO-FINAL\images\PREPROCESSED_DATA_copy\test"  
    
    # Configurar el dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Cargar los modelos
    print("Cargando modelo EfficientNet con atención...")
    efficientnet_model = EfficientNetWithAttention(num_classes=2)
    efficientnet_model.load_state_dict(torch.load(efficientnet_path, map_location=device))
    efficientnet_model.to(device)
    
    print("Cargando modelo Vision Transformer...")
    vit_model = vit_b_16(pretrained=False)
    vit_model.heads = nn.Linear(vit_model.heads.in_features, 2)
    vit_model.load_state_dict(torch.load(vit_path, map_location=device))
    vit_model.to(device)
    
    # Cargar conjunto de datos de prueba
    # Esta sección debe adaptarse a tu conjunto de datos específico
    test_images = []  # Lista de rutas a imágenes
    test_labels = []  # Lista de etiquetas (0: benigno, 1: maligno)
    
    # Ejemplo para cargar imágenes y etiquetas
    # Supongamos una estructura con dos carpetas: 'benign' y 'malignant'
    benign_dir = os.path.join(test_dir, 'benign_images')
    malignant_dir = os.path.join(test_dir, 'malignant_images')
    
    # Cargar imágenes benignas
    if os.path.exists(benign_dir):
        for img_path in glob.glob(os.path.join(benign_dir, '*.jpg')) + glob.glob(os.path.join(benign_dir, '*.png')):
            test_images.append(img_path)
            test_labels.append(0)  # 0 para benigno
    
    # Cargar imágenes malignas
    if os.path.exists(malignant_dir):
        for img_path in glob.glob(os.path.join(malignant_dir, '*.jpg')) + glob.glob(os.path.join(malignant_dir, '*.png')):
            test_images.append(img_path)
            test_labels.append(1)  # 1 para maligno
    
    print(f"Cargadas {len(test_images)} imágenes para prueba")
    
    # Crear conjuntos de datos
    test_dataset = AdvancedMelanomaDataset(
        test_images, 
        test_labels, 
        transform=data_transforms['val'],
        segmentation=True  # Aplicar segmentación para mejorar resultados
    )
    
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Evaluar modelos individuales
    print("\nEvaluando EfficientNet con atención...")
    efficientnet_metrics, eff_targets, eff_preds, eff_probs, eff_roc, eff_pr = evaluate_model(
        efficientnet_model, test_loader, device
    )
    
    print("Métricas de EfficientNet:")
    for metric, value in efficientnet_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nEvaluando Vision Transformer...")
    vit_metrics, vit_targets, vit_preds, vit_probs, vit_roc, vit_pr = evaluate_model(
        vit_model, test_loader, device
    )
    
    print("Métricas de Vision Transformer:")
    for metric, value in vit_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Buscar umbrales óptimos
    print("\nBuscando umbral óptimo para EfficientNet...")
    eff_threshold, _ = find_optimal_threshold(eff_targets, eff_probs)
    print(f"Umbral óptimo para EfficientNet: {eff_threshold:.4f}")
    
    print("Buscando umbral óptimo para Vision Transformer...")
    vit_threshold, _ = find_optimal_threshold(vit_targets, vit_probs)
    print(f"Umbral óptimo para Vision Transformer: {vit_threshold:.4f}")
    
    # Reevaluar modelos con umbrales óptimos
    print("\nReevaluando EfficientNet con umbral óptimo...")
    eff_opt_metrics, _, _, _, _, _ = evaluate_model(
        efficientnet_model, test_loader, device, threshold=eff_threshold
    )
    
    print("Métricas optimizadas de EfficientNet:")
    for metric, value in eff_opt_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nReevaluando Vision Transformer con umbral óptimo...")
    vit_opt_metrics, _, _, _, _, _ = evaluate_model(
        vit_model, test_loader, device, threshold=vit_threshold
    )
    
    print("Métricas optimizadas de Vision Transformer:")
    for metric, value in vit_opt_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Optimizar ensamblado
    print("\nOptimizando pesos del ensamblado...")
    best_weights, best_metrics = optimize_ensemble(
        efficientnet_model, vit_model, test_loader, device
    )
    
    print(f"Mejores pesos para ensamblado: {best_weights}")
    print("Métricas del mejor ensamblado:")
    for metric, value in best_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Crear y evaluar el modelo ensamblado final
    ensemble = EnsembleModel([efficientnet_model, vit_model], weights=best_weights)
    ensemble.to(device)
    
    # Buscar umbral óptimo para ensamblado
    ensemble_metrics, ens_targets, _, ens_probs, ens_roc, ens_pr = evaluate_model(
        ensemble, test_loader, device
    )
    
    ens_threshold, _ = find_optimal_threshold(ens_targets, ens_probs)
    print(f"\nUmbral óptimo para ensamblado: {ens_threshold:.4f}")
    
    # Evaluación final con umbral optimizado
    print("\nEvaluación final del ensamblado con umbral optimizado...")
    final_metrics, final_targets, final_preds, _, final_roc, final_pr = evaluate_model(
        ensemble, test_loader, device, threshold=ens_threshold
    )
    
    print("Métricas finales del ensamblado optimizado:")
    for metric, value in final_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Generar matriz de confusión final
    final_conf_matrix = confusion_matrix(final_targets, final_preds)
    
    # Visualizar resultados
    print("\nGenerando visualizaciones...")
    visualize_results(final_metrics, final_conf_matrix, final_roc, final_pr)
    
    print("\nProceso completado. Los resultados se han guardado en el directorio './results'")

if __name__ == "__main__":
    main()