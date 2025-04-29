import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, vit_b_16
from sklearn.metrics import f1_score, roc_curve, auc, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image


import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# Llamar al inicio del script
set_seed()
# ----- Definición de los modelos con la estructura correcta -----

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

# Modelo EfficientNet exactamente como está guardado
class OriginalEfficientNetWithAttention(nn.Module):
    def __init__(self, num_classes=2):
        super(OriginalEfficientNetWithAttention, self).__init__()
        self.base_model = efficientnet_b0(pretrained=True)
        feature_size = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Identity()
        
        # Módulo de atención
        self.attention = AttentionModule(1280)  # EfficientNetB0 tiene 1280 canales en la última capa
        
        # Clasificador
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.base_model.features(x)
        features = self.attention(features)
        features = self.base_model.avgpool(features)
        features = torch.flatten(features, 1)
        return self.classifier(features)

# ----- Funciones de carga de modelo -----

def load_efficientnet_model(model_path, device):
    model = OriginalEfficientNetWithAttention(num_classes=2)
    
    try:
        # Cargar state_dict y moverlo al dispositivo
        state_dict = torch.load(model_path, map_location=device)
        
        # Intentar cargar directamente
        model.load_state_dict(state_dict, strict=False)
        print("Modelo EfficientNet cargado correctamente (strict=False)")
    except Exception as e:
        print(f"Error al cargar el modelo EfficientNet: {str(e)}")
        print("El modelo se utilizará con los pesos de ImageNet iniciales")
    
    return model

def load_vit_model(model_path, device):
    # Cargar modelo base
    model = vit_b_16(pretrained=True)  # Usar pesos preentrenados como respaldo
    
    # Determinar estructura de heads y adaptarla para clasificación binaria
    if hasattr(model, 'heads'):
        if isinstance(model.heads, nn.Sequential):
            # Si es sequential, reemplazar la última capa
            last_layer_idx = len(model.heads) - 1
            if isinstance(model.heads[last_layer_idx], nn.Linear):
                in_features = model.heads[last_layer_idx].in_features
                model.heads[last_layer_idx] = nn.Linear(in_features, 2)
            else:
                # Si no podemos determinar los features, usar un valor estándar
                model.heads = nn.Linear(768, 2)  # ViT-B/16 tiene 768 features
        else:
            # Si es directamente un Linear
            in_features = model.heads.in_features
            model.heads = nn.Linear(in_features, 2)
    else:
        # Estructura alternativa: algunos modelos usan head en singular
        if hasattr(model, 'head'):
            if isinstance(model.head, nn.Linear):
                in_features = model.head.in_features
                model.head = nn.Linear(in_features, 2)
            else:
                model.head = nn.Linear(768, 2)
    
    try:
        # Intentar cargar el modelo
        checkpoint = torch.load(model_path, map_location=device)
        
        # Comprobar si es un state_dict o un modelo completo
        if hasattr(checkpoint, 'state_dict'):
            checkpoint = checkpoint.state_dict()
        
        # Cargar con strict=False
        model.load_state_dict(checkpoint, strict=False)
        print("Modelo ViT cargado correctamente")
    except Exception as e:
        print(f"Error al cargar modelo ViT: {str(e)}")
        print("El modelo se utilizará con los pesos de ImageNet iniciales")
    
    return model
def analizar_correlacion_errores(model1_preds, model2_preds, true_labels):
    # Errores del modelo 1
    errores_model1 = (model1_preds != true_labels)
    # Errores del modelo 2
    errores_model2 = (model2_preds != true_labels)
    
    # Calcular correlación de errores
    coincidencia_errores = np.mean(errores_model1 & errores_model2)
    print(f"% de ejemplos donde ambos modelos fallan: {coincidencia_errores*100:.2f}%")
    
    # Calcular proporción de errores por modelo
    error_rate_model1 = np.mean(errores_model1)
    error_rate_model2 = np.mean(errores_model2)
    print(f"Tasa de error EfficientNet: {error_rate_model1*100:.2f}%")
    print(f"Tasa de error ViT: {error_rate_model2*100:.2f}%")
# ----- Clase para ensamblaje de modelos -----

class EnsembleModel:
    def __init__(self, models, weights=None, temperatures=None):
        self.models = models
        self.weights = weights if weights is not None else [1/len(models)] * len(models)
        self.temperatures = temperatures if temperatures is not None else [1.0] * len(models)


    
    def eval(self):
        """Pone el modelo en modo evaluación"""
        self.training = False
        # También poner los modelos individuales en modo evaluación
        for model in self.models:
            model.eval()
        return self
    
    def train(self, mode=True):
        """Pone el modelo en modo entrenamiento"""
        self.training = mode
        for model in self.models:
            model.train(mode)
        return self
    
    # Y modifica predict para aplicar temperatura
    def predict(self, x):
        with torch.no_grad():
            outputs = []
            for i, model in enumerate(self.models):
                output = model(x)
                scaled_output = output / self.temperatures[i]
                probabilities = F.softmax(scaled_output, dim=1)
                outputs.append(self.weights[i] * probabilities)
            
            ensemble_output = sum(outputs)
            return ensemble_output
        

def verificar_pesos_cargados(model, nombre_modelo):
    # Verificar si los pesos parecen inicializados o cargados
    total_params = 0
    initialized_params = 0
    
    for name, param in model.named_parameters():
        total_params += 1
        # Los pesos inicializados suelen tener media cerca de 0 y std específica
        # dependiendo del inicializador (ej: kaiming, xavier)
        if -0.01 < param.data.mean().item() < 0.01 and param.data.std().item() < 0.05:
            initialized_params += 1
    
    # Si la mayoría de los parámetros parecen inicializados y no cargados
    if initialized_params / total_params > 0.7:  # más del 70% parecen inicializados
        print(f"⚠️ ADVERTENCIA: {nombre_modelo} parece tener pesos mayormente inicializados, no cargados.")
    else:
        print(f"✓ {nombre_modelo} parece tener pesos cargados correctamente.")
    
    # Imprimir algunas estadísticas de capas importantes
    layers_to_check = ["classifier", "fc", "heads", "head"]
    for name, param in model.named_parameters():
        for layer_name in layers_to_check:
            if layer_name in name and "weight" in name:
                print(f"{nombre_modelo} - {name}: media={param.data.mean().item():.4f}, std={param.data.std().item():.4f}")
# ----- Función de evaluación -----

def evaluate_model(model, dataloader, device, threshold=0.5):
    # Asegurarse de que el modelo esté en modo evaluación (si tiene el método)
    if hasattr(model, 'eval'):
        model.eval()
    
    all_targets = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Manejar distintos tipos de modelo
            if isinstance(model, EnsembleModel):
                probs = model.predict(inputs)
                class_probs = probs[:, 1].cpu().numpy()
            else:
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)
                class_probs = probs[:, 1].cpu().numpy()
            
            preds = (class_probs >= threshold).astype(int)
            
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(class_probs)
    
    # Calcular métricas
    f1 = f1_score(all_targets, all_preds)
    
    # DICE (=F1 en clasificación binaria)
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
    dice = 2 * tp / (2 * tp + fp + fn)
    
    # AUC-ROC
    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    auc_roc = auc(fpr, tpr)
    
    metrics = {
        'f1': f1,
        'dice': dice,
        'auc_roc': auc_roc
    }
    
    return metrics, all_targets, all_preds, all_probs

# ----- Optimización de umbrales -----

def optimize_threshold(y_true, y_scores):
    """Encuentra el umbral óptimo para maximizar F1"""
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold

# ----- Optimización de pesos para ensamblado -----

def optimize_weights(models, dataloader, device):
    """Busca los mejores pesos para el ensamblado"""
    weight_options = [
        [0.5, 0.5],  # Iguales
        [0.6, 0.4],  # Más peso a EfficientNet
        [0.7, 0.3],
        [0.4, 0.6],  # Más peso a ViT
        [0.3, 0.7]
    ]
    
    best_f1 = 0
    best_weights = weight_options[0]
    
    for weights in weight_options:
        ensemble = EnsembleModel(models, weights)
        metrics, _, _, _ = evaluate_model(ensemble, dataloader, device)
        
        print(f"Pesos {weights}: F1={metrics['f1']:.4f}, DICE={metrics['dice']:.4f}, AUC-ROC={metrics['auc_roc']:.4f}")
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_weights = weights
    
    return best_weights

def evaluar_con_etiquetas_invertidas(model, dataloader, device):
    model.eval()
    all_inverted_targets = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Invertir las etiquetas (0→1, 1→0)
            inverted_targets = 1 - targets
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_inverted_targets.extend(inverted_targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    # Calcular precisión con etiquetas invertidas
    accuracy = np.mean(np.array(all_inverted_targets) == np.array(all_preds))
    print(f"Precisión con etiquetas invertidas: {accuracy:.4f}")
    
    # Calcular otras métricas
    f1 = f1_score(all_inverted_targets, all_preds)
    print(f"F1 con etiquetas invertidas: {f1:.4f}")
    
    return accuracy, f1

# ----- NUEVA FUNCIÓN: Evaluación completa del ensemble con etiquetas invertidas -----
def evaluate_ensemble_inverted(ensemble, dataloader, device, threshold=0.5):
    """Evalúa el modelo ensemble con las etiquetas invertidas"""
    if hasattr(ensemble, 'eval'):
        ensemble.eval()
    
    all_inverted_targets = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Invertir las etiquetas (0→1, 1→0)
            inverted_targets = 1 - targets
            
            # Manejar el modelo de ensamblado
            probs = ensemble.predict(inputs)
            class_probs = probs[:, 1].cpu().numpy()  # Probabilidad de la clase positiva
            preds = (class_probs >= threshold).astype(int)
            
            all_inverted_targets.extend(inverted_targets.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(class_probs)
    
    # Calcular métricas con etiquetas invertidas
    f1 = f1_score(all_inverted_targets, all_preds)
    
    # DICE (=F1 en clasificación binaria)
    tn, fp, fn, tp = confusion_matrix(all_inverted_targets, all_preds).ravel()
    dice = 2 * tp / (2 * tp + fp + fn)
    
    # AUC-ROC
    fpr, tpr, _ = roc_curve(all_inverted_targets, all_probs)
    auc_roc = auc(fpr, tpr)
    
    # Calcular accuracy
    accuracy = np.mean(np.array(all_inverted_targets) == np.array(all_preds))
    
    metrics = {
        'accuracy': accuracy,
        'f1': f1,
        'dice': dice,
        'auc_roc': auc_roc
    }
    
    return metrics, all_inverted_targets, all_preds, all_probs

# ----- Función principal -----

def main():
    # Rutas a modelos
    efficientnet_path = r"C:\Users\jakif\CODE\PROYECTO-FINAL\COMPUTER_VISION\melanoma_model_1_torch_EFFICIENTNETB0_harvard_attention.pth"
    vit_path = r"C:\Users\jakif\CODE\PROYECTO-FINAL\COMPUTER_VISION\vision_transformers\best_vit_model_2.pth"
    
    # Configuración
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizando dispositivo: {device}")
    
    # Transformaciones
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Cargar modelos
    print("\nCargando EfficientNet...")
    efficientnet_model = load_efficientnet_model(efficientnet_path, device)
    efficientnet_model.to(device)
    
    print("\nCargando Vision Transformer...")
    vit_model = load_vit_model(vit_path, device)
    vit_model.to(device)
    
    print("\nVerificando pesos cargados...")
    verificar_pesos_cargados(efficientnet_model, "EfficientNet")
    verificar_pesos_cargados(vit_model, "Vision Transformer")
    # ===== IMPORTANTE: CARGA DE DATOS DE PRUEBA =====
    # Reemplaza esto con la carga real de tus datos de melanoma
    print("\nCargando datos de prueba...")
    """
    # Ejemplo de cómo configurar FakeData para simular un dataset de imágenes
    from torchvision.datasets import FakeData
    test_dataset = FakeData(size=100, image_size=(3, 224, 224), num_classes=2, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Si tienes datos reales, comenta lo anterior y descomenta esto:
    """
    test_dir = r"C:\Users\jakif\CODE\PROYECTO-FINAL\images\PREPROCESSED_DATA_copy\test"
    benign_images = glob.glob(os.path.join(test_dir, "benign_images", "*.jpg"))
    malignant_images = glob.glob(os.path.join(test_dir, "malignant_images", "*.jpg"))
    
    image_paths = benign_images + malignant_images
    labels = [0] * len(benign_images) + [1] * len(malignant_images)
    
    class MelanomaDataset(Dataset):
        def __init__(self, paths, labels, transform):
            self.paths = paths
            self.labels = labels
            self.transform = transform
            
        def __len__(self):
            return len(self.paths)
            
        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, self.labels[idx]
    
    test_dataset = MelanomaDataset(image_paths, labels, transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    
    # Evaluar modelos individuales
    print("\nEvaluando EfficientNet...")
    eff_metrics, eff_targets, eff_preds, eff_probs = evaluate_model(efficientnet_model, test_loader, device)
    print(f"Métricas: F1={eff_metrics['f1']:.4f}, DICE={eff_metrics['dice']:.4f}, AUC-ROC={eff_metrics['auc_roc']:.4f}")
    
    print("\nEvaluando Vision Transformer...")
    vit_metrics, vit_targets, vit_preds, vit_probs = evaluate_model(vit_model, test_loader, device)
    print(f"Métricas: F1={vit_metrics['f1']:.4f}, DICE={vit_metrics['dice']:.4f}, AUC-ROC={vit_metrics['auc_roc']:.4f}")
    
    print("\nAnalizando correlación de errores...")
    analizar_correlacion_errores(eff_preds, vit_preds, eff_targets)
    
    print("\nEvaluando con etiquetas invertidas...")
    accuracy, f1 = evaluar_con_etiquetas_invertidas(efficientnet_model, test_loader, device)
    # Optimizar umbrales
    print("\nOptimizando umbrales...")
    eff_threshold = optimize_threshold(eff_targets, eff_probs)
    print(f"Umbral óptimo para EfficientNet: {eff_threshold:.4f}")
    
    vit_threshold = optimize_threshold(vit_targets, vit_probs)
    print(f"Umbral óptimo para Vision Transformer: {vit_threshold:.4f}")
    
    # Optimizar ensamblado
    print("\nOptimizando pesos del ensamblado...")
    models = [efficientnet_model, vit_model]
    best_weights = optimize_weights(models, test_loader, device)
    print(f"Mejores pesos para ensamblado: {best_weights}")
    
    # Crear y evaluar el ensamblado final
    ensemble = EnsembleModel(models, best_weights)
    ens_metrics, ens_targets, ens_preds, ens_probs = evaluate_model(ensemble, test_loader, device)
    print(f"\nMétricas del ensamblado: F1={ens_metrics['f1']:.4f}, DICE={ens_metrics['dice']:.4f}, AUC-ROC={ens_metrics['auc_roc']:.4f}")
    
    # Optimizar umbral del ensamblado
    ens_threshold = optimize_threshold(ens_targets, ens_probs)
    print(f"Umbral óptimo para el ensamblado: {ens_threshold:.4f}")
    
    # Evaluación final con umbral optimizado
    final_metrics, _, _, _ = evaluate_model(ensemble, test_loader, device, threshold=ens_threshold)
    print(f"\nMétricas finales optimizadas: F1={final_metrics['f1']:.4f}, DICE={final_metrics['dice']:.4f}, AUC-ROC={final_metrics['auc_roc']:.4f}")
    
    # AÑADIDO: Evaluación final con etiquetas invertidas
    print("\n======= EVALUACIÓN FINAL CON ETIQUETAS INVERTIDAS =======")
    inverted_metrics, inv_targets, inv_preds, inv_probs = evaluate_ensemble_inverted(ensemble, test_loader, device, threshold=ens_threshold)
    print(f"Métricas con etiquetas invertidas:")
    print(f"Accuracy: {inverted_metrics['accuracy']:.4f}")
    print(f"F1: {inverted_metrics['f1']:.4f}")
    print(f"DICE: {inverted_metrics['dice']:.4f}")
    print(f"AUC-ROC: {inverted_metrics['auc_roc']:.4f}")
    
    # Optimizar umbral para etiquetas invertidas
    inv_threshold = optimize_threshold(inv_targets, inv_probs)
    print(f"Umbral óptimo para etiquetas invertidas: {inv_threshold:.4f}")
    
    # Evaluación final con umbral optimizado para etiquetas invertidas
    final_inv_metrics, _, _, _ = evaluate_ensemble_inverted(ensemble, test_loader, device, threshold=inv_threshold)
    print(f"\nMétricas finales con etiquetas invertidas y umbral optimizado:")
    print(f"Accuracy: {final_inv_metrics['accuracy']:.4f}")
    print(f"F1: {final_inv_metrics['f1']:.4f}")
    print(f"DICE: {final_inv_metrics['dice']:.4f}")
    print(f"AUC-ROC: {final_inv_metrics['auc_roc']:.4f}")

if __name__ == "__main__":
    main()