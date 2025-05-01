import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import timm
from torch.optim.lr_scheduler import OneCycleLR
import os

# Paths a los modelos pre-entrenados
VIT_MODEL_PATH = r"C:\Users\jakif\CODE\PROYECTO-FINAL\COMPUTER_VISION\vision_transformers\best_vit_model.pth"
EFFICIENT_MODEL_PATH = r"C:\Users\jakif\CODE\PROYECTO-FINAL\COMPUTER_VISION\melanoma_model_1_torch_EFFICIENTNETB0_harvard_attention.pth"

# Definimos el dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Recreamos la arquitectura del modelo ViT
def create_vit_classifier(num_classes=2, dropout_rate=0.2):
    # Use vit_large_patch16_224 to match the checkpoint dimensions
    model = timm.create_model('vit_large_patch16_224', pretrained=True, drop_rate=dropout_rate)
    
    # Modify the classification layer
    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(in_features, num_classes)
    )
    
    return model
    
    
# Definición del bloque SE para EfficientNet
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Squeeze: Global Average Pooling
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # Reducción de dimensionalidad
            nn.ReLU(inplace=True),  # Activación
            nn.Linear(channel // reduction, channel, bias=False),  # Restauración de dimensionalidad
            nn.Sigmoid()  # Escala entre 0 y 1 para recalibración
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)  # [B, C, H, W] -> [B, C, 1, 1] -> [B, C]
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)  # [B, C] -> [B, C, 1, 1]
        # Recalibración: multiplicación de canales por sus pesos
        return x * y.expand_as(x)  # Broadcast y multiplicar

# Recreamos la arquitectura del modelo EfficientNet con SE
class SEEfficientNet(nn.Module):
    def __init__(self, num_classes=1):
        super(SEEfficientNet, self).__init__()
        # Cargar modelo base preentrenado
        self.base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # Obtener la capa de características del EfficientNet
        self.features = self.base_model.features
        
        # Añadir bloques SE a cada bloque de características
        for i in range(len(self.features)):
            # Obtenemos el número de canales del bloque actual
            if hasattr(self.features[i], 'block'):
                # Si es un MBConvBlock, añadir SE a nivel de bloque
                channels = self.features[i]._blocks[-1].project_conv.out_channels
                # Insertar un bloque SE después de cada MBConvBlock
                se_block = SEBlock(channels)
                # Guardamos el bloque SE en el módulo para que pueda ser accedido durante forward
                setattr(self, f'se_block_{i}', se_block)
        
        # Capa de clasificación más compleja con dropout y capas intermedias
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),  # Dropout para regularización
            nn.Linear(in_features=1280, out_features=512),
            nn.BatchNorm1d(512),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=512, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=128, out_features=num_classes)  # Salida final (binaria)
        )
    
    def forward(self, x):
        # Pasar por cada bloque de features y aplicar SE donde corresponda
        for i in range(len(self.features)):
            # Aplicar el bloque convolucional
            x = self.features[i](x)
            # Si tiene un bloque SE asociado, aplicarlo
            if hasattr(self, f'se_block_{i}'):
                se_block = getattr(self, f'se_block_{i}')
                x = se_block(x)
        
        # Global average pooling y flatten
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        
        # Clasificador
        return self.classifier(x)

# Definir el modelo de ensamblado
class EnsembleModel(nn.Module):
    def __init__(self, vit_model, efficientnet_model, vit_weight=0.6, efficientnet_weight=0.4):
        super(EnsembleModel, self).__init__()
        self.vit_model = vit_model
        self.efficientnet_model = efficientnet_model
        self.vit_weight = vit_weight
        self.efficientnet_weight = efficientnet_weight
        
    def forward(self, x):
        # Obtenemos predicciones de cada modelo
        vit_output = self.vit_model(x)
        efficientnet_output = self.efficientnet_model(x)
        
        # Verificamos y ajustamos dimensiones
        # Para clasificación binaria, aseguramos que ambos modelos tengan la misma forma
        if vit_output.shape[1] > 1 and efficientnet_output.shape[1] == 1:
            # Si ViT da [batch_size, 2] y EfficientNet da [batch_size, 1]
            # Convertimos EfficientNet a formato ViT
            eff_sigmoid = torch.sigmoid(efficientnet_output).view(-1, 1)
            efficientnet_probs = torch.cat([1 - eff_sigmoid, eff_sigmoid], dim=1)
            
            # Usamos softmax para ViT
            vit_probs = torch.softmax(vit_output, dim=1)
            
        elif vit_output.shape[1] == 1 and efficientnet_output.shape[1] == 1:
            # Ambos son binarios con sigmoid
            vit_probs = torch.sigmoid(vit_output)
            efficientnet_probs = torch.sigmoid(efficientnet_output)
            
        else:
            # Multiclase: usamos softmax para ambos
            vit_probs = torch.softmax(vit_output, dim=1)
            efficientnet_probs = torch.softmax(efficientnet_output, dim=1)
        
        # Combinamos las predicciones con pesos
        ensemble_output = (self.vit_weight * vit_probs + 
                          self.efficientnet_weight * efficientnet_probs)
        
        return ensemble_output
    
    def get_individual_predictions(self, x):
        """Retorna las predicciones individuales de cada modelo para análisis"""
        vit_output = self.vit_model(x)
        efficientnet_output = self.efficientnet_model(x)
        
        return vit_output, efficientnet_output

# Función para cargar los modelos pre-entrenados
def load_pretrained_models(num_classes=2):
    # Cargar modelo ViT
    vit_model = create_vit_classifier(num_classes=num_classes)
    
    # Cargar estado del modelo ViT
    vit_checkpoint = torch.load(VIT_MODEL_PATH, map_location=device)
    # Comprobar si es un checkpoint completo o solo el state_dict
    if 'model_state_dict' in vit_checkpoint:
        vit_model.load_state_dict(vit_checkpoint['model_state_dict'])
    else:
        vit_model.load_state_dict(vit_checkpoint)
    
    vit_model = vit_model.to(device)
    vit_model.eval()  # Poner en modo evaluación
    
    # Cargar modelo EfficientNet
    if num_classes == 2:
        # Para clasificación binaria, EfficientNet usa 1 salida
        efficientnet_model = SEEfficientNet(num_classes=1)
    else:
        efficientnet_model = SEEfficientNet(num_classes=num_classes)
    
    # Cargar estado del modelo EfficientNet
    eff_checkpoint = torch.load(EFFICIENT_MODEL_PATH, map_location=device)
    
    # Comprobar si es un checkpoint completo o solo el state_dict
    if 'model_state_dict' in eff_checkpoint:
        efficientnet_model.load_state_dict(eff_checkpoint['model_state_dict'])
    else:
        efficientnet_model.load_state_dict(eff_checkpoint)
    
    efficientnet_model = efficientnet_model.to(device)
    efficientnet_model.eval()  # Poner en modo evaluación
    
    return vit_model, efficientnet_model

# Función para evaluar el modelo ensamblado
def evaluate_ensemble(ensemble_model, test_loader, num_classes=2):
    ensemble_model.eval()
    all_preds = []
    all_true = []
    
    vit_preds = []
    efficientnet_preds = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Imprimir dimensiones para depuración
            batch_size = inputs.size(0)
            print(f"Batch size: {batch_size}, Labels shape: {labels.shape}")
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Obtenemos predicciones del ensamble
            outputs = ensemble_model(inputs)
            print(f"Ensemble output shape: {outputs.shape}")
            
            # También obtenemos predicciones individuales para análisis
            vit_outputs, efficientnet_outputs = ensemble_model.get_individual_predictions(inputs)
            print(f"ViT output shape: {vit_outputs.shape}, EfficientNet output shape: {efficientnet_outputs.shape}")
            
            # Procesamos las predicciones según sea clasificación binaria o multiclase
            if num_classes == 2:
                # Binary classification - asegurarnos de que todas las salidas tengan la forma correcta
                if outputs.shape[1] > 1:  # Si es [batch_size, 2]
                    # Tomamos la probabilidad de la clase positiva (índice 1)
                    preds = (outputs[:, 1] > 0.5).float()
                else:  # Si es [batch_size, 1] o [batch_size]
                    preds = (outputs.view(-1) > 0.5).float()
                
                # Lo mismo para las predicciones individuales
                if vit_outputs.shape[1] > 1:
                    vit_probs = torch.softmax(vit_outputs, dim=1)[:, 1]  # Probabilidad de clase positiva
                    vit_preds_batch = (vit_probs > 0.5).float()
                else:
                    vit_preds_batch = (torch.sigmoid(vit_outputs.view(-1)) > 0.5).float()
                
                if efficientnet_outputs.shape[1] > 1:
                    eff_probs = torch.softmax(efficientnet_outputs, dim=1)[:, 1]
                    efficientnet_preds_batch = (eff_probs > 0.5).float()
                else:
                    efficientnet_preds_batch = (torch.sigmoid(efficientnet_outputs.view(-1)) > 0.5).float()
                
            else:
                # Multiclass classification
                preds = torch.argmax(outputs, dim=1)
                vit_preds_batch = torch.argmax(torch.softmax(vit_outputs, dim=1), dim=1)
                efficientnet_preds_batch = torch.argmax(torch.softmax(efficientnet_outputs, dim=1), dim=1)
            
            # Asegurarse de que todo tenga la forma correcta
            preds = preds.view(-1)
            vit_preds_batch = vit_preds_batch.view(-1)
            efficientnet_preds_batch = efficientnet_preds_batch.view(-1)
            
            # Asegurarse de que las etiquetas tengan la forma correcta
            labels = labels.view(-1)
            
            # Verificar formas después de ajustes
            print(f"After reshaping - preds: {preds.shape}, labels: {labels.shape}")
            print(f"After reshaping - vit_preds: {vit_preds_batch.shape}, eff_preds: {efficientnet_preds_batch.shape}")
            
            # Verificar que las dimensiones coinciden
            if preds.shape[0] != labels.shape[0]:
                raise ValueError(f"Error de dimensiones: preds={preds.shape}, labels={labels.shape}")
            
            # Convertir a numpy y añadir a las listas
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(labels.cpu().numpy())
            
            vit_preds.extend(vit_preds_batch.cpu().numpy())
            efficientnet_preds.extend(efficientnet_preds_batch.cpu().numpy())
    
    # Imprimir longitudes para depuración
    print(f"Longitud de all_true: {len(all_true)}")
    print(f"Longitud de all_preds: {len(all_preds)}")
    print(f"Longitud de vit_preds: {len(vit_preds)}")
    print(f"Longitud de efficientnet_preds: {len(efficientnet_preds)}")
    
    # Verificar que las longitudes coinciden
    if len(all_true) != len(all_preds):
        raise ValueError(f"Error: all_true({len(all_true)}) y all_preds({len(all_preds)}) tienen longitudes diferentes")
    
    # Calculamos métricas para el ensamble
    cm = confusion_matrix(all_true, all_preds)
    report = classification_report(all_true, all_preds, output_dict=True)
    f1 = f1_score(all_true, all_preds, average='weighted')
    
    # Calculamos métricas para modelos individuales
    vit_f1 = f1_score(all_true, vit_preds, average='weighted')
    efficientnet_f1 = f1_score(all_true, efficientnet_preds, average='weighted')
    
    print(f"F1 Score (ViT): {vit_f1:.4f}")
    print(f"F1 Score (EfficientNet): {efficientnet_f1:.4f}")
    print(f"F1 Score (Ensemble): {f1:.4f}")
    
    return cm, report, f1

# Función para plotear la matriz de confusión
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Guardar la matriz de confusión
    plt.savefig('confusion_matrix_ensemble_final_kaggle.png')
    plt.show()

# Definimos el conjunto de transformaciones para evaluación
def get_test_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Función principal para crear y evaluar el ensamblado
def create_and_evaluate_ensemble(test_loader, num_classes=2, vit_weight=0.6, efficientnet_weight=0.4):
    # Cargar modelos pre-entrenados
    vit_model, efficientnet_model = load_pretrained_models(num_classes)
    
    # Crear modelo ensamblado
    ensemble_model = EnsembleModel(
        vit_model, 
        efficientnet_model, 
        vit_weight=vit_weight, 
        efficientnet_weight=efficientnet_weight
    ).to(device)
    
    # Evaluar el modelo ensamblado
    cm, report, f1 = evaluate_ensemble(ensemble_model, test_loader, num_classes)
    
    # Plotear matriz de confusión
    if num_classes == 2:
        class_names = ['Clase 0', 'Clase 1']
    else:
        class_names = [f'Clase {i}' for i in range(num_classes)]
    
    plot_confusion_matrix(cm, class_names)
    
    # Mostrar reporte detallado
    print("\nInforme de clasificación:")
    for cls in report:
        if cls not in ['accuracy', 'macro avg', 'weighted avg']:
            print(f"Clase {cls}: Precision={report[cls]['precision']:.4f}, Recall={report[cls]['recall']:.4f}, F1={report[cls]['f1-score']:.4f}")
    
    print(f"\nAccuracy: {report['accuracy']:.4f}")
    print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1: {report['weighted avg']['f1-score']:.4f}")
    
    return ensemble_model, cm, report

# Ejemplo de uso (comentado ya que necesita el conjunto de datos real)
"""
# Definir el conjunto de datos
test_dataset = YourDataset(root="path_to_test_data", transform=get_test_transforms())
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Crear y evaluar el ensamblado (ajusta los pesos según sea necesario)
ensemble_model, cm, report = create_and_evaluate_ensemble(
    test_loader, 
    num_classes=2,  # Ajusta según tu caso (2 para binario, >2 para multiclase)
    vit_weight=0.6,  # Peso para el modelo ViT
    efficientnet_weight=0.4  # Peso para el modelo EfficientNet
)
"""

# Definir el conjunto de datos
path_test =r"C:\Users\jakif\CODE\PROYECTO-FINAL\images\PREPROCESSED_DATA_copy\test"
from torchvision.datasets import ImageFolder

test_dataset = ImageFolder(root=path_test, transform=get_test_transforms())
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Crear y evaluar el ensamblado (ajusta los pesos según sea necesario)
ensemble_model, cm, report = create_and_evaluate_ensemble(
    test_loader, 
    num_classes=2,  # Ajusta según tu caso (2 para binario, >2 para multiclase)
    vit_weight=0.6,  # Peso para el modelo ViT
    efficientnet_weight=0.4  # Peso para el modelo EfficientNet
)
