import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import timm
import os

# Definir la clase SEBlock necesaria para el modelo EfficientNet
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Definir la clase SEEfficientNet
class SEEfficientNet(nn.Module):
    def __init__(self, num_classes=1):
        super(SEEfficientNet, self).__init__()
        self.base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.features = self.base_model.features
        
        # Añadir bloques SE a cada bloque de características
        for i in range(len(self.features)):
            if hasattr(self.features[i], 'block'):
                channels = self.features[i]._blocks[-1].project_conv.out_channels
                se_block = SEBlock(channels)
                setattr(self, f'se_block_{i}', se_block)
        
        # Capa de clasificación
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=1280, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=512, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=128, out_features=num_classes)
        )
    
    def forward(self, x):
        for i in range(len(self.features)):
            x = self.features[i](x)
            if hasattr(self, f'se_block_{i}'):
                se_block = getattr(self, f'se_block_{i}')
                x = se_block(x)
        
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        
        return self.classifier(x)

# Función para crear el modelo ViT
def create_vit_classifier(num_classes=2, dropout_rate=0.2):
    model = timm.create_model('vit_large_patch16_224', pretrained=True, drop_rate=dropout_rate)
    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(in_features, num_classes)
    )
    return model

# Definir la clase EnsembleModel
class EnsembleModel(nn.Module):
    def __init__(self, vit_model, efficientnet_model, vit_weight=0.6, efficientnet_weight=0.4):
        super(EnsembleModel, self).__init__()
        self.vit_model = vit_model
        self.efficientnet_model = efficientnet_model
        self.vit_weight = vit_weight
        self.efficientnet_weight = efficientnet_weight
        
    def forward(self, x):
        vit_output = self.vit_model(x)
        efficientnet_output = self.efficientnet_model(x)
        
        # Ajustar dimensiones según la forma de salida
        if vit_output.shape[1] > 1 and efficientnet_output.shape[1] == 1:
            eff_sigmoid = torch.sigmoid(efficientnet_output).view(-1, 1)
            efficientnet_probs = torch.cat([1 - eff_sigmoid, eff_sigmoid], dim=1)
            vit_probs = torch.softmax(vit_output, dim=1)
            
        elif vit_output.shape[1] == 1 and efficientnet_output.shape[1] == 1:
            vit_probs = torch.sigmoid(vit_output)
            efficientnet_probs = torch.sigmoid(efficientnet_output)
            
        else:
            vit_probs = torch.softmax(vit_output, dim=1)
            efficientnet_probs = torch.softmax(efficientnet_output, dim=1)
        
        # Combinar con pesos
        ensemble_output = (self.vit_weight * vit_probs + 
                          self.efficientnet_weight * efficientnet_probs)
        
        return ensemble_output
    
    def get_individual_predictions(self, x):
        vit_output = self.vit_model(x)
        efficientnet_output = self.efficientnet_model(x)
        return vit_output, efficientnet_output

def load_and_test_model(test_data_path, model_path, vit_model_path, efficientnet_model_path, num_classes=2):
    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Cargar modelos base
    print("Cargando modelos base...")
    vit_model = create_vit_classifier(num_classes=num_classes)
    
    # Cargar estado del modelo ViT
    vit_checkpoint = torch.load(vit_model_path, map_location=device)
    if 'model_state_dict' in vit_checkpoint:
        vit_model.load_state_dict(vit_checkpoint['model_state_dict'])
    else:
        vit_model.load_state_dict(vit_checkpoint)
    
    vit_model = vit_model.to(device)
    vit_model.eval()
    
    # Cargar modelo EfficientNet
    if num_classes == 2:
        efficientnet_model = SEEfficientNet(num_classes=1)
    else:
        efficientnet_model = SEEfficientNet(num_classes=num_classes)
    
    eff_checkpoint = torch.load(efficientnet_model_path, map_location=device)
    if 'model_state_dict' in eff_checkpoint:
        efficientnet_model.load_state_dict(eff_checkpoint['model_state_dict'])
    else:
        efficientnet_model.load_state_dict(eff_checkpoint)
    
    efficientnet_model = efficientnet_model.to(device)
    efficientnet_model.eval()
    
    # Crear modelo ensemble
    ensemble_model = EnsembleModel(vit_model, efficientnet_model)
    
    # Cargar pesos optimizados
    print("Cargando ensemble optimizado...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extraer información del checkpoint
    if 'best_weights' in checkpoint:
        vit_weight, efficientnet_weight = checkpoint['best_weights']
        best_method = checkpoint.get('best_method', 'No especificado')
        best_fitness = checkpoint.get('best_fitness', 0)
        
        print(f"Método: {best_method}")
        print(f"Pesos: ViT={vit_weight:.4f}, EfficientNet={efficientnet_weight:.4f}")
        print(f"F1 Score reportado: {best_fitness:.4f}")
        
        # Actualizar pesos en el modelo
        ensemble_model.vit_weight = vit_weight
        ensemble_model.efficientnet_weight = efficientnet_weight
    
    # Cargar estado del modelo si existe
    if 'model_state_dict' in checkpoint:
        ensemble_model.load_state_dict(checkpoint['model_state_dict'])
    
    ensemble_model = ensemble_model.to(device)
    ensemble_model.eval()
    
    # Preparar datos de test
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = ImageFolder(root=test_data_path, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    print(f"Conjunto de datos cargado con {len(test_dataset)} imágenes")
    
    # Evaluar el modelo
    print("Evaluando el modelo optimizado...")
    all_preds = []
    all_true = []
    all_preds_vit = []
    all_preds_efficientnet = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Predicciones del ensemble
            outputs = ensemble_model(inputs)
            
            # Obtener predicciones individuales
            vit_outputs, efficientnet_outputs = ensemble_model.get_individual_predictions(inputs)
            
            # Procesar predicciones ensemble
            if num_classes == 2:
                if outputs.shape[1] > 1:
                    preds = (outputs[:, 1] > 0.5).float()
                else:
                    preds = (outputs.view(-1) > 0.5).float()
            else:
                preds = torch.argmax(outputs, dim=1)
            
            # Procesar predicciones individuales
            if num_classes == 2:
                if vit_outputs.shape[1] > 1:
                    vit_preds = (vit_outputs[:, 1] > 0.5).float()
                else:
                    vit_preds = (vit_outputs.view(-1) > 0.5).float()
                
                if efficientnet_outputs.shape[1] > 1:
                    efficientnet_preds = (efficientnet_outputs[:, 1] > 0.5).float()
                else:
                    efficientnet_preds = (efficientnet_outputs.view(-1) > 0.5).float()
            else:
                vit_preds = torch.argmax(vit_outputs, dim=1)
                efficientnet_preds = torch.argmax(efficientnet_outputs, dim=1)
            
            # Guardar predicciones y etiquetas
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(labels.cpu().numpy())
            all_preds_vit.extend(vit_preds.cpu().numpy())
            all_preds_efficientnet.extend(efficientnet_preds.cpu().numpy())
    
    # Calcular métricas para el ensemble
    print("\n--- Resultados del Ensemble Optimizado ---")
    cm = confusion_matrix(all_true, all_preds)
    f1 = f1_score(all_true, all_preds, average='weighted')
    
    print(f"F1 Score (weighted): {f1:.4f}")
    print("Matriz de Confusión:")
    print(cm)
    print("\nReporte de Clasificación:")
    print(classification_report(all_true, all_preds))
    
    # Calcular métricas para los modelos individuales
    f1_vit = f1_score(all_true, all_preds_vit, average='weighted')
    f1_efficientnet = f1_score(all_true, all_preds_efficientnet, average='weighted')
    
    print("\n--- Comparación de Modelos ---")
    print(f"ViT F1 Score: {f1_vit:.4f}")
    print(f"EfficientNet F1 Score: {f1_efficientnet:.4f}")
    print(f"Ensemble F1 Score: {f1:.4f}")
    
    # Visualizar matriz de confusión
    plt.figure(figsize=(10, 8))
    class_names = test_dataset.classes
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Optimized Ensemble')
    plt.tight_layout()
    plt.savefig('confusion_matrix_test.png')
    
    # Visualizar comparación de F1 scores
    plt.figure(figsize=(10, 6))
    models = ['ViT', 'EfficientNet', 'Ensemble']
    scores = [f1_vit, f1_efficientnet, f1]
    plt.bar(models, scores, color=['blue', 'green', 'orange'])
    plt.ylim(0, 1.0)
    plt.ylabel('F1 Score')
    plt.title('Model Performance Comparison')
    for i, v in enumerate(scores):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center')
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    
    print("\nGráficos guardados como 'confusion_matrix_test.png' y 'model_comparison.png'")
    
    return f1, f1_vit, f1_efficientnet

if __name__ == "__main__":
    # Definir rutas (actualiza estas rutas según tu entorno)
    TEST_DATA_PATH = r"C:\Users\jakif\CODE\PROYECTO-FINAL\images\melanoma_cancer_dataset\test"  # Actualiza esta ruta
    MODEL_PATH = r"C:\Users\jakif\CODE\PROYECTO-FINAL\COMPUTER_VISION\Ensamblado_mejores_modelos\optimized_ensemble_model.pth"  # Ruta al modelo optimizado
    VIT_MODEL_PATH = r"C:\Users\jakif\CODE\PROYECTO-FINAL\COMPUTER_VISION\vision_transformers\best_vit_model.pth"
    EFFICIENT_MODEL_PATH = r"C:\Users\jakif\CODE\PROYECTO-FINAL\COMPUTER_VISION\melanoma_model_1_torch_EFFICIENTNETB0_harvard_attention.pth"
    # Ejecutar la evaluación
    load_and_test_model(
        TEST_DATA_PATH, 
        MODEL_PATH, 
        VIT_MODEL_PATH, 
        EFFICIENT_MODEL_PATH, 
        num_classes=2  # Ajusta según tu caso
    )