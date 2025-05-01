import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
from timm import create_model
import pandas as pd

# Definir la función para calcular el coeficiente Dice
def dice_coefficient(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))

class MedicalImageDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Rutas a las carpetas con imágenes benignas y malignas
        benign_dir = os.path.join(root_dir, split, 'benign_images')
        malignant_dir = os.path.join(root_dir, split, 'malignant_images')
        
        self.image_paths = []
        self.labels = []
        
        # Cargar imágenes benignas
        for img_name in os.listdir(benign_dir):
            if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                self.image_paths.append(os.path.join(benign_dir, img_name))
                self.labels.append(0)  # 0 para benigno
        
        # Cargar imágenes malignas
        for img_name in os.listdir(malignant_dir):
            if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                self.image_paths.append(os.path.join(malignant_dir, img_name))
                self.labels.append(1)  # 1 para maligno
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def create_vit_classifier(num_classes=2):
    # Cargar modelo ViT-B/16 pre-entrenado 
    model = create_model('vit_large_patch16_224', pretrained=True)
    
    # Modificar la capa de clasificación para clasificación binaria
    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Dropout(0.5),  # Añadir dropout para regularización
        nn.Linear(in_features, num_classes)
    )
    
    return model
class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
def train_model(model, train_loader, val_loader, device, num_epochs=30):

        # Calcular pesos para clases desbalanceadas
    class_counts = np.bincount([label for _, label in train_loader.dataset])
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)  # Añadido weight_decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1)  # Cambiado a ReduceLROnPlateau
    
    early_stopping = EarlyStopping(patience=5, verbose=True)  # Añadir early stopping
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 
               'train_auc': [], 'val_auc': [], 'train_dice': [], 'val_dice': [],
               'train_f1': [], 'val_f1': [], 'train_recall': [], 'val_recall': []}
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_probs = []
        train_targets = []
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Añadir label smoothing para regularización
            loss = (1 - 0.1) * loss + 0.1 * -(torch.log_softmax(outputs, dim=1).mean())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            train_preds.extend(preds.cpu().numpy())
            train_probs.extend(probs[:, 1].detach().cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = accuracy_score(train_targets, train_preds)
        train_auc = roc_auc_score(train_targets, train_probs)
        train_dice = dice_coefficient(train_targets, train_preds)
        train_f1 = precision_recall_fscore_support(train_targets, train_preds, average='binary')[2]
        train_recall = precision_recall_fscore_support(train_targets, train_preds, average='binary')[1]
        
        # Modo evaluación
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_probs = []
        val_targets = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                val_preds.extend(preds.cpu().numpy())
                val_probs.extend(probs[:, 1].detach().cpu().numpy())  # Probabilidad de la clase maligna
                val_targets.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(val_targets, val_preds)
        val_auc = roc_auc_score(val_targets, val_probs)
        val_dice = dice_coefficient(val_targets, val_preds)
        val_f1 = precision_recall_fscore_support(val_targets, val_preds, average='binary')[2]
        val_recall = precision_recall_fscore_support(val_targets, val_preds, average='binary')[1]
        # Actualizar historial
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['train_recall'].append(train_recall)
        history['val_recall'].append(val_recall)
        
        # Guardar mejor modelo basado en AUC
        if val_auc > best_val_acc:
            best_val_acc = val_auc
            torch.save(model.state_dict(), 'best_vit_model.pth')
        
        scheduler.step(val_auc)
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
        
        print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, '
              f'Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, '
              f'Train Dice: {train_dice:.4f}, Val Dice: {val_dice:.4f}'
              f'Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, '
              f'Train Recall: {train_recall:.4f}, Val Recall: {val_recall:.4f}')
    
    return history

def evaluate_model(model, test_loader, device):
    model.eval()
    test_preds = []
    test_probs = []
    test_targets = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Usar temperatura para suavizar las probabilidades
            temperature = 0.5
            scaled_outputs = outputs / temperature
            probs = torch.softmax(scaled_outputs, dim=1)
            
            _, preds = torch.max(outputs, 1)
            
            test_preds.extend(preds.cpu().numpy())
            test_probs.extend(probs[:, 1].detach().cpu().numpy())
            test_targets.extend(labels.cpu().numpy())
    
    # Calcular métricas
    accuracy = accuracy_score(test_targets, test_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(test_targets, test_preds, average='binary')
    roc_auc = roc_auc_score(test_targets, test_probs)
    dice = dice_coefficient(test_targets, test_preds)
    conf_matrix = confusion_matrix(test_targets, test_preds)
    
    print(f'Test Metrics:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'AUC-ROC: {roc_auc:.4f}')
    print(f'Dice Coefficient: {dice:.4f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    
    # Visualización de la matriz de confusión
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = ['Benigno', 'Maligno']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Anotar valores en la matriz de confusión
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # Visualizar curva ROC
    fpr, tpr, _ = roc_curve(test_targets, test_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.show()
    
    return accuracy, precision, recall, f1, roc_auc, dice, conf_matrix

def reshape_transform(tensor, height=14, width=14):
    """
    Reshape the ViT transformer output to be compatible with GradCAM
    """
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result
def visualize_gradcam(model, test_loader, device, num_images=5):
    # Import after making sure it's installed
    try:
        from pytorch_grad_cam import GradCAM, EigenCAM, XGradCAM, GradCAMPlusPlus
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        print("pytorch-grad-cam not installed. Install it with: pip install pytorch-grad-cam")
        return
    
    model.eval()
    
    # Get some images from the test set
    batch_images, batch_labels = next(iter(test_loader))
    images = batch_images[:num_images].to(device)
    labels = batch_labels[:num_images].cpu().numpy()
    
    # For ViT, we need to target a different layer
    # Use the last attention block as the target layer
    target_layers = [model.blocks[-1].norm1]  # Last normalization layer
    
    # Use EigenCAM with proper parameters for ViT
    cam = EigenCAM(
        model=model, 
        target_layers=target_layers, 
        use_cuda=(device.type == 'cuda'),
        reshape_transform=reshape_transform  # Add this function (defined below)
    )
    
    plt.figure(figsize=(15, 4*num_images))
    
    for i in range(num_images):
        input_tensor = images[i].unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
        
        pred_class = preds.item()
        pred_prob = probs[0, pred_class].item()
        
        # Generate attention map using EigenCAM - don't specify targets parameter
        grayscale_cam = cam(input_tensor=input_tensor)
        grayscale_cam = grayscale_cam[0, :]
        
        # Convert image to appropriate format for visualization
        img_np = images[i].cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())  # Normalize to [0,1]
        
        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        
        # Show original image and attention map
        plt.subplot(num_images, 3, i*3+1)
        plt.imshow(img_np)
        plt.title(f'Original (True: {"Malignant" if labels[i] == 1 else "Benign"})')
        plt.axis('off')
        
        plt.subplot(num_images, 3, i*3+2)
        plt.imshow(grayscale_cam, cmap='jet')
        plt.title('Attention Map')
        plt.axis('off')
        
        plt.subplot(num_images, 3, i*3+3)
        plt.imshow(visualization)
        plt.title(f'Pred: {"Malignant" if pred_class == 1 else "Benign"} ({pred_prob:.2f})')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('gradcam_visualizations.png')
    plt.show()




def main():
    # Configurar ruta a los datos
    data_dir = r"C:\Users\jakif\CODE\PROYECTO-FINAL\images\PREPROCESSED_DATA_copy"
    
    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Definir transformaciones
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Más variación en el recorte
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),  # Añadido flip vertical
        transforms.RandomRotation(30),  # Mayor rango de rotación
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Más variación de color
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Pequeñas traslaciones
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),  # Añadir blur aleatorio
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),  # Añadir random erasing
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Crear conjuntos de datos
    train_dataset = MedicalImageDataset(root_dir=data_dir, split='train', transform=transform_train)
    test_dataset = MedicalImageDataset(root_dir=data_dir, split='test', transform=transform_test)
    
    # Verificar tamaño de los conjuntos de datos
    print(f"Tamaño del conjunto de entrenamiento: {len(train_dataset)}")
    print(f"Tamaño del conjunto de prueba: {len(test_dataset)}")
    
    # Dividir el conjunto de entrenamiento en entrenamiento y validación (80-20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Crear cargadores de datos
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Crear modelo ViT-B/16
    model = create_vit_classifier(num_classes=2)
    model = model.to(device)
    
    # Resumen del modelo
    print(f"Modelo creado: ViT-B/16 para clasificación binaria")
    
    # Entrenar modelo
    print("Iniciando entrenamiento...")
    history = train_model(model, train_loader, val_loader, device, num_epochs=30)
    
    # Cargar mejor modelo
    model.load_state_dict(torch.load('best_vit_model.pth'))
    
    # Evaluar modelo en conjunto de prueba
    print("Evaluando en conjunto de prueba...")
    accuracy, precision, recall, f1, roc_auc, dice, conf_matrix = evaluate_model(model, test_loader, device)
    
    # Visualizar resultados de entrenamiento
    plt.figure(figsize=(15, 10))
    
    # Gráfico de pérdida
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    
    # Gráfico de precisión
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()
    
    # Gráfico de AUC-ROC
    plt.subplot(2, 2, 3)
    plt.plot(history['train_auc'], label='Train AUC')
    plt.plot(history['val_auc'], label='Val AUC')
    plt.title('AUC-ROC')
    plt.legend()
    
    # Gráfico de Dice
    plt.subplot(2, 2, 4)
    plt.plot(history['train_dice'], label='Train Dice')
    plt.plot(history['val_dice'], label='Val Dice')
    plt.title('Dice Coefficient')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    # Visualizar GradCAM para algunas imágenes de prueba

    print("Generando visualizaciones GradCAM...")
    visualize_gradcam(model, test_loader, device)
    
    # Guardar resultados en un archivo
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': roc_auc,
        'dice': dice
    }
    
    # Guardar resultados en un archivo CSV
    pd.DataFrame([results]).to_csv('vit_classification_results.csv', index=False)
    print("Resultados guardados en 'vit_classification_results.csv'")

if __name__ == "__main__":
    # Asegurarse de que se instalen las dependencias necesarias
    try:
        import timm
    except ImportError:
        print("Instalando dependencias necesarias...")
        import subprocess
        subprocess.check_call(["pip", "install", "torch", "torchvision", "timm", "pytorch-grad-cam", "scikit-learn", "matplotlib", "pandas", "pillow"])
        
    main()