import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import pipeline
from torch.utils.data import Dataset, DataLoader
import os
import argparse

# Define command line arguments
parser = argparse.ArgumentParser(description='Train and evaluate transformer model for melanoma classification')
parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')
parser.add_argument('--save_path', type=str, default='./finetuned_model', help='Path to save the model')
parser.add_argument('--test_size', type=float, default=0.2, help='Test set proportion')
parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
parser.add_argument('--model_name', type=str, default='rjac/biobert-ICD10-L3-mimic', help='Base model name')
args = parser.parse_args()

# Custom PyTorch Dataset class
class MelanomaDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Return a dictionary with inputs and label
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
    # Method to get original text
    def get_text(self, idx):
        return self.texts[idx]

# Function to load data
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    texts = [item['text'] for item in data['data']]
    labels = [item['label'] for item in data['data']]
    
    # Create label to index mapping
    unique_labels = sorted(set(labels))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for idx, label in enumerate(unique_labels)}
    
    # Convert labels to indices
    label_ids = [label_to_id[label] for label in labels]
    
    return texts, label_ids, label_to_id, id_to_label, data

# Function to predict with a model - using the model directly without pipeline
def predict_with_model(model, tokenizer, texts, device="cpu"):
    model.eval()
    model.to(device)
    predictions = []
    scores = []
    
    with torch.no_grad():
        for text in texts:
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Prediction
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Get class with highest probability
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            pred_score = probs[0, pred_class].item()
            
            predictions.append(pred_class)
            scores.append(pred_score)
            
    return predictions, scores

# Function to evaluate the model on the test set
def evaluate_model(model, test_dataset, test_texts, tokenizer, id_to_label):
    # Get true labels
    true_labels = [item['labels'].item() for i, item in enumerate(test_dataset)]
    
    # Make predictions directly with the model
    predictions, confidence_scores = predict_with_model(model, tokenizer, test_texts)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='weighted'
    )
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Create confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=[id_to_label[i].replace('_', ' ').title() for i in range(len(id_to_label))],
        yticklabels=[id_to_label[i].replace('_', ' ').title() for i in range(len(id_to_label))]
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    # Create DataFrame with predictions and confidence
    results_df = pd.DataFrame({
        'True Label': [id_to_label[label] for label in true_labels],
        'Predicted': [id_to_label[pred] for pred in predictions],
        'Confidence': confidence_scores,
        'Correct': [true_labels[i] == predictions[i] for i in range(len(true_labels))]
    })
    
    # Group by true label and calculate average confidence
    avg_confidence = results_df.groupby('True Label')['Confidence'].mean().reset_index()
    
    # Sort labels by concern level
    order = ['not_concerning', 'mildly_concerning', 'moderately_concerning', 'highly_concerning']
    avg_confidence['True Label'] = pd.Categorical(avg_confidence['True Label'], categories=order, ordered=True)
    avg_confidence = avg_confidence.sort_values('True Label')
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='True Label', y='Confidence', data=avg_confidence)
    plt.title('Average Confidence Score by Class')
    plt.xlabel('True Label')
    plt.ylabel('Average Confidence')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confidence_by_class.png')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'confidence_scores': results_df
    }

# Main function
def main():
    print(f"Loading data from 'melanoma_data.json'...")
    texts, label_ids, label_to_id, id_to_label, raw_data = load_data('dataset_answers.json')
    
    print(f"Dataset loaded with {len(texts)} examples and {len(label_to_id)} classes")
    print("Available labels:")
    for label, idx in label_to_id.items():
        print(f"  - {label}: {idx}")
    
    # Split data into training and test sets
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, label_ids, test_size=args.test_size, random_state=42, stratify=label_ids
    )
    
    print(f"Training set: {len(train_texts)} examples")
    print(f"Test set: {len(test_texts)} examples")
    
    # Load tokenizer
    print(f"Loading tokenizer for '{args.model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Create datasets
    train_dataset = MelanomaDataset(train_texts, train_labels, tokenizer)
    test_dataset = MelanomaDataset(test_texts, test_labels, tokenizer)
    
    # Load pretrained model
    print(f"Loading pretrained model '{args.model_name}'...")
    
    # Correct label mapping configuration
    num_labels = len(label_to_id)
    id2label = {str(i): label for i, label in id_to_label.items()}
    label2id = {label: str(i) for label, i in label_to_id.items()}
    
    print("Label configuration:")
    print("id2label:", id2label)
    print("label2id:", label2id)
    
    pretrained_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True  # Ignore mismatched layer sizes
    )
    
    # Evaluate pretrained model
    print("Evaluating pretrained model...")
    pretrained_results = evaluate_model(pretrained_model, test_dataset, test_texts, tokenizer, id_to_label)
    
    # Print pretrained model results
    print("\nPretrained model results:")
    print(f"Accuracy: {pretrained_results['accuracy']:.4f}")
    print(f"Precision: {pretrained_results['precision']:.4f}")
    print(f"Recall: {pretrained_results['recall']:.4f}")
    print(f"F1-Score: {pretrained_results['f1']:.4f}")
    
    # Define function to compute metrics during training
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='weighted'
        )
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    # Configure training arguments
    print(f"Setting up training with {args.epochs} epochs and batch size {args.batch_size}...")
    
    # Create results directory if it doesn't exist
    os.makedirs('./results', exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=0,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        learning_rate=args.learning_rate,
        report_to='none',  # Disable reporting to Weights & Biases, etc.
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=pretrained_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    print("Starting model training...")
    trainer.train()
    
    # Save the fine-tuned model
    print(f"Saving fine-tuned model to '{args.save_path}'...")
    trainer.save_model(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    
    # Save label configuration
    config_file = os.path.join(args.save_path, "label_config.json")
    with open(config_file, "w") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f)
    
    # Evaluate the fine-tuned model
    print("Evaluating fine-tuned model...")
    finetuned_model = trainer.model
    finetuned_results = evaluate_model(finetuned_model, test_dataset, test_texts, tokenizer, id_to_label)
    
    # Print fine-tuned model results
    print("\nFine-tuned model results:")
    print(f"Accuracy: {finetuned_results['accuracy']:.4f}")
    print(f"Precision: {finetuned_results['precision']:.4f}")
    print(f"Recall: {finetuned_results['recall']:.4f}")
    print(f"F1-Score: {finetuned_results['f1']:.4f}")
    
    # Compare results
    print("\nFine-tuned model improvement over pretrained:")
    print(f"Accuracy: {finetuned_results['accuracy'] - pretrained_results['accuracy']:.4f} (+{(finetuned_results['accuracy'] - pretrained_results['accuracy']) * 100:.1f}%)")
    print(f"Precision: {finetuned_results['precision'] - pretrained_results['precision']:.4f} (+{(finetuned_results['precision'] - pretrained_results['precision']) * 100:.1f}%)")
    print(f"Recall: {finetuned_results['recall'] - pretrained_results['recall']:.4f} (+{(finetuned_results['recall'] - pretrained_results['recall']) * 100:.1f}%)")
    print(f"F1-Score: {finetuned_results['f1'] - pretrained_results['f1']:.4f} (+{(finetuned_results['f1'] - pretrained_results['f1']) * 100:.1f}%)")
    
    # Save results
    results = {
        'pretrained': {
            'accuracy': float(pretrained_results['accuracy']),
            'precision': float(pretrained_results['precision']),
            'recall': float(pretrained_results['recall']),
            'f1': float(pretrained_results['f1'])
        },
        'finetuned': {
            'accuracy': float(finetuned_results['accuracy']),
            'precision': float(finetuned_results['precision']),
            'recall': float(finetuned_results['recall']),
            'f1': float(finetuned_results['f1'])
        }
    }
    
    with open('model_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Results saved to 'model_comparison_results.json'")
    
    # Visualize comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    pretrained_scores = [results['pretrained'][m] for m in metrics]
    finetuned_scores = [results['finetuned'][m] for m in metrics]
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, pretrained_scores, width, label='Pretrained')
    plt.bar(x + width/2, finetuned_scores, width, label='Fine-tuned')
    
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title('Model performance comparison')
    plt.xticks(x, metrics)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig('model_comparison.png')
    print("Comparison plot saved to 'model_comparison.png'")
    
    # Save examples for interface reference
    sample_examples = {}
    for label in label_to_id.keys():
        examples = [item['text'] for item in raw_data['data'] if item['label'] == label]
        sample_examples[label] = examples[:5]  # Save up to 5 examples per category
    
    with open('sample_examples.json', 'w') as f:
        json.dump(sample_examples, f, indent=2)
    
    print("Examples saved to 'sample_examples.json'")

if __name__ == "__main__":
    main()