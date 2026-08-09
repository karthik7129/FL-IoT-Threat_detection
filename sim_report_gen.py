import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import os
from model import NeuralNetwork
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    def __init__(self, model_path, data_dir, results_dir):
        """
        Initialize the model evaluator
        
        Args:
            model_path: Path to the saved model
            data_dir: Directory containing the reduced data files
            results_dir: Directory to save results
        """
        self.model_path = model_path
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Label mapping - only for allowed labels as in client1.py
        self.allowed_labels = [0, 1, 2, 3, 5, 6, 7, 8, 9]  # Same as client1.py
        self.label_mapping = {
            0: 'benign',
            1: 'gafgyt.combo', 
            2: 'gafgyt.junk',
            3: 'gafgyt.tcp',
            5: 'mirai.ack',
            6: 'mirai.scan',
            7: 'mirai.syn',
            8: 'mirai.udp',
            9: 'mirai.udpplain'
        }
        
        # Store results for reporting
        self.evaluation_results = {}
        
    def load_model(self):
        """Load the saved global model"""
        # First, determine input size from one of the data files
        sample_file = os.path.join(self.data_dir, 'device_1_reduced.csv')
        sample_df = pd.read_csv(sample_file)
        input_size = len([col for col in sample_df.columns if col != 'label'])
        
        # Initialize model with correct input size
        self.model = NeuralNetwork(input_size=input_size, num_classes=10)
        
        # Load the saved state dict
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully from {self.model_path}")
        print(f"Model architecture: Input size = {input_size}, Output classes = 10")
        
    def load_and_preprocess_data(self, file_path):
        """
        Load and preprocess data following the same approach as client1.py
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            X_scaled: Preprocessed features
            y: Labels
        """
        # Read the dataset
        df = pd.read_csv(file_path)
        print(f"Dataset shape for {os.path.basename(file_path)}: {df.shape}")

        # Filter for labels 0,1,2,3,5,6,7,8,9 (as in client1.py)
        df_filtered = df[df['label'].isin(self.allowed_labels)].copy()
        print(f"Filtered dataset shape: {df_filtered.shape}")

        # Separate features and labels
        feature_columns = [col for col in df_filtered.columns if col != 'label']
        X = df_filtered[feature_columns].values
        y = df_filtered['label'].values

        # Standardize features (as in client1.py)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return X_scaled, y, df_filtered
    
    def evaluate_device(self, device_file):
        """
        Evaluate model on a specific device's data
        
        Args:
            device_file: Path to device data file
            
        Returns:
            Dictionary containing evaluation metrics
        """
        device_name = os.path.basename(device_file).replace('.csv', '')
        print(f"\n{'='*50}")
        print(f"Evaluating on {device_name}")
        print(f"{'='*50}")
        
        # Load and preprocess data
        X, y, df_filtered = self.load_and_preprocess_data(device_file)
        
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        # Create DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
        
        # Evaluate model
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
        
        # Per-class metrics
        unique_labels = sorted(set(all_labels))
        class_report = classification_report(all_labels, all_predictions, 
                                           labels=unique_labels,
                                           target_names=[self.label_mapping[i] for i in unique_labels],
                                           output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Store results
        results = {
            'device': device_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'loss': total_loss / len(dataloader),
            'predictions': all_predictions,
            'labels': all_labels,
            'confusion_matrix': cm,
            'class_report': class_report,
            'data_distribution': df_filtered['label'].value_counts().sort_index()
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Average Loss: {results['loss']:.4f}")
        
        return results
    
    def evaluate_single_device(self):
        """Evaluate model on a single device (device_1)"""
        device_file = 'device_1_reduced.csv'
        file_path = os.path.join(self.data_dir, device_file)
        
        if not os.path.exists(file_path):
            print(f"Error: {device_file} not found in {self.data_dir}")
            return
        
        print(f"Testing on single device: {device_file}")
        device_results = self.evaluate_device(file_path)
        device_name = device_results['device']
        self.evaluation_results[device_name] = device_results
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix for the evaluated device"""
        if not self.evaluation_results:
            print("No evaluation results to plot")
            return
            
        device, results = next(iter(self.evaluation_results.items()))
        cm = results['confusion_matrix']
        labels = sorted(set(results['labels']))
        label_names = [self.label_mapping[label] for label in labels]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=label_names, yticklabels=label_names)
        plt.title(f'Confusion Matrix - {device}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        save_path = os.path.join(self.results_dir, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Confusion matrix saved to: {save_path}")
    
    def plot_performance_metrics(self):
        """Plot basic performance metrics"""
        if not self.evaluation_results:
            print("No evaluation results to plot")
            return
            
        device, results = next(iter(self.evaluation_results.items()))
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        values = [results[metric] for metric in metrics]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, values, alpha=0.8, color=['skyblue', 'lightgreen', 'orange', 'lightcoral'])
        plt.title(f'Performance Metrics - {device}')
        plt.ylabel('Score')
        plt.ylim(0, 1.1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(self.results_dir, 'performance_metrics.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Performance metrics plot saved to: {save_path}")
    
    def generate_detailed_report(self):
        """Generate a detailed evaluation report"""
        print("\n" + "="*80)
        print("MODEL EVALUATION REPORT")
        print("="*80)
        
        if not self.evaluation_results:
            print("No evaluation results to report")
            return
            
        device, results = next(iter(self.evaluation_results.items()))
        
        print(f"\nEVALUATION RESULTS FOR {device.upper()}:")
        print(f"{'='*40}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1-Score: {results['f1_score']:.4f}")
        print(f"Loss: {results['loss']:.4f}")
        print(f"Total Samples: {len(results['labels'])}")
        
        # Class-wise performance
        print(f"\nCLASS-WISE PERFORMANCE:")
        print(f"{'='*40}")
        class_f1_scores = {k: v['f1-score'] for k, v in results['class_report'].items() 
                         if k not in ['accuracy', 'macro avg', 'weighted avg']}
        
        for class_name, f1_score in sorted(class_f1_scores.items(), key=lambda x: x[1], reverse=True):
            precision = results['class_report'][class_name]['precision']
            recall = results['class_report'][class_name]['recall']
            support = results['class_report'][class_name]['support']
            print(f"  {class_name:15}: P={precision:.3f}, R={recall:.3f}, F1={f1_score:.3f}, Support={support}")
        
        # Save report to file
        self.save_report_to_file()
    
    def save_report_to_file(self):
        """Save the evaluation report to a text file"""
        report_path = os.path.join(self.results_dir, 'evaluation_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("MODEL EVALUATION REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Data Directory: {self.data_dir}\n")
            f.write(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}\n")
            f.write(f"Allowed Labels: {self.allowed_labels}\n\n")
            
            if not self.evaluation_results:
                f.write("No evaluation results available.\n")
                return
                
            device, results = next(iter(self.evaluation_results.items()))
            
            f.write(f"EVALUATION RESULTS FOR {device}:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"Precision: {results['precision']:.4f}\n")
            f.write(f"Recall: {results['recall']:.4f}\n")
            f.write(f"F1-Score: {results['f1_score']:.4f}\n")
            f.write(f"Loss: {results['loss']:.4f}\n")
            f.write(f"Total Samples: {len(results['labels'])}\n\n")
            
            # Class-wise results
            f.write("CLASS-WISE PERFORMANCE:\n")
            f.write("-" * 30 + "\n")
            class_f1_scores = {k: v['f1-score'] for k, v in results['class_report'].items() 
                             if k not in ['accuracy', 'macro avg', 'weighted avg']}
            
            for class_name, f1_score in sorted(class_f1_scores.items(), key=lambda x: x[1], reverse=True):
                precision = results['class_report'][class_name]['precision']
                recall = results['class_report'][class_name]['recall']
                support = results['class_report'][class_name]['support']
                f.write(f"{class_name:15}: P={precision:.3f}, R={recall:.3f}, F1={f1_score:.3f}, Support={support}\n")
        
        print(f"\nReport saved to: {report_path}")

def main():
    """Main function to run the evaluation"""
    # Paths
    model_path = r"c:\Users\karthik cse\PycharmProjects\MegaProject\SavedGlobalModel\final_model.pth"
    data_dir = r"c:\Users\karthik cse\PycharmProjects\MegaProject\ReducedData"
    results_dir = r"c:\Users\karthik cse\PycharmProjects\MegaProject\Results"
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        return
    
    print("Starting Model Evaluation...")
    print(f"Model: {model_path}")
    print(f"Data Directory: {data_dir}")
    print(f"Results Directory: {results_dir}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model_path, data_dir, results_dir)
    
    # Load model
    evaluator.load_model()
    
    # Evaluate on single device
    evaluator.evaluate_single_device()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    evaluator.plot_confusion_matrix()
    evaluator.plot_performance_metrics()
    
    # Generate detailed report
    evaluator.generate_detailed_report()
    
    print("\nEvaluation completed successfully!")
    print("Generated files in Results folder:")
    print("  - confusion_matrix.png")
    print("  - performance_metrics.png") 
    print("  - evaluation_report.txt")

if __name__ == "__main__":
    main()
