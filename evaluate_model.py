#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script
Generates detailed metrics, graphs, and reports for the saved federated learning model
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_curve, auc, 
    cohen_kappa_score, matthews_corrcoef
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import os
import json
from datetime import datetime
from model import NeuralNetwork
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ComprehensiveModelEvaluator:
    """
    Comprehensive evaluation suite for federated learning model
    Generates metrics, confusion matrices, ROC curves, and detailed reports
    """
    
    def __init__(self, model_path, data_dir, results_dir='EvaluationResults'):
        self.model_path = model_path
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Label mapping (same as training)
        self.allowed_labels = [0, 1, 2, 3, 5, 6, 7, 8, 9]
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
        
        # Reverse mapping for display
        self.class_names = [self.label_mapping[i] for i in self.allowed_labels]
        
        # Storage for results
        self.all_predictions = []
        self.all_labels = []
        self.all_probabilities = []
        
        print(f"Evaluator initialized")
        print(f"Device: {self.device}")
        print(f"Results will be saved to: {self.results_dir}")
    
    def load_model(self):
        """Load the saved federated learning model"""
        print("\n" + "="*60)
        print("Loading Model")
        print("="*60)
        
        # Determine input size from data
        sample_file = os.path.join(self.data_dir, 'device_1_reduced.csv')
        sample_df = pd.read_csv(sample_file)
        input_size = len([col for col in sample_df.columns if col != 'label'])
        
        # Initialize and load model
        self.model = NeuralNetwork(input_size=input_size, num_classes=10)
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"[OK] Model loaded: {os.path.basename(self.model_path)}")
        print(f"[OK] Architecture: {input_size} inputs -> 10 classes")
        print(f"[OK] Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def load_test_data(self, test_size=0.2, random_state=42):
        """
        Load and prepare test data from all available device files
        
        Args:
            test_size: Proportion to use for testing (default 0.2 = 20%)
            random_state: Random seed for reproducibility
        """
        print("\n" + "="*60)
        print("Loading Test Data")
        print("="*60)
        
        all_X = []
        all_y = []
        
        # Find all device files
        device_files = [f for f in os.listdir(self.data_dir) if f.startswith('device_') and f.endswith('.csv')]
        
        print(f"Found {len(device_files)} device data files")
        
        for device_file in device_files:
            file_path = os.path.join(self.data_dir, device_file)
            df = pd.read_csv(file_path)
            
            # Filter for allowed labels
            df_filtered = df[df['label'].isin(self.allowed_labels)].copy()
            
            # Extract features and labels
            feature_columns = [col for col in df_filtered.columns if col != 'label']
            X = df_filtered[feature_columns].values
            y = df_filtered['label'].values
            
            all_X.append(X)
            all_y.append(y)
            
            print(f"  [OK] {device_file}: {len(y)} samples")
        
        # Combine all data
        X_combined = np.vstack(all_X)
        y_combined = np.concatenate(all_y)
        
        print(f"\nTotal samples: {len(y_combined):,}")
        
        # Split into train/test (we only use test portion)
        _, X_test, _, y_test = train_test_split(
            X_combined, y_combined, 
            test_size=test_size, 
            stratify=y_combined, 
            random_state=random_state
        )
        
        # Standardize
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)
        
        # Convert to tensors
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        
        # Create DataLoader
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        
        print(f"Test set size: {len(y_test):,} samples ({test_size*100}%)")
        print(f"Test batches: {len(test_loader)}")
        
        # Print class distribution
        print("\nClass distribution in test set:")
        for label in self.allowed_labels:
            count = np.sum(y_test == label)
            percentage = (count / len(y_test)) * 100
            print(f"  {self.label_mapping[label]:15s}: {count:6d} ({percentage:5.2f}%)")
        
        return test_loader
    
    def evaluate(self, test_loader):
        """Run model evaluation on test data"""
        print("\n" + "="*60)
        print("Running Evaluation")
        print("="*60)
        
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                # Collect results
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Progress
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches", end='\r')
        
        print(f"\n[OK] Evaluation complete: {total:,} samples processed")
        
        # Store results
        self.all_predictions = np.array(all_preds)
        self.all_labels = np.array(all_labels)
        self.all_probabilities = np.array(all_probs)
        
        # Calculate overall accuracy
        overall_accuracy = 100 * correct / total
        print(f"[OK] Overall Accuracy: {overall_accuracy:.2f}%")
        
        return overall_accuracy
    
    def plot_confusion_matrix(self):
        """Generate and save confusion matrix heatmap"""
        print("\n" + "="*60)
        print("Generating Confusion Matrix")
        print("="*60)
        
        cm = confusion_matrix(self.all_labels, self.all_predictions)
        
        # Create single plot with absolute counts
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names,
                    ax=ax, cbar_kws={'label': 'Count'})
        ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=14)
        ax.set_xlabel('Predicted Label', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)
        
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {save_path}")
        plt.close()
        
        return cm
    
    def plot_class_metrics(self):
        """Plot per-class precision, recall, and F1-score"""
        print("\n" + "="*60)
        print("Generating Class-wise Metrics")
        print("="*60)
        
        # Calculate metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            self.all_labels, self.all_predictions, labels=self.allowed_labels
        )
        
        # Create DataFrame for easier plotting
        metrics_df = pd.DataFrame({
            'Class': self.class_names,
            'Precision': precision * 100,
            'Recall': recall * 100,
            'F1-Score': f1 * 100,
            'Support': support
        })
        
        # Plot 1: Precision, Recall, F1-Score comparison
        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(len(self.class_names))
        width = 0.25
        
        ax.bar(x - width, metrics_df['Precision'], width, label='Precision', alpha=0.8)
        ax.bar(x, metrics_df['Recall'], width, label='Recall', alpha=0.8)
        ax.bar(x + width, metrics_df['F1-Score'], width, label='F1-Score', alpha=0.8)
        ax.set_ylabel('Score (%)', fontsize=14)
        ax.set_title('Per-Class Metrics: Precision, Recall, F1-Score', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend(fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 105])
        
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'precision_recall_f1.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {save_path}")
        plt.close()
        
        # Plot 2: F1-Score by class
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['#2ecc71' if f1 >= 0.9 else '#f39c12' if f1 >= 0.7 else '#e74c3c' for f1 in f1]
        ax.barh(self.class_names, metrics_df['F1-Score'], color=colors, alpha=0.8)
        ax.set_xlabel('F1-Score (%)', fontsize=14)
        ax.set_title('F1-Score by Class', fontsize=16, fontweight='bold')
        ax.set_xlim([0, 105])
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'f1_scores.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {save_path}")
        plt.close()
        
        # Plot 3: Support (sample count) by class
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.bar(self.class_names, metrics_df['Support'], color='steelblue', alpha=0.8)
        ax.set_ylabel('Number of Samples', fontsize=14)
        ax.set_title('Test Set Distribution by Class', fontsize=16, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'class_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {save_path}")
        plt.close()
        
        return metrics_df
    
    def plot_roc_curves(self):
        """Generate ROC curves for each class"""
        print("\n" + "="*60)
        print("Generating ROC Curves")
        print("="*60)
        
        # Binarize labels for ROC curve
        y_bin = label_binarize(self.all_labels, classes=self.allowed_labels)
        
        # Calculate ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i, label in enumerate(self.allowed_labels):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], self.all_probabilities[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot
        fig, axes = plt.subplots(3, 3, figsize=(18, 16))
        axes = axes.ravel()
        
        for i, label in enumerate(self.allowed_labels):
            axes[i].plot(fpr[i], tpr[i], color='darkorange', lw=2,
                        label=f'ROC (AUC = {roc_auc[i]:.3f})')
            axes[i].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            axes[i].set_xlim([0.0, 1.0])
            axes[i].set_ylim([0.0, 1.05])
            axes[i].set_xlabel('False Positive Rate', fontsize=10)
            axes[i].set_ylabel('True Positive Rate', fontsize=10)
            axes[i].set_title(f'{self.class_names[i]}', fontsize=12, fontweight='bold')
            axes[i].legend(loc="lower right", fontsize=9)
            axes[i].grid(alpha=0.3)
        
        plt.suptitle('ROC Curves for Each Class', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'roc_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {save_path}")
        plt.close()
        
        return roc_auc
    
    def generate_report(self, overall_accuracy, metrics_df, cm, roc_auc):
        """Generate comprehensive text and JSON reports"""
        print("\n" + "="*60)
        print("Generating Evaluation Report")
        print("="*60)
        
        # Calculate additional metrics
        macro_precision = metrics_df['Precision'].mean()
        macro_recall = metrics_df['Recall'].mean()
        macro_f1 = metrics_df['F1-Score'].mean()
        
        weighted_precision = np.average(metrics_df['Precision'], weights=metrics_df['Support'])
        weighted_recall = np.average(metrics_df['Recall'], weights=metrics_df['Support'])
        weighted_f1 = np.average(metrics_df['F1-Score'], weights=metrics_df['Support'])
        
        kappa = cohen_kappa_score(self.all_labels, self.all_predictions)
        mcc = matthews_corrcoef(self.all_labels, self.all_predictions)
        
        # Mean ROC AUC
        mean_auc = np.mean(list(roc_auc.values()))
        
        # Create text report
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("FEDERATED LEARNING MODEL - EVALUATION REPORT")
        report_lines.append("="*80)
        report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Model: {os.path.basename(self.model_path)}")
        report_lines.append(f"Device: {self.device}")
        report_lines.append(f"\n" + "="*80)
        report_lines.append("OVERALL METRICS")
        report_lines.append("="*80)
        report_lines.append(f"Overall Accuracy:        {overall_accuracy:.2f}%")
        report_lines.append(f"Mean ROC AUC:           {mean_auc:.4f}")
        report_lines.append(f"Cohen's Kappa:          {kappa:.4f}")
        report_lines.append(f"Matthews Corr Coef:     {mcc:.4f}")
        report_lines.append(f"\n{'Metric':<25} {'Macro Avg':<15} {'Weighted Avg'}")
        report_lines.append("-"*55)
        report_lines.append(f"{'Precision':<25} {macro_precision:>6.2f}%        {weighted_precision:>6.2f}%")
        report_lines.append(f"{'Recall':<25} {macro_recall:>6.2f}%        {weighted_recall:>6.2f}%")
        report_lines.append(f"{'F1-Score':<25} {macro_f1:>6.2f}%        {weighted_f1:>6.2f}%")
        
        report_lines.append(f"\n" + "="*80)
        report_lines.append("PER-CLASS METRICS")
        report_lines.append("="*80)
        report_lines.append(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'AUC':<12} {'Support'}")
        report_lines.append("-"*80)
        
        for i, (idx, row) in enumerate(metrics_df.iterrows()):
            report_lines.append(
                f"{row['Class']:<20} "
                f"{row['Precision']:>6.2f}%     "
                f"{row['Recall']:>6.2f}%     "
                f"{row['F1-Score']:>6.2f}%     "
                f"{roc_auc[i]:>6.4f}     "
                f"{int(row['Support']):>6d}"
            )
        
        report_lines.append(f"\n" + "="*80)
        report_lines.append("CONFUSION MATRIX (Counts)")
        report_lines.append("="*80)
        report_lines.append("\nPredicted -->")
        header = "True v".ljust(15) + "  ".join([name[:8].ljust(8) for name in self.class_names])
        report_lines.append(header)
        report_lines.append("-"*80)
        
        for i, true_label in enumerate(self.class_names):
            row_str = true_label[:15].ljust(15) + "  ".join([f"{cm[i][j]:>8d}" for j in range(len(self.class_names))])
            report_lines.append(row_str)
        
        report_lines.append("\n" + "="*80)
        report_lines.append("END OF REPORT")
        report_lines.append("="*80)
        
        # Save text report
        report_text = "\n".join(report_lines)
        text_path = os.path.join(self.results_dir, 'evaluation_report.txt')
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"[OK] Saved text report: {text_path}")
        
        # Create JSON report
        json_report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_path': self.model_path,
                'device': str(self.device),
                'test_samples': len(self.all_labels)
            },
            'overall_metrics': {
                'accuracy': float(overall_accuracy),
                'mean_roc_auc': float(mean_auc),
                'cohen_kappa': float(kappa),
                'matthews_corr_coef': float(mcc),
                'macro_precision': float(macro_precision),
                'macro_recall': float(macro_recall),
                'macro_f1': float(macro_f1),
                'weighted_precision': float(weighted_precision),
                'weighted_recall': float(weighted_recall),
                'weighted_f1': float(weighted_f1)
            },
            'per_class_metrics': []
        }
        
        for i, (idx, row) in enumerate(metrics_df.iterrows()):
            json_report['per_class_metrics'].append({
                'class': row['Class'],
                'precision': float(row['Precision']),
                'recall': float(row['Recall']),
                'f1_score': float(row['F1-Score']),
                'roc_auc': float(roc_auc[i]),
                'support': int(row['Support'])
            })
        
        json_path = os.path.join(self.results_dir, 'evaluation_metrics.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=4)
        print(f"[OK] Saved JSON metrics: {json_path}")
        
        # Print summary to console
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Overall Accuracy: {overall_accuracy:.2f}%")
        print(f"Mean ROC AUC:     {mean_auc:.4f}")
        print(f"Macro F1-Score:   {macro_f1:.2f}%")
        print(f"Cohen's Kappa:    {kappa:.4f}")
        print("="*60)
        
        return report_text

def main():
    """Main evaluation workflow"""
    print("\n" + "="*80)
    print(" "*20 + "FEDERATED LEARNING MODEL EVALUATION")
    print("="*80)
    
    # Configuration
    model_path = r"c:\Users\karthik cse\PycharmProjects\MegaProject\SavedGlobalModel\final_model.pth"
    data_dir = r"c:\Users\karthik cse\PycharmProjects\MegaProject\ReducedData"
    results_dir = r"c:\Users\karthik cse\PycharmProjects\MegaProject\EvaluationResults"
    
    # Verify model exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        print("\nAvailable models in SavedGlobalModel/:")
        model_dir = os.path.dirname(model_path)
        if os.path.exists(model_dir):
            for f in os.listdir(model_dir):
                if f.endswith('.pth'):
                    print(f"  - {f}")
        return
    
    # Initialize evaluator
    evaluator = ComprehensiveModelEvaluator(model_path, data_dir, results_dir)
    
    # Run evaluation pipeline
    evaluator.load_model()
    test_loader = evaluator.load_test_data(test_size=0.2)
    overall_accuracy = evaluator.evaluate(test_loader)
    
    # Generate visualizations
    cm = evaluator.plot_confusion_matrix()
    metrics_df = evaluator.plot_class_metrics()
    roc_auc = evaluator.plot_roc_curves()
    
    # Generate reports
    evaluator.generate_report(overall_accuracy, metrics_df, cm, roc_auc)
    
    print("\n" + "="*80)
    print("[OK] EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {results_dir}")
    print("\nGenerated files:")
    print("  [OK] confusion_matrix.png         - Confusion matrix (counts)")
    print("  [OK] precision_recall_f1.png      - Per-class precision, recall, F1-score")
    print("  [OK] f1_scores.png                - F1-scores by class")
    print("  [OK] class_distribution.png       - Test set distribution")
    print("  [OK] roc_curve_<classname>.png    - Individual ROC curves (9 files)")
    print("  [OK] evaluation_report.txt        - Detailed text report")
    print("  [OK] evaluation_metrics.json      - Machine-readable metrics")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
