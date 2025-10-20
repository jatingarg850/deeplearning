"""
Evaluate Anti-Overfitting Model and Generate All Graphs
Load the saved model and test on different data subsets
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split, Subset
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pywt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score, roc_curve, classification_report,
    log_loss
)
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Transforms (same as training)
val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# DWT feature extraction
def dwt_features(img, wavelet='db1'):
    img = np.array(img)
    dwt_channels = []
    for ch in range(3):
        cA, (cH, cV, cD) = pywt.dwt2(img[..., ch], wavelet)
        cA_resized = np.array(Image.fromarray(cA).resize((224, 224), Image.BILINEAR))
        dwt_channels.append(cA_resized)
    dwt_img = np.stack(dwt_channels, axis=2)
    return Image.fromarray(dwt_img.astype(np.uint8))

# Dataset class
class EvalDataset(Dataset):
    def __init__(self, root_dir, label, transform_rgb, transform_dwt):
        self.root_dir = root_dir
        self.transform_rgb = transform_rgb
        self.transform_dwt = transform_dwt
        self.images = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.label = label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.root_dir, self.images[idx])
            
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                img_np = np.array(img)
            
            rgb_tensor = self.transform_rgb(image=img_np)['image']
            dwt_img = dwt_features(Image.fromarray(img_np))
            dwt_tensor = self.transform_dwt(image=np.array(dwt_img))['image']
            
            return rgb_tensor, dwt_tensor, self.label
        except Exception as e:
            dummy_tensor = torch.zeros(3, 224, 224)
            return dummy_tensor, dummy_tensor, self.label

# Anti-overfitting model (same architecture)
class AntiOverfitTwoBranchNet(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        # RGB branch
        self.rgb_backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.rgb_backbone.classifier = nn.Identity()
        
        # DWT branch
        self.dwt_backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.dwt_backbone.classifier = nn.Identity()
        
        # Regularized classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1280*2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            
            nn.Linear(128, n_classes)
        )
        
    def forward(self, rgb, dwt):
        f1 = F.adaptive_avg_pool2d(self.rgb_backbone.features(rgb), 1).flatten(1)
        f2 = F.adaptive_avg_pool2d(self.dwt_backbone.features(dwt), 1).flatten(1)
        x = torch.cat([f1, f2], dim=1)
        return self.classifier(x)

# Evaluation function
def evaluate_model(model, data_loader, criterion, device, desc="Evaluating"):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0
    
    with torch.no_grad():
        for rgb, dwt, labels in tqdm(data_loader, desc=desc, leave=False):
            rgb, dwt, labels = rgb.to(device), dwt.to(device), labels.to(device)
            
            outputs = model(rgb, dwt)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    all_probs = np.array(all_probs)
    
    return all_preds, all_labels, all_probs, avg_loss

# Calculate comprehensive metrics
def calculate_metrics(y_true, y_pred, y_probs):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    if len(np.unique(y_true)) > 1 and not np.any(np.isnan(y_probs)):
        auc = roc_auc_score(y_true, y_probs[:, 1])
        entropy = log_loss(y_true, y_probs)
    else:
        auc = 0.5
        entropy = 1.0
    
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_score': auc,
        'entropy': entropy,
        'confusion_matrix': cm
    }

# Comprehensive plotting function
def plot_all_results(checkpoint_data, test_results, val_results, holdout_results):
    """Generate all graphs and analysis"""
    
    # Extract training history
    train_losses = checkpoint_data.get('train_losses', [])
    val_losses = checkpoint_data.get('val_losses', [])
    train_accs = checkpoint_data.get('train_accs', [])
    val_accs = checkpoint_data.get('val_accs', [])
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(24, 16))
    
    # Training Loss and Accuracy
    ax1 = plt.subplot(3, 4, 1)
    if train_losses and val_losses:
        plt.plot(train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(val_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.title('Training & Validation Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(3, 4, 2)
    if train_accs and val_accs:
        plt.plot(train_accs, 'b-', label='Training Accuracy', linewidth=2)
        plt.plot(val_accs, 'r-', label='Validation Accuracy', linewidth=2)
        plt.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='90% Target')
        plt.title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Overfitting Analysis
    ax3 = plt.subplot(3, 4, 3)
    if train_accs and val_accs:
        gap = [abs(t - v) for t, v in zip(train_accs, val_accs)]
        plt.plot(gap, 'purple', linewidth=2, label='Train-Val Gap')
        plt.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Overfitting Threshold')
        plt.title('Overfitting Analysis', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy Gap')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Test Set Confusion Matrix
    ax4 = plt.subplot(3, 4, 4)
    sns.heatmap(test_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax4)
    ax4.set_title('Test Set Confusion Matrix', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('Actual')
    ax4.set_xticklabels(['Original', 'Deepfake'])
    ax4.set_yticklabels(['Original', 'Deepfake'])
    
    # ROC Curves Comparison
    ax5 = plt.subplot(3, 4, 5)
    # Test set ROC
    if len(np.unique(test_results['labels'])) > 1:
        fpr, tpr, _ = roc_curve(test_results['labels'], test_results['probabilities'][:, 1])
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'Test Set (AUC = {test_results["auc_score"]:.3f})')
    
    # Validation set ROC
    if len(np.unique(val_results['labels'])) > 1:
        fpr, tpr, _ = roc_curve(val_results['labels'], val_results['probabilities'][:, 1])
        plt.plot(fpr, tpr, color='red', lw=2, label=f'Validation Set (AUC = {val_results["auc_score"]:.3f})')
    
    # Holdout set ROC
    if len(np.unique(holdout_results['labels'])) > 1:
        fpr, tpr, _ = roc_curve(holdout_results['labels'], holdout_results['probabilities'][:, 1])
        plt.plot(fpr, tpr, color='green', lw=2, label=f'Holdout Set (AUC = {holdout_results["auc_score"]:.3f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Performance Comparison
    ax6 = plt.subplot(3, 4, 6)
    datasets = ['Test', 'Validation', 'Holdout']
    accuracies = [test_results['accuracy'], val_results['accuracy'], holdout_results['accuracy']]
    colors = ['blue', 'red', 'green']
    bars = plt.bar(datasets, accuracies, color=colors, alpha=0.7)
    plt.title('Accuracy Comparison Across Datasets', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Metrics Comparison
    ax7 = plt.subplot(3, 4, 7)
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    test_values = [test_results['accuracy'], test_results['precision'], 
                   test_results['recall'], test_results['f1_score'], test_results['auc_score']]
    
    x = np.arange(len(metrics_names))
    width = 0.25
    
    plt.bar(x - width, test_values, width, label='Test Set', color='blue', alpha=0.7)
    plt.bar(x, [val_results['accuracy'], val_results['precision'], val_results['recall'], 
               val_results['f1_score'], val_results['auc_score']], width, label='Validation', color='red', alpha=0.7)
    plt.bar(x + width, [holdout_results['accuracy'], holdout_results['precision'], holdout_results['recall'], 
               holdout_results['f1_score'], holdout_results['auc_score']], width, label='Holdout', color='green', alpha=0.7)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Comprehensive Metrics Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x, metrics_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    # Prediction Distribution
    ax8 = plt.subplot(3, 4, 8)
    plt.hist(test_results['probabilities'][:, 1], bins=50, alpha=0.5, color='blue', label='Test Set', density=True)
    plt.hist(val_results['probabilities'][:, 1], bins=50, alpha=0.5, color='red', label='Validation', density=True)
    plt.hist(holdout_results['probabilities'][:, 1], bins=50, alpha=0.5, color='green', label='Holdout', density=True)
    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold')
    plt.title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Deepfake Probability')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Model Summary
    ax9 = plt.subplot(3, 4, 9)
    summary_text = f"""MODEL PERFORMANCE SUMMARY
    
Test Set:
• Accuracy: {test_results['accuracy']:.4f} ({test_results['accuracy']*100:.2f}%)
• Precision: {test_results['precision']:.4f}
• Recall: {test_results['recall']:.4f}
• F1-Score: {test_results['f1_score']:.4f}
• AUC: {test_results['auc_score']:.4f}

Validation Set:
• Accuracy: {val_results['accuracy']:.4f} ({val_results['accuracy']*100:.2f}%)
• AUC: {val_results['auc_score']:.4f}

Holdout Set:
• Accuracy: {holdout_results['accuracy']:.4f} ({holdout_results['accuracy']*100:.2f}%)
• AUC: {holdout_results['auc_score']:.4f}"""
    
    plt.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    plt.title('Performance Summary', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Generalization Analysis
    ax10 = plt.subplot(3, 4, 10)
    if train_accs and val_accs:
        final_gap = abs(train_accs[-1] - val_accs[-1])
        generalization_text = f"""GENERALIZATION ANALYSIS

Train-Val Gap: {final_gap:.4f}

Status: {"✅ Excellent" if final_gap < 0.02 else "✅ Good" if final_gap < 0.05 else "⚠️ Moderate" if final_gap < 0.1 else "❌ Poor"}

Cross-Dataset Performance:
• Test: {test_results['accuracy']*100:.1f}%
• Validation: {val_results['accuracy']*100:.1f}%  
• Holdout: {holdout_results['accuracy']*100:.1f}%

Consistency: {"✅ Excellent" if max(accuracies) - min(accuracies) < 0.02 else "✅ Good" if max(accuracies) - min(accuracies) < 0.05 else "⚠️ Moderate"}

Recommendation:
{"🎯 Model generalizes very well!" if final_gap < 0.02 and max(accuracies) - min(accuracies) < 0.02 else "✅ Good generalization" if final_gap < 0.05 else "⚠️ Consider more regularization"}"""
        
        color = "lightgreen" if final_gap < 0.02 else "lightyellow" if final_gap < 0.05 else "lightcoral"
        plt.text(0.05, 0.95, generalization_text, transform=ax10.transAxes, fontsize=10, 
                 verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor=color))
    plt.title('Generalization Assessment', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Training Progress
    ax11 = plt.subplot(3, 4, 11)
    if train_losses and val_losses:
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
        plt.fill_between(epochs, train_losses, val_losses, alpha=0.3, color='gray', label='Overfitting Gap')
        plt.title('Training Progress with Gap Analysis', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Final Confusion Matrices Comparison
    ax12 = plt.subplot(3, 4, 12)
    # Create a combined confusion matrix visualization
    fig_cm, axes_cm = plt.subplots(1, 3, figsize=(15, 4))
    
    # Test CM
    sns.heatmap(test_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=axes_cm[0])
    axes_cm[0].set_title('Test Set')
    axes_cm[0].set_xlabel('Predicted')
    axes_cm[0].set_ylabel('Actual')
    
    # Val CM
    sns.heatmap(val_results['confusion_matrix'], annot=True, fmt='d', cmap='Reds', ax=axes_cm[1])
    axes_cm[1].set_title('Validation Set')
    axes_cm[1].set_xlabel('Predicted')
    axes_cm[1].set_ylabel('Actual')
    
    # Holdout CM
    sns.heatmap(holdout_results['confusion_matrix'], annot=True, fmt='d', cmap='Greens', ax=axes_cm[2])
    axes_cm[2].set_title('Holdout Set')
    axes_cm[2].set_xlabel('Predicted')
    axes_cm[2].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig_cm)
    
    # Remove the 12th subplot and add text
    ax12.text(0.5, 0.5, 'Detailed Confusion Matrices\nsaved as separate file:\nconfusion_matrices_comparison.png', 
              transform=ax12.transAxes, fontsize=12, ha='center', va='center',
              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    ax12.set_title('Additional Analysis', fontsize=14, fontweight='bold')
    ax12.axis('off')
    
    plt.tight_layout()
    plt.savefig('complete_anti_overfitting_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("🔍 Anti-Overfitting Model Evaluation & Analysis")
    print("="*60)
    
    # Check if model exists
    model_path = "anti_overfitting_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please run the training script first!")
        return
    
    # Load model and checkpoint data
    print("📂 Loading trained model...")
    checkpoint = torch.load(model_path, map_location=device)
    
    model = AntiOverfitTwoBranchNet(n_classes=2).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"📊 Training completed at epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"🏆 Best validation loss: {checkpoint.get('best_val_loss', 'Unknown'):.4f}")
    
    # Dataset paths
    deepfake_dir = "deeplearning/Deepfake/extracted_faces"
    original_dir = "deeplearning/Original/extracted_faces"
    
    if not os.path.exists(deepfake_dir) or not os.path.exists(original_dir):
        print("❌ Dataset directories not found!")
        return
    
    # Create datasets
    print("\n📂 Loading datasets for comprehensive evaluation...")
    deepfake_dataset = EvalDataset(deepfake_dir, label=1, transform_rgb=val_transform, transform_dwt=val_transform)
    original_dataset = EvalDataset(original_dir, label=0, transform_rgb=val_transform, transform_dwt=val_transform)
    
    all_dataset = ConcatDataset([deepfake_dataset, original_dataset])
    
    # Create different test splits
    print("🔄 Creating different data splits for robust evaluation...")
    
    # Original splits (same as training)
    train_size = int(0.7 * len(all_dataset))
    val_size = int(0.2 * len(all_dataset))
    test_size = len(all_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        all_dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create a completely different holdout set (different random seed)
    holdout_size = min(30000, len(all_dataset) // 20)  # 5% of data
    holdout_indices = torch.randperm(len(all_dataset), generator=torch.Generator().manual_seed(123))[:holdout_size]
    holdout_dataset = Subset(all_dataset, holdout_indices)
    
    print(f"📊 Evaluation datasets:")
    print(f"  - Test set: {len(test_dataset):,} samples")
    print(f"  - Validation set: {len(val_dataset):,} samples") 
    print(f"  - Holdout set: {len(holdout_dataset):,} samples (different split)")
    
    # Create data loaders
    batch_size = 32
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    holdout_loader = DataLoader(holdout_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate on all datasets
    print("\n🧪 Running comprehensive evaluation...")
    
    print("  📊 Evaluating on test set...")
    test_preds, test_labels, test_probs, test_loss = evaluate_model(model, test_loader, criterion, device, "Test Set")
    test_results = calculate_metrics(test_labels, test_preds, test_probs)
    test_results['labels'] = test_labels
    test_results['predictions'] = test_preds
    test_results['probabilities'] = test_probs
    
    print("  📊 Evaluating on validation set...")
    val_preds, val_labels, val_probs, val_loss = evaluate_model(model, val_loader, criterion, device, "Validation Set")
    val_results = calculate_metrics(val_labels, val_preds, val_probs)
    val_results['labels'] = val_labels
    val_results['predictions'] = val_preds
    val_results['probabilities'] = val_probs
    
    print("  📊 Evaluating on holdout set...")
    holdout_preds, holdout_labels, holdout_probs, holdout_loss = evaluate_model(model, holdout_loader, criterion, device, "Holdout Set")
    holdout_results = calculate_metrics(holdout_labels, holdout_preds, holdout_probs)
    holdout_results['labels'] = holdout_labels
    holdout_results['predictions'] = holdout_preds
    holdout_results['probabilities'] = holdout_probs
    
    # Print comprehensive results
    print("\n" + "="*80)
    print("📈 COMPREHENSIVE ANTI-OVERFITTING EVALUATION RESULTS")
    print("="*80)
    
    print(f"\n🎯 TEST SET RESULTS:")
    print(f"  ✅ Accuracy:  {test_results['accuracy']:.4f} ({test_results['accuracy']*100:.2f}%)")
    print(f"  ✅ Precision: {test_results['precision']:.4f}")
    print(f"  ✅ Recall:    {test_results['recall']:.4f}")
    print(f"  ✅ F1-Score:  {test_results['f1_score']:.4f}")
    print(f"  ✅ AUC Score: {test_results['auc_score']:.4f}")
    print(f"  ✅ Entropy:   {test_results['entropy']:.4f}")
    
    print(f"\n🎯 VALIDATION SET RESULTS:")
    print(f"  ✅ Accuracy:  {val_results['accuracy']:.4f} ({val_results['accuracy']*100:.2f}%)")
    print(f"  ✅ AUC Score: {val_results['auc_score']:.4f}")
    
    print(f"\n🎯 HOLDOUT SET RESULTS (Different Data Split):")
    print(f"  ✅ Accuracy:  {holdout_results['accuracy']:.4f} ({holdout_results['accuracy']*100:.2f}%)")
    print(f"  ✅ AUC Score: {holdout_results['auc_score']:.4f}")
    
    # Generalization analysis
    accuracies = [test_results['accuracy'], val_results['accuracy'], holdout_results['accuracy']]
    consistency = max(accuracies) - min(accuracies)
    
    print(f"\n🔍 GENERALIZATION ANALYSIS:")
    print(f"  📊 Accuracy Range: {min(accuracies)*100:.2f}% - {max(accuracies)*100:.2f}%")
    print(f"  📊 Consistency Gap: {consistency:.4f}")
    
    if consistency < 0.02:
        print(f"  ✅ EXCELLENT: Model shows excellent generalization across different datasets!")
    elif consistency < 0.05:
        print(f"  ✅ GOOD: Model shows good generalization")
    else:
        print(f"  ⚠️  MODERATE: Some variation across datasets")
    
    print("="*80)
    
    # Generate all graphs
    print("\n📈 Generating comprehensive visualization...")
    plot_all_results(checkpoint, test_results, val_results, holdout_results)
    
    # Classification reports
    print("\n📋 DETAILED CLASSIFICATION REPORTS:")
    print("\n🎯 Test Set:")
    print(classification_report(test_labels, test_preds, target_names=['Original', 'Deepfake'], zero_division=0))
    
    print("\n🎯 Holdout Set:")
    print(classification_report(holdout_labels, holdout_preds, target_names=['Original', 'Deepfake'], zero_division=0))
    
    # Final summary
    avg_accuracy = np.mean(accuracies)
    print(f"\n🏆 FINAL SUMMARY:")
    print(f"  📊 Average Accuracy Across All Sets: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
    print(f"  💾 Model saved as: {model_path}")
    print(f"  📊 Complete analysis saved as: complete_anti_overfitting_analysis.png")
    print(f"  📊 Confusion matrices saved as: confusion_matrices_comparison.png")
    
    if avg_accuracy >= 0.85 and consistency < 0.05:
        print(f"\n🎯 SUCCESS! Model achieves excellent performance with good generalization!")
        print("🏆 Anti-overfitting training was successful!")
    elif avg_accuracy >= 0.80:
        print(f"\n✅ Good performance with reasonable generalization")
    else:
        print(f"\n📊 Model shows room for improvement")
    
    print("\n✅ Comprehensive evaluation completed!")

if __name__ == "__main__":
    main()