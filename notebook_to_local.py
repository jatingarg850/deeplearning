"""
Converted from paper-3-2-2.ipynb to run locally
Evaluates the trained deepfake detection model on local dataset
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import numpy as np
from PIL import Image
import pywt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)
from tqdm import tqdm

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom FF++ dataset (label 0 = Original, 1 = Deepfake)
class CustomFaceDataset(Dataset):
    def __init__(self, root_dir, label, transform_rgb, transform_dwt):
        self.root_dir = root_dir
        self.transform_rgb = transform_rgb
        self.transform_dwt = transform_dwt
        self.images = [f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        self.label = label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        img = Image.open(img_path).convert('RGB')
        
        # RGB tensor
        rgb_tensor = self.transform_rgb(img)
        
        # DWT tensor
        dwt_img = self.dwt_features(img)
        dwt_tensor = self.transform_dwt(dwt_img)
        
        return rgb_tensor, dwt_tensor, self.label

    def dwt_features(self, img, wavelet='db1'):
        img = np.array(img)
        dwt_channels = []
        for ch in range(3):
            cA, _ = pywt.dwt2(img[..., ch], wavelet)
            cA_resized = Image.fromarray(cA).resize(img.shape[:2][::-1], Image.BILINEAR)
            dwt_channels.append(np.array(cA_resized))
        dwt_img = np.stack(dwt_channels, axis=2)
        return Image.fromarray(dwt_img.astype(np.uint8))

# Simple model architecture (from notebook)
class SimpleTwoBranchNet(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.rgb_backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.rgb_backbone.classifier = nn.Identity()
        self.dwt_backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.dwt_backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(1280*2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, n_classes)
        )
    
    def forward(self, rgb, dwt):
        f1 = F.adaptive_avg_pool2d(self.rgb_backbone.features(rgb), 1).view(rgb.size(0), -1)
        f2 = F.adaptive_avg_pool2d(self.dwt_backbone.features(dwt), 1).view(dwt.size(0), -1)
        x = torch.cat([f1, f2], dim=1)
        return self.classifier(x)

# Robust model architecture (your trained model)
class RobustTwoBranchNet(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        # RGB branch
        self.rgb_backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.rgb_backbone.classifier = nn.Identity()
        
        # DWT branch  
        self.dwt_backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.dwt_backbone.classifier = nn.Identity()
        
        # Robust classifier
        self.classifier = nn.Sequential(
            nn.Linear(1280*2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(256, n_classes)
        )
        
    def forward(self, rgb, dwt):
        # Extract features
        f1 = F.adaptive_avg_pool2d(self.rgb_backbone.features(rgb), 1).flatten(1)
        f2 = F.adaptive_avg_pool2d(self.dwt_backbone.features(dwt), 1).flatten(1)
        
        # Concatenate and classify
        x = torch.cat([f1, f2], dim=1)
        return self.classifier(x)

# TorchVision transforms (from notebook)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

def evaluate_model_comprehensive(model, test_loader, device):
    """Comprehensive model evaluation with all metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("🔍 Running comprehensive evaluation...")
    
    with torch.no_grad():
        for rgb, dwt, labels in tqdm(test_loader, desc="Evaluating"):
            rgb, dwt, labels = rgb.to(device), dwt.to(device), labels.to(device)
            
            outputs = model(rgb, dwt)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    all_probs = np.array(all_probs)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # AUC and other metrics
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        auc = 0.5
    
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_score': auc,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }

def plot_results(results):
    """Create comprehensive result plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Confusion Matrix
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
    axes[0,0].set_title('Confusion Matrix')
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('Actual')
    axes[0,0].set_xticklabels(['Original', 'Deepfake'])
    axes[0,0].set_yticklabels(['Original', 'Deepfake'])
    
    # ROC Curve
    if len(np.unique(results['labels'])) > 1:
        fpr, tpr, _ = roc_curve(results['labels'], results['probabilities'][:, 1])
        axes[0,1].plot(fpr, tpr, color='darkorange', lw=2, 
                      label=f'ROC curve (AUC = {results["auc_score"]:.4f})')
        axes[0,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0,1].set_xlim([0.0, 1.0])
        axes[0,1].set_ylim([0.0, 1.05])
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC Curve')
        axes[0,1].legend(loc="lower right")
    
    # Metrics Bar Plot
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    metrics_values = [results['accuracy'], results['precision'], results['recall'], 
                     results['f1_score'], results['auc_score']]
    bars = axes[1,0].bar(metrics_names, metrics_values, 
                        color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'])
    axes[1,0].set_title('Performance Metrics')
    axes[1,0].set_ylabel('Score')
    axes[1,0].set_ylim([0, 1])
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                      f'{value:.3f}', ha='center', va='bottom')
    
    # Prediction Distribution
    axes[1,1].hist(results['probabilities'][:, 1], bins=50, alpha=0.7, 
                  color='purple', edgecolor='black')
    axes[1,1].set_title('Prediction Probability Distribution')
    axes[1,1].set_xlabel('Deepfake Probability')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('notebook_evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("🚀 Starting Notebook-to-Local Evaluation")
    print("="*50)
    
    # Dataset paths (local)
    deepfake_dir = "deeplearning/Deepfake/extracted_faces"
    original_dir = "deeplearning/Original/extracted_faces"
    
    # Check if directories exist
    if not os.path.exists(deepfake_dir) or not os.path.exists(original_dir):
        print("❌ Dataset directories not found!")
        print(f"Expected: {deepfake_dir}")
        print(f"Expected: {original_dir}")
        return
    
    # Create datasets
    print("📂 Loading datasets...")
    deepfake_dataset = CustomFaceDataset(deepfake_dir, 1, test_transform, test_transform)
    original_dataset = CustomFaceDataset(original_dir, 0, test_transform, test_transform)
    
    # Combine datasets
    test_dataset = ConcatDataset([deepfake_dataset, original_dataset])
    
    # Use a subset for faster evaluation (10% of data)
    subset_size = min(50000, len(test_dataset) // 10)
    indices = torch.randperm(len(test_dataset))[:subset_size]
    from torch.utils.data import Subset
    test_subset = Subset(test_dataset, indices)
    
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"📊 Dataset Info:")
    print(f"  - Deepfake samples: {len(deepfake_dataset):,}")
    print(f"  - Original samples: {len(original_dataset):,}")
    print(f"  - Total samples: {len(test_dataset):,}")
    print(f"  - Evaluation subset: {len(test_subset):,}")
    
    # Load your trained model
    print("\n🔍 Loading trained model...")
    model_path = "best_deepfake_model_robust.pth"
    
    if os.path.exists(model_path):
        print(f"✅ Found trained model: {model_path}")
        
        # Load the robust model (your trained one)
        model = RobustTwoBranchNet(n_classes=2).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"📈 Model loaded successfully!")
        print(f"   - Training accuracy: {checkpoint['best_val_acc']:.4f} ({checkpoint['best_val_acc']*100:.2f}%)")
        
    else:
        print(f"❌ Trained model not found: {model_path}")
        print("🔄 Creating and using untrained model for architecture test...")
        model = RobustTwoBranchNet(n_classes=2).to(device)
    
    # Evaluate model
    print(f"\n🧪 Evaluating model on {len(test_subset):,} samples...")
    results = evaluate_model_comprehensive(model, test_loader, device)
    
    # Print results
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    print(f"✅ Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"✅ Precision: {results['precision']:.4f}")
    print(f"✅ Recall:    {results['recall']:.4f}")
    print(f"✅ F1-Score:  {results['f1_score']:.4f}")
    print(f"✅ AUC Score: {results['auc_score']:.4f}")
    print("="*60)
    
    # Detailed classification report
    print("\n📋 Detailed Classification Report:")
    print(classification_report(results['labels'], results['predictions'], 
                              target_names=['Original', 'Deepfake'], zero_division=0))
    
    # Create plots
    print("\n📈 Generating visualization...")
    plot_results(results)
    
    # Success message
    if results['accuracy'] >= 0.90:
        print(f"\n🎯 EXCELLENT! Model achieved {results['accuracy']*100:.2f}% accuracy")
        print("🏆 Successfully converted notebook evaluation to local script!")
    else:
        print(f"\n📊 Model achieved {results['accuracy']*100:.2f}% accuracy")
    
    print(f"\n💾 Results saved as: notebook_evaluation_results.png")
    print("✅ Evaluation completed successfully!")

if __name__ == "__main__":
    main()