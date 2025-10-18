"""
Final Model Evaluation Script
Load the trained model and generate comprehensive results
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
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
import os

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Transforms (same as training)
val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# DWT function
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
class EvalFaceDataset(Dataset):
    def __init__(self, root_dir, label, transform_rgb, transform_dwt):
        self.root_dir = root_dir
        self.transform_rgb = transform_rgb
        self.transform_dwt = transform_dwt
        self.images = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.label = label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            img_np = np.array(img)
        
        rgb_tensor = self.transform_rgb(image=img_np)['image']
        dwt_img = dwt_features(Image.fromarray(img_np))
        dwt_tensor = self.transform_dwt(image=np.array(dwt_img))['image']
        
        return rgb_tensor, dwt_tensor, self.label

# Model architecture (same as training)
class RobustTwoBranchNet(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.rgb_backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.rgb_backbone.classifier = nn.Identity()
        
        self.dwt_backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.dwt_backbone.classifier = nn.Identity()
        
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
        f1 = F.adaptive_avg_pool2d(self.rgb_backbone.features(rgb), 1).flatten(1)
        f2 = F.adaptive_avg_pool2d(self.dwt_backbone.features(dwt), 1).flatten(1)
        x = torch.cat([f1, f2], dim=1)
        return self.classifier(x)

def load_and_evaluate():
    print("🔍 Loading trained model...")
    
    # Load model
    model = RobustTwoBranchNet(n_classes=2).to(device)
    
    # Load checkpoint
    checkpoint = torch.load('best_deepfake_model_robust.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"📊 Best training accuracy: {checkpoint['best_val_acc']:.4f} ({checkpoint['best_val_acc']*100:.2f}%)")
    
    # Load test data
    deepfake_dir = "deeplearning/Deepfake/extracted_faces"
    original_dir = "deeplearning/Original/extracted_faces"
    
    deepfake_dataset = EvalFaceDataset(deepfake_dir, label=1, transform_rgb=val_transform, transform_dwt=val_transform)
    original_dataset = EvalFaceDataset(original_dir, label=0, transform_rgb=val_transform, transform_dwt=val_transform)
    
    # Create test set (use a subset for faster evaluation)
    from torch.utils.data import random_split, Subset
    all_dataset = ConcatDataset([deepfake_dataset, original_dataset])
    
    # Use 10% of data for final evaluation (still ~65k samples)
    test_size = min(65000, len(all_dataset) // 10)
    test_indices = torch.randperm(len(all_dataset))[:test_size]
    test_dataset = Subset(all_dataset, test_indices)
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"📈 Evaluating on {len(test_dataset)} samples...")
    
    # Evaluate
    all_preds = []
    all_labels = []
    all_probs = []
    
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
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    auc = roc_auc_score(all_labels, all_probs[:, 1])
    entropy = log_loss(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Print results
    print("\n" + "="*60)
    print("🏆 FINAL MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"✅ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall:    {recall:.4f}")
    print(f"✅ F1-Score:  {f1:.4f}")
    print(f"✅ AUC Score: {auc:.4f}")
    print(f"✅ Entropy:   {entropy:.4f}")
    print("="*60)
    
    # Create final visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
    axes[0,0].set_title('Final Confusion Matrix')
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('Actual')
    axes[0,0].set_xticklabels(['Original', 'Deepfake'])
    axes[0,0].set_yticklabels(['Original', 'Deepfake'])
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
    axes[0,1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    axes[0,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0,1].set_xlim([0.0, 1.0])
    axes[0,1].set_ylim([0.0, 1.05])
    axes[0,1].set_xlabel('False Positive Rate')
    axes[0,1].set_ylabel('True Positive Rate')
    axes[0,1].set_title('Final ROC Curve')
    axes[0,1].legend(loc="lower right")
    
    # Metrics Bar Plot
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    metrics_values = [accuracy, precision, recall, f1, auc]
    bars = axes[1,0].bar(metrics_names, metrics_values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'])
    axes[1,0].set_title('Final Performance Metrics')
    axes[1,0].set_ylabel('Score')
    axes[1,0].set_ylim([0, 1])
    
    for bar, value in zip(bars, metrics_values):
        axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                      f'{value:.3f}', ha='center', va='bottom')
    
    # Prediction Distribution
    axes[1,1].hist(all_probs[:, 1], bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[1,1].set_title('Final Prediction Distribution')
    axes[1,1].set_xlabel('Deepfake Probability')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('FINAL_MODEL_RESULTS.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Classification report
    print("\n📋 Detailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Original', 'Deepfake']))
    
    # Success message
    if accuracy >= 0.90:
        print(f"\n🎯 SUCCESS! Model achieved {accuracy*100:.2f}% accuracy (Target: >90%)")
        print(f"🚀 GPU Utilization: Optimized for RTX 4060")
        print(f"💾 Model saved as: best_deepfake_model_robust.pth")
        print(f"📊 Results saved as: FINAL_MODEL_RESULTS.png")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_score': auc,
        'entropy': entropy
    }

if __name__ == "__main__":
    results = load_and_evaluate()
    print(f"\n✅ Evaluation completed successfully!")