"""
Complete Training Script - Converted from paper-3-2-2.ipynb
Trains the SimpleTwoBranchNet model locally on your RTX 4060
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
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
import time
import copy
from tqdm import tqdm

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Custom FF++ dataset (from notebook)
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

# SimpleTwoBranchNet model (from notebook)
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

# TorchVision transforms (from notebook)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

def evaluate_model_detailed(model, val_loader, device):
    """Detailed evaluation with all metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for rgb, dwt, labels in tqdm(val_loader, desc="Evaluating", leave=False):
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

def plot_training_results(train_losses, train_accs, val_accs, final_results):
    """Plot comprehensive training results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Training Loss
    axes[0,0].plot(train_losses, 'b-', label='Training Loss')
    axes[0,0].set_title('Training Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Training & Validation Accuracy
    axes[0,1].plot(train_accs, 'b-', label='Training Accuracy')
    axes[0,1].plot(val_accs, 'r-', label='Validation Accuracy')
    axes[0,1].set_title('Training & Validation Accuracy')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Accuracy')
    axes[0,1].legend()
    axes[0,1].grid(True)
    axes[0,1].axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='90% Target')
    
    # Confusion Matrix
    sns.heatmap(final_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=axes[0,2])
    axes[0,2].set_title('Final Confusion Matrix')
    axes[0,2].set_xlabel('Predicted')
    axes[0,2].set_ylabel('Actual')
    axes[0,2].set_xticklabels(['Original', 'Deepfake'])
    axes[0,2].set_yticklabels(['Original', 'Deepfake'])
    
    # ROC Curve
    if len(np.unique(final_results['labels'])) > 1:
        fpr, tpr, _ = roc_curve(final_results['labels'], final_results['probabilities'][:, 1])
        axes[1,0].plot(fpr, tpr, color='darkorange', lw=2, 
                      label=f'ROC curve (AUC = {final_results["auc_score"]:.4f})')
        axes[1,0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1,0].set_xlim([0.0, 1.0])
        axes[1,0].set_ylim([0.0, 1.05])
        axes[1,0].set_xlabel('False Positive Rate')
        axes[1,0].set_ylabel('True Positive Rate')
        axes[1,0].set_title('ROC Curve')
        axes[1,0].legend(loc="lower right")
    
    # Metrics Bar Plot
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    metrics_values = [final_results['accuracy'], final_results['precision'], 
                     final_results['recall'], final_results['f1_score'], final_results['auc_score']]
    bars = axes[1,1].bar(metrics_names, metrics_values, 
                        color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'])
    axes[1,1].set_title('Final Performance Metrics')
    axes[1,1].set_ylabel('Score')
    axes[1,1].set_ylim([0, 1])
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                      f'{value:.3f}', ha='center', va='bottom')
    
    # Prediction Distribution
    axes[1,2].hist(final_results['probabilities'][:, 1], bins=50, alpha=0.7, 
                  color='purple', edgecolor='black')
    axes[1,2].set_title('Prediction Probability Distribution')
    axes[1,2].set_xlabel('Deepfake Probability')
    axes[1,2].set_ylabel('Frequency')
    axes[1,2].axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
    axes[1,2].legend()
    
    plt.tight_layout()
    plt.savefig('notebook_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("🚀 Starting Notebook Training Script (Local)")
    print("="*60)
    
    # Dataset paths (local)
    deepfake_dir = "deeplearning/Deepfake/extracted_faces"
    original_dir = "deeplearning/Original/extracted_faces"
    
    # Check if directories exist
    if not os.path.exists(deepfake_dir) or not os.path.exists(original_dir):
        print("❌ Dataset directories not found!")
        print(f"Expected: {deepfake_dir}")
        print(f"Expected: {original_dir}")
        return
    
    # Create datasets (exactly like notebook)
    print("📂 Loading datasets...")
    deepfake_ds = CustomFaceDataset(deepfake_dir, 1, train_transform, train_transform)
    original_ds = CustomFaceDataset(original_dir, 0, train_transform, train_transform)
    
    # Combine datasets
    full_ds = ConcatDataset([deepfake_ds, original_ds])
    
    # Split 80% train, 20% val (exactly like notebook)
    train_size = int(0.8 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    # DataLoaders (optimized for RTX 4060)
    batch_size = 16  # Reduced for RTX 4060 stability
    num_workers = 0  # Windows compatibility
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    print(f"📊 Dataset Info:")
    print(f"  - Deepfake samples: {len(deepfake_ds):,}")
    print(f"  - Original samples: {len(original_ds):,}")
    print(f"  - Total samples: {len(full_ds):,}")
    print(f"  - Train dataset size: {len(train_ds):,}")
    print(f"  - Validation dataset size: {len(val_ds):,}")
    print(f"  - Batch size: {batch_size}")
    
    # Initialize model (exactly like notebook)
    print("\n🏗️ Initializing SimpleTwoBranchNet model...")
    model = SimpleTwoBranchNet(n_classes=2).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer, criterion, scheduler (exactly like notebook)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=2)
    
    # Training setup (from notebook)
    num_epochs = 10  # As in notebook
    best_val_acc = 0
    best_model_wts = model.state_dict()
    
    # Training history
    train_losses = []
    train_accs = []
    val_accs = []
    
    print(f"\n🎯 Starting training for {num_epochs} epochs...")
    print(f"Target: >90% accuracy")
    
    # Training loop (exactly like notebook)
    for epoch in range(num_epochs):
        start = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for rgb, dwt, labels in train_pbar:
            rgb = rgb.to(device)
            dwt = dwt.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(rgb, dwt)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            current_acc = correct / total
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase (exactly like notebook)
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for rgb, dwt, labels in tqdm(val_loader, desc="Validation", leave=False):
                rgb = rgb.to(device)
                dwt = dwt.to(device)
                labels = labels.to(device)
                
                outputs = model(rgb, dwt)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        val_accs.append(val_acc)
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_wts,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'train_losses': train_losses,
                'train_accs': train_accs,
                'val_accs': val_accs
            }, "notebook_best_model.pth")
            print(f"\n✅ Validation accuracy improved to {val_acc:.4f}. Saved best model.")
        
        duration = time.time() - start
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Time={duration:.1f}s")
        print("-" * 60)
    
    # Load best weights for final evaluation (like notebook)
    model.load_state_dict(best_model_wts)
    model.eval()
    
    print(f"\n🎉 Training completed!")
    print(f"🏆 Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    
    # Final comprehensive evaluation
    print("\n📊 Running final evaluation...")
    final_results = evaluate_model_detailed(model, val_loader, device)
    
    # Print final results
    print("\n" + "="*60)
    print("📈 FINAL TRAINING RESULTS")
    print("="*60)
    print(f"✅ Final Accuracy:  {final_results['accuracy']:.4f} ({final_results['accuracy']*100:.2f}%)")
    print(f"✅ Precision: {final_results['precision']:.4f}")
    print(f"✅ Recall:    {final_results['recall']:.4f}")
    print(f"✅ F1-Score:  {final_results['f1_score']:.4f}")
    print(f"✅ AUC Score: {final_results['auc_score']:.4f}")
    print("="*60)
    
    # Detailed classification report
    print("\n📋 Detailed Classification Report:")
    print(classification_report(final_results['labels'], final_results['predictions'], 
                              target_names=['Original', 'Deepfake'], zero_division=0))
    
    # Create comprehensive plots
    print("\n📈 Generating training visualization...")
    plot_training_results(train_losses, train_accs, val_accs, final_results)
    
    # Success message
    if final_results['accuracy'] >= 0.90:
        print(f"\n🎯 SUCCESS! Model achieved {final_results['accuracy']*100:.2f}% accuracy (Target: >90%)")
        print("🏆 Notebook training successfully completed locally!")
    else:
        print(f"\n📊 Model achieved {final_results['accuracy']*100:.2f}% accuracy")
        print("💡 Consider training for more epochs or adjusting hyperparameters")
    
    print(f"\n💾 Model saved as: notebook_best_model.pth")
    print(f"📊 Results saved as: notebook_training_results.png")
    print("✅ Training script completed successfully!")

if __name__ == "__main__":
    main()