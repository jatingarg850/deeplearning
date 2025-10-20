"""
Anti-Overfitting Deepfake Detection Training
Designed to prevent overfitting and achieve realistic, generalizable performance
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
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
import copy
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Anti-overfitting transforms with heavy augmentation
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.4),
    A.Rotate(limit=20, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.6),
    A.GaussianBlur(blur_limit=(3, 7), p=0.4),
    A.MotionBlur(blur_limit=5, p=0.3),
    A.MedianBlur(blur_limit=3, p=0.2),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
    A.CoarseDropout(num_holes_range=(3, 8), hole_height_range=(8, 32), hole_width_range=(8, 32), p=0.5),
    A.GridDropout(ratio=0.3, p=0.3),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    A.CLAHE(clip_limit=2.0, p=0.3),
    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
    A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.2),
    A.OneOf([
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.4),
    ], p=0.4),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

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
class AntiOverfitDataset(Dataset):
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
            
            # Apply different augmentations to RGB and DWT
            rgb_tensor = self.transform_rgb(image=img_np)['image']
            
            # Apply different augmentation to DWT to prevent overfitting
            dwt_img = dwt_features(Image.fromarray(img_np))
            dwt_tensor = self.transform_dwt(image=np.array(dwt_img))['image']
            
            return rgb_tensor, dwt_tensor, self.label
        except Exception as e:
            # Return a dummy sample if there's an error
            dummy_tensor = torch.zeros(3, 224, 224)
            return dummy_tensor, dummy_tensor, self.label

# Anti-overfitting model with regularization
class AntiOverfitTwoBranchNet(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        # RGB branch with frozen early layers
        self.rgb_backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.rgb_backbone.classifier = nn.Identity()
        
        # Freeze early layers to prevent overfitting
        for param in list(self.rgb_backbone.parameters())[:50]:
            param.requires_grad = False
        
        # DWT branch with frozen early layers
        self.dwt_backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.dwt_backbone.classifier = nn.Identity()
        
        # Freeze early layers
        for param in list(self.dwt_backbone.parameters())[:50]:
            param.requires_grad = False
        
        # Regularized classifier with heavy dropout
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  # Input dropout
            nn.Linear(1280*2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),  # Heavy dropout
            
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
        
        # Initialize weights with smaller values
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)  # Smaller initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, rgb, dwt):
        # Extract features
        f1 = F.adaptive_avg_pool2d(self.rgb_backbone.features(rgb), 1).flatten(1)
        f2 = F.adaptive_avg_pool2d(self.dwt_backbone.features(dwt), 1).flatten(1)
        
        # Add noise during training to prevent overfitting
        if self.training:
            f1 = f1 + torch.randn_like(f1) * 0.01
            f2 = f2 + torch.randn_like(f2) * 0.01
        
        # Concatenate and classify
        x = torch.cat([f1, f2], dim=1)
        return self.classifier(x)

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

# Evaluation function
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0
    
    with torch.no_grad():
        for rgb, dwt, labels in tqdm(data_loader, desc="Evaluating", leave=False):
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

# Comprehensive metrics calculation
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
def plot_comprehensive_results(train_losses, val_losses, train_accs, val_accs, final_metrics, y_true, y_pred, y_probs):
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    # Training Loss
    axes[0,0].plot(train_losses, 'b-', label='Training Loss', linewidth=2)
    axes[0,0].plot(val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0,0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Training Accuracy
    axes[0,1].plot(train_accs, 'b-', label='Training Accuracy', linewidth=2)
    axes[0,1].plot(val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    axes[0,1].axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='90% Target')
    axes[0,1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Accuracy')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Overfitting Analysis
    gap = [abs(t - v) for t, v in zip(train_accs, val_accs)]
    axes[0,2].plot(gap, 'purple', linewidth=2, label='Train-Val Gap')
    axes[0,2].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Overfitting Threshold')
    axes[0,2].set_title('Overfitting Analysis (Train-Val Gap)', fontsize=14, fontweight='bold')
    axes[0,2].set_xlabel('Epoch')
    axes[0,2].set_ylabel('Accuracy Gap')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # Confusion Matrix
    sns.heatmap(final_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=axes[1,0])
    axes[1,0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Predicted')
    axes[1,0].set_ylabel('Actual')
    axes[1,0].set_xticklabels(['Original', 'Deepfake'])
    axes[1,0].set_yticklabels(['Original', 'Deepfake'])
    
    # ROC Curve
    if len(np.unique(y_true)) > 1 and not np.any(np.isnan(y_probs)):
        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
        axes[1,1].plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {final_metrics["auc_score"]:.4f})')
        axes[1,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1,1].set_xlim([0.0, 1.0])
        axes[1,1].set_ylim([0.0, 1.05])
        axes[1,1].set_xlabel('False Positive Rate')
        axes[1,1].set_ylabel('True Positive Rate')
        axes[1,1].set_title('ROC Curve', fontsize=14, fontweight='bold')
        axes[1,1].legend(loc="lower right")
        axes[1,1].grid(True, alpha=0.3)
    
    # Metrics Bar Plot
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    metrics_values = [final_metrics['accuracy'], final_metrics['precision'], 
                     final_metrics['recall'], final_metrics['f1_score'], final_metrics['auc_score']]
    bars = axes[1,2].bar(metrics_names, metrics_values, 
                        color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'])
    axes[1,2].set_title('Performance Metrics', fontsize=14, fontweight='bold')
    axes[1,2].set_ylabel('Score')
    axes[1,2].set_ylim([0, 1])
    axes[1,2].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        axes[1,2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                      f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Prediction Distribution
    if not np.any(np.isnan(y_probs)):
        axes[2,0].hist(y_probs[:, 1], bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[2,0].set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
        axes[2,0].set_xlabel('Deepfake Probability')
        axes[2,0].set_ylabel('Frequency')
        axes[2,0].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
        axes[2,0].legend()
        axes[2,0].grid(True, alpha=0.3)
    
    # Learning Rate vs Epoch (if available)
    axes[2,1].text(0.5, 0.5, f'Final Results:\n\nAccuracy: {final_metrics["accuracy"]:.4f}\nPrecision: {final_metrics["precision"]:.4f}\nRecall: {final_metrics["recall"]:.4f}\nF1-Score: {final_metrics["f1_score"]:.4f}\nAUC: {final_metrics["auc_score"]:.4f}\nEntropy: {final_metrics["entropy"]:.4f}', 
                  transform=axes[2,1].transAxes, fontsize=12, verticalalignment='center', 
                  horizontalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    axes[2,1].set_title('Final Metrics Summary', fontsize=14, fontweight='bold')
    axes[2,1].axis('off')
    
    # Model Complexity Analysis
    overfitting_score = np.mean(gap[-5:]) if len(gap) >= 5 else np.mean(gap)
    generalization_text = "Good Generalization" if overfitting_score < 0.05 else "Potential Overfitting"
    color = "green" if overfitting_score < 0.05 else "red"
    
    axes[2,2].text(0.5, 0.5, f'Generalization Analysis:\n\nTrain-Val Gap: {overfitting_score:.4f}\nStatus: {generalization_text}\n\nRecommendation:\n{"✅ Model generalizes well" if overfitting_score < 0.05 else "⚠️ Consider more regularization"}', 
                  transform=axes[2,2].transAxes, fontsize=12, verticalalignment='center', 
                  horizontalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
    axes[2,2].set_title('Generalization Assessment', fontsize=14, fontweight='bold')
    axes[2,2].axis('off')
    
    plt.tight_layout()
    plt.savefig('anti_overfitting_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("🚀 Anti-Overfitting Deepfake Detection Training")
    print("="*60)
    print("🎯 Goal: Train a model that generalizes well (prevents overfitting)")
    
    # Dataset paths
    deepfake_dir = "deeplearning/Deepfake/extracted_faces"
    original_dir = "deeplearning/Original/extracted_faces"
    
    if not os.path.exists(deepfake_dir) or not os.path.exists(original_dir):
        print("❌ Dataset directories not found!")
        return
    
    # Create datasets with heavy augmentation
    print("📂 Loading datasets with anti-overfitting augmentation...")
    deepfake_dataset = AntiOverfitDataset(deepfake_dir, label=1, transform_rgb=train_transform, transform_dwt=val_transform)
    original_dataset = AntiOverfitDataset(original_dir, label=0, transform_rgb=train_transform, transform_dwt=val_transform)
    
    # Combine and split
    from torch.utils.data import random_split
    all_dataset = ConcatDataset([deepfake_dataset, original_dataset])
    train_size = int(0.7 * len(all_dataset))  # Smaller training set
    val_size = int(0.2 * len(all_dataset))
    test_size = len(all_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        all_dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"📊 Dataset splits (to prevent overfitting):")
    print(f"  - Training: {len(train_dataset):,} (70%)")
    print(f"  - Validation: {len(val_dataset):,} (20%)")
    print(f"  - Test: {len(test_dataset):,} (10%)")
    
    # DataLoaders with smaller batch size
    batch_size = 12  # Smaller batch for better generalization
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # Initialize anti-overfitting model
    print("\n🏗️ Initializing Anti-Overfitting Model...")
    model = AntiOverfitTwoBranchNet(n_classes=2).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Anti-overfitting training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)  # Higher label smoothing
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=5e-5,  # Lower learning rate
        weight_decay=1e-3,  # Higher weight decay
        betas=(0.9, 0.999)
    )
    
    # Scheduler with more aggressive decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=7, min_delta=0.001)
    
    # Training setup
    num_epochs = 25  # More epochs with early stopping
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    
    # Training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print(f"\n🎯 Starting anti-overfitting training...")
    print(f"Max epochs: {num_epochs} (with early stopping)")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for rgb, dwt, labels in train_pbar:
            rgb, dwt, labels = rgb.to(device), dwt.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(rgb, dwt)
            loss = criterion(outputs, labels)
            
            # L2 regularization
            l2_reg = torch.tensor(0., device=device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += 1e-5 * l2_reg
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
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
        
        # Validation phase
        val_preds, val_labels, val_probs, val_loss = evaluate_model(model, val_loader, criterion, device)
        val_acc = accuracy_score(val_labels, val_preds)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        scheduler.step(val_loss)
        
        epoch_time = time.time() - start_time
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  Train-Val Gap: {abs(train_acc - val_acc):.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Save best model based on validation loss (not accuracy)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_wts,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs
            }, "anti_overfitting_model.pth")
            print(f"  ✅ Best model saved (Val Loss: {val_loss:.4f})")
        
        # Early stopping check
        if early_stopping(val_loss):
            print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
            break
        
        print("-" * 60)
    
    # Load best model and final evaluation
    model.load_state_dict(best_model_wts)
    model.eval()
    
    print(f"\n🎉 Training completed!")
    print(f"🏆 Best validation loss: {best_val_loss:.4f}")
    
    # Final evaluation on test set
    print("\n📊 Final evaluation on test set...")
    test_preds, test_labels, test_probs, test_loss = evaluate_model(model, test_loader, criterion, device)
    final_metrics = calculate_metrics(test_labels, test_preds, test_probs)
    
    print("\n" + "="*60)
    print("📈 FINAL ANTI-OVERFITTING RESULTS")
    print("="*60)
    print(f"✅ Test Accuracy:  {final_metrics['accuracy']:.4f} ({final_metrics['accuracy']*100:.2f}%)")
    print(f"✅ Precision: {final_metrics['precision']:.4f}")
    print(f"✅ Recall:    {final_metrics['recall']:.4f}")
    print(f"✅ F1-Score:  {final_metrics['f1_score']:.4f}")
    print(f"✅ AUC Score: {final_metrics['auc_score']:.4f}")
    print(f"✅ Entropy:   {final_metrics['entropy']:.4f}")
    
    # Overfitting analysis
    final_gap = abs(train_accs[-1] - val_accs[-1])
    print(f"\n🔍 Overfitting Analysis:")
    print(f"  Final Train-Val Gap: {final_gap:.4f}")
    if final_gap < 0.05:
        print("  ✅ Good generalization - No significant overfitting")
    elif final_gap < 0.1:
        print("  ⚠️  Mild overfitting detected")
    else:
        print("  ❌ Significant overfitting detected")
    
    print("="*60)
    
    # Classification report
    print("\n📋 Detailed Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=['Original', 'Deepfake'], zero_division=0))
    
    # Create comprehensive plots
    print("\n📈 Generating comprehensive visualization...")
    plot_comprehensive_results(train_losses, val_losses, train_accs, val_accs, 
                             final_metrics, test_labels, test_preds, test_probs)
    
    # Success message
    print(f"\n💾 Model saved as: anti_overfitting_model.pth")
    print(f"📊 Results saved as: anti_overfitting_training_results.png")
    
    if final_metrics['accuracy'] >= 0.85 and final_gap < 0.05:
        print(f"\n🎯 SUCCESS! Achieved {final_metrics['accuracy']*100:.2f}% accuracy with good generalization!")
        print("🏆 Model trained successfully without overfitting!")
    elif final_metrics['accuracy'] >= 0.80:
        print(f"\n📊 Good performance: {final_metrics['accuracy']*100:.2f}% accuracy")
        print("✅ Model shows reasonable generalization")
    else:
        print(f"\n📊 Model achieved {final_metrics['accuracy']*100:.2f}% accuracy")
        print("💡 Consider adjusting hyperparameters for better performance")
    
    print("✅ Anti-overfitting training completed!")

if __name__ == "__main__":
    main()