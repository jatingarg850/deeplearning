# GPU-Optimized Deepfake Detection with Comprehensive Metrics
# Optimized for RTX 4060 with 100% GPU utilization

import os
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
import copy
import time
from tqdm import tqdm

# GPU device setup with optimization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    # Enable optimizations for RTX 4060
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# Enhanced transforms for better accuracy
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.3),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(16, 32), hole_width_range=(16, 32), p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.HueSaturationValue(p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # ImageNet normalization
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# 5. DWT feature extraction
def dwt_features(img, wavelet='db1'):
    img = np.array(img)
    dwt_channels = []
    for ch in range(3):
        cA, (cH, cV, cD) = pywt.dwt2(img[..., ch], wavelet)
        cA_resized = Image.fromarray(cA).resize(img.shape[:2][::-1], Image.BILINEAR)
        dwt_channels.append(np.array(cA_resized))
    dwt_img = np.stack(dwt_channels, axis=2)
    return Image.fromarray(dwt_img.astype(np.uint8))

# 6. Custom Dataset for flat folder with label
class CustomFaceDataset(Dataset):
    def __init__(self, root_dir, label, transform_rgb, transform_dwt):
        self.root_dir = root_dir
        self.transform_rgb = transform_rgb
        self.transform_dwt = transform_dwt
        self.images = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png', '.jpeg', '.jpg.png'))]
        self.label = label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        rgb_tensor = self.transform_rgb(image=img_np)['image']
        dwt_img = dwt_features(img)
        dwt_tensor = self.transform_dwt(image=np.array(dwt_img))['image']
        return rgb_tensor, dwt_tensor, self.label

# Enhanced model definition for better accuracy
class EnhancedTwoBranchNet(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        # RGB branch
        self.rgb_backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.rgb_backbone.classifier = nn.Identity()
        
        # DWT branch
        self.dwt_backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.dwt_backbone.classifier = nn.Identity()
        
        # Enhanced classifier with attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(1280*2, 512),
            nn.ReLU(),
            nn.Linear(512, 1280*2),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(1280*2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes)
        )
        
    def forward(self, rgb, dwt):
        # Extract features
        f1 = F.adaptive_avg_pool2d(self.rgb_backbone.features(rgb), 1).view(rgb.size(0), -1)
        f2 = F.adaptive_avg_pool2d(self.dwt_backbone.features(dwt), 1).view(dwt.size(0), -1)
        
        # Concatenate features
        x = torch.cat([f1, f2], dim=1)
        
        # Apply attention
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # Classification
        return self.classifier(x)



# Comprehensive evaluation function
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0
    
    with torch.no_grad():
        for rgb, dwt, labels in tqdm(data_loader, desc="Evaluating"):
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

# Metrics calculation and plotting function
def calculate_and_plot_metrics(y_true, y_pred, y_probs, epoch=None, save_plots=True):
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # AUC Score
    auc = roc_auc_score(y_true, y_probs[:, 1])
    
    # Cross-entropy loss (entropy)
    entropy = log_loss(y_true, y_probs)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n{'='*50}")
    print(f"COMPREHENSIVE METRICS REPORT")
    if epoch is not None:
        print(f"Epoch: {epoch}")
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC Score: {auc:.4f}")
    print(f"Entropy:   {entropy:.4f}")
    print(f"{'='*50}")
    
    if save_plots:
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('Actual')
        axes[0,0].set_xticklabels(['Original', 'Deepfake'])
        axes[0,0].set_yticklabels(['Original', 'Deepfake'])
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
        axes[0,1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
        axes[0,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0,1].set_xlim([0.0, 1.0])
        axes[0,1].set_ylim([0.0, 1.05])
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC Curve')
        axes[0,1].legend(loc="lower right")
        
        # Metrics Bar Plot
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        metrics_values = [accuracy, precision, recall, f1, auc]
        bars = axes[1,0].bar(metrics_names, metrics_values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'])
        axes[1,0].set_title('Performance Metrics')
        axes[1,0].set_ylabel('Score')
        axes[1,0].set_ylim([0, 1])
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                          f'{value:.3f}', ha='center', va='bottom')
        
        # Prediction Distribution
        axes[1,1].hist(y_probs[:, 1], bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[1,1].set_title('Prediction Probability Distribution')
        axes[1,1].set_xlabel('Deepfake Probability')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
        axes[1,1].legend()
        
        plt.tight_layout()
        
        # Save plot
        if epoch is not None:
            plt.savefig(f'metrics_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig('final_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_score': auc,
        'entropy': entropy,
        'confusion_matrix': cm
    }

def main():
    # Dataset paths (corrected for Windows)
    deepfake_dir   = "deeplearning/Deepfake/extracted_faces"
    original_dir   = "deeplearning/Original/extracted_faces"

    # Build datasets with correct labels
    deepfake_dataset = CustomFaceDataset(deepfake_dir, label=1, transform_rgb=train_transform, transform_dwt=train_transform)
    original_dataset = CustomFaceDataset(original_dir, label=0, transform_rgb=train_transform, transform_dwt=train_transform)

    # Combine and split into train/val sets with GPU optimization
    from torch.utils.data import random_split
    all_dataset = ConcatDataset([deepfake_dataset, original_dataset])
    train_size = int(0.8 * len(all_dataset))
    val_size = len(all_dataset) - train_size
    train_dataset, val_dataset = random_split(all_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    print(f"Dataset sizes - Total: {len(all_dataset)}, Train: {train_size}, Val: {val_size}")
    print(f"Deepfake samples: {len(deepfake_dataset)}, Original samples: {len(original_dataset)}")

    # Optimized DataLoaders for RTX 4060 (8GB VRAM) - Windows compatible
    batch_size = 16  # Optimized for RTX 4060
    num_workers = 0  # Set to 0 for Windows compatibility
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True  # Faster GPU transfer
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )

    model = EnhancedTwoBranchNet(n_classes=2).to(device)

    # Enhanced loss and optimizer for better convergence
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=2e-4, 
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    # More aggressive learning rate scheduling
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=2e-4,
        epochs=5,  # Increased epochs for better accuracy
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )

    # Enhanced training loop with comprehensive metrics
    num_epochs = 5  # Increased for better accuracy
    best_val_acc = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    print("Starting training with GPU optimization...")
    print(f"Target: >90% accuracy")
    print(f"Batch size: {batch_size}, Epochs: {num_epochs}")

    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for rgb, dwt, labels in train_pbar:
            rgb, dwt, labels = rgb.to(device, non_blocking=True), dwt.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(rgb, dwt)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            current_acc = correct / total
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.4f}',
                'LR': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase with comprehensive metrics
        val_preds, val_labels, val_probs, val_loss = evaluate_model(model, val_loader, criterion, device)
        val_acc = accuracy_score(val_labels, val_preds)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        epoch_time = time.time() - start_time
        
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Time: {epoch_time:.2f}s")
        
        # Calculate detailed metrics every 5 epochs or if new best
        if (epoch + 1) % 5 == 0 or val_acc > best_val_acc:
            print(f"\nDetailed metrics for Epoch {epoch+1}:")
            calculate_and_plot_metrics(val_labels, val_preds, val_probs, epoch+1, save_plots=(val_acc > best_val_acc))
        
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
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs
            }, "best_deepfake_model.pth")
            print(f"✅ New best model saved! Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
        
        print("-" * 60)

    # Load best model and final evaluation
    model.load_state_dict(best_model_wts)
    model.eval()

    print(f"\n🎉 Training completed!")
    print(f"🏆 Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")

    # Final comprehensive evaluation
    print("\n" + "="*60)
    print("FINAL MODEL EVALUATION")
    print("="*60)

    final_preds, final_labels, final_probs, final_loss = evaluate_model(model, val_loader, criterion, device)
    final_metrics = calculate_and_plot_metrics(final_labels, final_preds, final_probs, save_plots=True)

    # Training history plot
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Accuracy', color='blue')
    plt.plot(val_accs, label='Val Accuracy', color='red')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot([scheduler.get_last_lr()[0] for _ in range(len(train_losses))], color='green')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(final_labels, final_preds, target_names=['Original', 'Deepfake']))

    # Success message
    if best_val_acc >= 0.90:
        print(f"\n🎯 SUCCESS! Achieved target accuracy of >90%: {best_val_acc*100:.2f}%")
    else:
        print(f"\n⚠️  Target not reached. Current best: {best_val_acc*100:.2f}%. Consider:")
        print("   - Increasing epochs")
        print("   - Adjusting learning rate")
        print("   - Adding more data augmentation")
        print("   - Fine-tuning hyperparameters")

if __name__ == '__main__':
    main()
