# GPU-Optimized Deepfake Detection - Robust Version
# Fixed NaN issues and optimized for >90% accuracy on RTX 4060

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
import warnings
warnings.filterwarnings('ignore')

# GPU device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    torch.backends.cudnn.benchmark = True

# Simplified but effective transforms
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.3),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(16, 32), hole_width_range=(16, 32), p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Optimized DWT feature extraction
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
class RobustFaceDataset(Dataset):
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
            
            # Apply transforms
            rgb_tensor = self.transform_rgb(image=img_np)['image']
            
            # DWT processing
            dwt_img = dwt_features(Image.fromarray(img_np))
            dwt_tensor = self.transform_dwt(image=np.array(dwt_img))['image']
            
            return rgb_tensor, dwt_tensor, self.label
        except Exception as e:
            # Return a dummy sample if there's an error
            print(f"Error loading {img_path}: {e}")
            dummy_tensor = torch.zeros(3, 224, 224)
            return dummy_tensor, dummy_tensor, self.label

# Robust model architecture
class RobustTwoBranchNet(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        # RGB branch
        self.rgb_backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.rgb_backbone.classifier = nn.Identity()
        
        # DWT branch  
        self.dwt_backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.dwt_backbone.classifier = nn.Identity()
        
        # Robust classifier with proper initialization
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
        
        # Initialize classifier weights
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, rgb, dwt):
        # Extract features
        f1 = F.adaptive_avg_pool2d(self.rgb_backbone.features(rgb), 1).flatten(1)
        f2 = F.adaptive_avg_pool2d(self.dwt_backbone.features(dwt), 1).flatten(1)
        
        # Concatenate and classify
        x = torch.cat([f1, f2], dim=1)
        return self.classifier(x)

# Safe evaluation function
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
            
            # Check for NaN
            if torch.isnan(loss):
                continue
                
            total_loss += loss.item()
            
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    all_probs = np.array(all_probs)
    
    return all_preds, all_labels, all_probs, avg_loss

# Safe metrics calculation
def calculate_and_plot_metrics(y_true, y_pred, y_probs, epoch=None, save_plots=True):
    try:
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Check for valid probabilities
        if len(y_probs) > 0 and not np.any(np.isnan(y_probs)):
            auc = roc_auc_score(y_true, y_probs[:, 1])
            entropy = log_loss(y_true, y_probs)
        else:
            auc = 0.5
            entropy = 1.0
        
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
        
        if save_plots and len(y_probs) > 0:
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
            if not np.any(np.isnan(y_probs)):
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
            if not np.any(np.isnan(y_probs)):
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
    except Exception as e:
        print(f"Error in metrics calculation: {e}")
        return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0, 'auc_score': 0.5, 'entropy': 1.0}

def main():
    # Dataset paths
    deepfake_dir = "deeplearning/Deepfake/extracted_faces"
    original_dir = "deeplearning/Original/extracted_faces"

    # Build datasets
    deepfake_dataset = RobustFaceDataset(deepfake_dir, label=1, transform_rgb=train_transform, transform_dwt=train_transform)
    original_dataset = RobustFaceDataset(original_dir, label=0, transform_rgb=train_transform, transform_dwt=train_transform)

    # Combine and split datasets
    from torch.utils.data import random_split
    all_dataset = ConcatDataset([deepfake_dataset, original_dataset])
    train_size = int(0.8 * len(all_dataset))
    val_size = len(all_dataset) - train_size
    train_dataset, val_dataset = random_split(all_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    print(f"Dataset sizes - Total: {len(all_dataset)}, Train: {train_size}, Val: {val_size}")
    print(f"Deepfake samples: {len(deepfake_dataset)}, Original samples: {len(original_dataset)}")

    # Optimized DataLoaders - Windows compatible
    batch_size = 16  # Conservative for stability
    num_workers = 0  # Avoid multiprocessing issues on Windows
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )

    # Initialize model
    model = RobustTwoBranchNet(n_classes=2).to(device)
    
    # Loss and optimizer - conservative settings
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1e-4,  # Conservative learning rate
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler
    steps_per_epoch = len(train_loader)
    print(f"Steps per epoch: {steps_per_epoch}")
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.8, 
        patience=3,
        min_lr=1e-6
    )

    # Training setup
    num_epochs = 5
    best_val_acc = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    print(f"\n🚀 Starting ROBUST training for RTX 4060!")
    print(f"Target: >90% accuracy")
    print(f"Batch size: {batch_size}, Epochs: {num_epochs}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

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
            
            # Check for NaN
            if torch.isnan(loss):
                print("NaN loss detected, skipping batch")
                continue
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
        
        train_loss = running_loss / total if total > 0 else 0
        train_acc = correct / total if total > 0 else 0
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        val_preds, val_labels, val_probs, val_loss = evaluate_model(model, val_loader, criterion, device)
        val_acc = accuracy_score(val_labels, val_preds) if len(val_labels) > 0 else 0
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Step scheduler
        scheduler.step(val_acc)
        
        epoch_time = time.time() - start_time
        
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Time: {epoch_time:.2f}s")
        
        # Calculate detailed metrics every 3 epochs or if new best
        if (epoch + 1) % 3 == 0 or val_acc > best_val_acc:
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
                'best_val_acc': best_val_acc,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs
            }, "best_deepfake_model_robust.pth")
            print(f"✅ New best model saved! Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
        
        print("-" * 60)

    # Final evaluation
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
    plt.plot(val_accs, color='green', marker='o')
    plt.title('Validation Accuracy Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.axhline(y=0.9, color='red', linestyle='--', label='90% Target')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history_robust.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Classification report
    if len(final_labels) > 0:
        print("\nDetailed Classification Report:")
        print(classification_report(final_labels, final_preds, target_names=['Original', 'Deepfake'], zero_division=0))

    # Success message
    if best_val_acc >= 0.90:
        print(f"\n🎯 SUCCESS! Achieved target accuracy of >90%: {best_val_acc*100:.2f}%")
    else:
        print(f"\n⚠️  Current best: {best_val_acc*100:.2f}%")
        if best_val_acc < 0.6:
            print("   Model may need more training or different architecture")
        else:
            print("   Close to target! Consider running more epochs")

if __name__ == '__main__':
    main()