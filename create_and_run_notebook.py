"""
Create and execute anti-overfitting training notebook
"""
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
import sys

# Create a new notebook
nb = nbf.v4.new_notebook()

# Add cells
cells = []

# Cell 1: Title
cells.append(nbf.v4.new_markdown_cell("""# Anti-Overfitting Deepfake Detection Model
## Training and Validation Notebook

This notebook trains a two-branch neural network (RGB + DWT) with heavy regularization to prevent overfitting."""))

# Cell 2: Imports
cells.append(nbf.v4.new_code_cell("""import os
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

print("✅ All libraries imported successfully!")"""))

# Cell 3: GPU Setup
cells.append(nbf.v4.new_markdown_cell("### GPU Setup"))
cells.append(nbf.v4.new_code_cell("""device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠️ GPU not available, using CPU")"""))

# Cell 4: Transforms
cells.append(nbf.v4.new_markdown_cell("### Data Augmentation"))
cells.append(nbf.v4.new_code_cell("""# Heavy augmentation for training
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.7),
    A.GaussianBlur(p=0.4),
    A.GaussNoise(p=0.4),
    A.CoarseDropout(num_holes_range=(3, 8), hole_height_range=(8, 32), hole_width_range=(8, 32), p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

print("✅ Transforms configured")"""))

# Cell 5: DWT
cells.append(nbf.v4.new_markdown_cell("### DWT Feature Extraction"))
cells.append(nbf.v4.new_code_cell("""def dwt_features(img, wavelet='db1'):
    img = np.array(img)
    dwt_channels = []
    for ch in range(3):
        cA, (cH, cV, cD) = pywt.dwt2(img[..., ch], wavelet)
        cA_resized = np.array(Image.fromarray(cA).resize((224, 224), Image.BILINEAR))
        dwt_channels.append(cA_resized)
    dwt_img = np.stack(dwt_channels, axis=2)
    return Image.fromarray(dwt_img.astype(np.uint8))

print("✅ DWT function defined")"""))

# Cell 6: Dataset
cells.append(nbf.v4.new_markdown_cell("### Dataset Class"))
cells.append(nbf.v4.new_code_cell("""class AntiOverfitDataset(Dataset):
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
        except:
            dummy_tensor = torch.zeros(3, 224, 224)
            return dummy_tensor, dummy_tensor, self.label

print("✅ Dataset class defined")"""))

# Cell 7: Model
cells.append(nbf.v4.new_markdown_cell("### Model Architecture"))
cells.append(nbf.v4.new_code_cell("""class AntiOverfitTwoBranchNet(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.rgb_backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.rgb_backbone.classifier = nn.Identity()
        
        for param in list(self.rgb_backbone.parameters())[:50]:
            param.requires_grad = False
        
        self.dwt_backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.dwt_backbone.classifier = nn.Identity()
        
        for param in list(self.dwt_backbone.parameters())[:50]:
            param.requires_grad = False
        
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
        
        if self.training:
            f1 = f1 + torch.randn_like(f1) * 0.01
            f2 = f2 + torch.randn_like(f2) * 0.01
        
        x = torch.cat([f1, f2], dim=1)
        return self.classifier(x)

print("✅ Model architecture defined")"""))

# Cell 8: Helper functions
cells.append(nbf.v4.new_markdown_cell("### Helper Functions"))
cells.append(nbf.v4.new_code_cell("""class EarlyStopping:
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

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
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
    
    return all_preds, all_labels, np.array(all_probs), total_loss / len(data_loader)

def calculate_metrics(y_true, y_pred, y_probs):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_probs[:, 1])
    else:
        auc = 0.5
    
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy, 'precision': precision, 'recall': recall,
        'f1_score': f1, 'auc_score': auc, 'confusion_matrix': cm
    }

print("✅ Helper functions defined")"""))

# Cell 9: Load Data
cells.append(nbf.v4.new_markdown_cell("### Load Dataset"))
cells.append(nbf.v4.new_code_cell("""print("🚀 Loading datasets...")

deepfake_dir = "deeplearning/Deepfake/extracted_faces"
original_dir = "deeplearning/Original/extracted_faces"

deepfake_dataset = AntiOverfitDataset(deepfake_dir, label=1, transform_rgb=train_transform, transform_dwt=val_transform)
original_dataset = AntiOverfitDataset(original_dir, label=0, transform_rgb=train_transform, transform_dwt=val_transform)

all_dataset = ConcatDataset([deepfake_dataset, original_dataset])
train_size = int(0.7 * len(all_dataset))
val_size = int(0.2 * len(all_dataset))
test_size = len(all_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    all_dataset, [train_size, val_size, test_size], 
    generator=torch.Generator().manual_seed(42)
)

print(f"Training: {len(train_dataset):,} | Validation: {len(val_dataset):,} | Test: {len(test_dataset):,}")

batch_size = 12
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print("✅ DataLoaders ready")"""))

# Cell 10: Initialize
cells.append(nbf.v4.new_markdown_cell("### Initialize Training"))
cells.append(nbf.v4.new_code_cell("""model = AntiOverfitTwoBranchNet(n_classes=2).to(device)
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
early_stopping = EarlyStopping(patience=7)

num_epochs = 25
best_val_loss = float('inf')
best_model_wts = copy.deepcopy(model.state_dict())
train_losses, val_losses, train_accs, val_accs = [], [], [], []

print("✅ Ready to train!")"""))

# Cell 11: Training
cells.append(nbf.v4.new_markdown_cell("### Training Loop"))
cells.append(nbf.v4.new_code_cell("""print("🎯 Starting training...\\n")

for epoch in range(num_epochs):
    start_time = time.time()
    
    # Training
    model.train()
    running_loss, correct, total = 0, 0, 0
    
    for rgb, dwt, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        rgb, dwt, labels = rgb.to(device), dwt.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(rgb, dwt)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item() * labels.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    train_loss = running_loss / total
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # Validation
    val_preds, val_labels, val_probs, val_loss = evaluate_model(model, val_loader, criterion, device)
    val_acc = accuracy_score(val_labels, val_preds)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Acc={val_acc:.4f} | Gap={abs(train_acc-val_acc):.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save({'model_state_dict': best_model_wts, 'train_losses': train_losses, 
                   'val_losses': val_losses, 'train_accs': train_accs, 'val_accs': val_accs}, 
                   "anti_overfitting_model.pth")
        print(f"  ✅ Best model saved!")
    
    if early_stopping(val_loss):
        print(f"\\n🛑 Early stopping at epoch {epoch+1}")
        break

print("\\n🎉 Training completed!")"""))

# Cell 12: Evaluate
cells.append(nbf.v4.new_markdown_cell("### Final Evaluation"))
cells.append(nbf.v4.new_code_cell("""model.load_state_dict(best_model_wts)
model.eval()

test_preds, test_labels, test_probs, test_loss = evaluate_model(model, test_loader, criterion, device)
final_metrics = calculate_metrics(test_labels, test_preds, test_probs)

print("\\n" + "="*60)
print("📈 FINAL RESULTS")
print("="*60)
print(f"Accuracy:  {final_metrics['accuracy']:.4f} ({final_metrics['accuracy']*100:.2f}%)")
print(f"Precision: {final_metrics['precision']:.4f}")
print(f"Recall:    {final_metrics['recall']:.4f}")
print(f"F1-Score:  {final_metrics['f1_score']:.4f}")
print(f"AUC Score: {final_metrics['auc_score']:.4f}")

final_gap = abs(train_accs[-1] - val_accs[-1])
print(f"\\nTrain-Val Gap: {final_gap:.4f}")
if final_gap < 0.05:
    print("✅ Good generalization!")
else:
    print("⚠️ Some overfitting detected")

print("\\n" + classification_report(test_labels, test_preds, target_names=['Original', 'Deepfake']))"""))

# Cell 13: Visualizations
cells.append(nbf.v4.new_markdown_cell("### Visualizations"))
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Loss
axes[0,0].plot(train_losses, 'b-', label='Train', linewidth=2)
axes[0,0].plot(val_losses, 'r-', label='Val', linewidth=2)
axes[0,0].set_title('Loss', fontweight='bold')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Accuracy
axes[0,1].plot(train_accs, 'b-', label='Train', linewidth=2)
axes[0,1].plot(val_accs, 'r-', label='Val', linewidth=2)
axes[0,1].set_title('Accuracy', fontweight='bold')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Overfitting Gap
gap = [abs(t - v) for t, v in zip(train_accs, val_accs)]
axes[1,0].plot(gap, 'purple', linewidth=2)
axes[1,0].axhline(y=0.05, color='red', linestyle='--', label='Threshold')
axes[1,0].set_title('Train-Val Gap', fontweight='bold')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Confusion Matrix
sns.heatmap(final_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=axes[1,1])
axes[1,1].set_title('Confusion Matrix', fontweight='bold')
axes[1,1].set_xlabel('Predicted')
axes[1,1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Visualization saved as 'training_results.png'")"""))

# Cell 14: Summary
cells.append(nbf.v4.new_markdown_cell("### Summary"))
cells.append(nbf.v4.new_code_cell("""print("\\n" + "="*60)
print("🎯 TRAINING SUMMARY")
print("="*60)
print(f"Model saved: anti_overfitting_model.pth")
print(f"Visualization: training_results.png")
print(f"\\nFinal Accuracy: {final_metrics['accuracy']*100:.2f}%")
print(f"AUC Score: {final_metrics['auc_score']:.4f}")
print(f"Train-Val Gap: {final_gap:.4f}")

if final_metrics['accuracy'] >= 0.85 and final_gap < 0.05:
    print("\\n🎯 SUCCESS! Excellent performance with good generalization!")
elif final_metrics['accuracy'] >= 0.80:
    print("\\n✅ Good performance with reasonable generalization")
else:
    print("\\n📊 Model shows room for improvement")

print("\\n✅ Anti-overfitting training completed!")
print("="*60)"""))

# Add all cells to notebook
nb['cells'] = cells

# Save the notebook
notebook_path = 'anti_overfitting_training.ipynb'
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"✅ Notebook created: {notebook_path}")
print("\\n🚀 Now executing the notebook...")
print("="*60)

# Execute the notebook
ep = ExecutePreprocessor(timeout=3600, kernel_name='python3')

try:
    ep.preprocess(nb, {'metadata': {'path': './'}})
    
    # Save executed notebook
    output_path = 'anti_overfitting_training_executed.ipynb'
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    print("\\n" + "="*60)
    print(f"✅ SUCCESS! Notebook executed with all outputs!")
    print(f"💾 Results saved to: {output_path}")
    print("="*60)
    
except Exception as e:
    print(f"\\n❌ Error: {str(e)}")
    # Save partial results
    output_path = 'anti_overfitting_training_partial.ipynb'
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"💾 Partial results saved to: {output_path}")
    sys.exit(1)
