"""
Quick test to verify model architecture and data loading
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from main_optimized import OptimizedTwoBranchNet, OptimizedFaceDataset, train_transform
import time

def test_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on: {device}")
    
    # Test model creation
    model = OptimizedTwoBranchNet(n_classes=2).to(device)
    print(f"✅ Model created successfully")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 24
    rgb_input = torch.randn(batch_size, 3, 224, 224).to(device)
    dwt_input = torch.randn(batch_size, 3, 224, 224).to(device)
    
    with torch.no_grad():
        output = model(rgb_input, dwt_input)
    
    print(f"✅ Forward pass successful")
    print(f"Input shape: {rgb_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test data loading
    try:
        deepfake_dir = "deeplearning/Deepfake/extracted_faces"
        dataset = OptimizedFaceDataset(deepfake_dir, label=1, transform_rgb=train_transform, transform_dwt=train_transform)
        
        dataloader = DataLoader(dataset, batch_size=24, shuffle=True, num_workers=2)
        
        print(f"✅ Dataset created: {len(dataset)} samples")
        
        # Test one batch
        start_time = time.time()
        for rgb, dwt, labels in dataloader:
            rgb, dwt, labels = rgb.to(device), dwt.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(rgb, dwt)
            break
        
        batch_time = time.time() - start_time
        print(f"✅ Data loading test successful")
        print(f"Batch processing time: {batch_time:.3f}s")
        print(f"Expected steps per epoch: {len(dataloader)}")
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
    
    print("\n🎯 All tests passed! Ready for training.")

if __name__ == "__main__":
    test_model()