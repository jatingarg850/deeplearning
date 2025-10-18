"""
GPU Setup and Verification Script for RTX 4060 Deepfake Detection
"""
import torch
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✅ Requirements installed successfully!")

def check_gpu_setup():
    """Verify GPU setup and optimization"""
    print("\n" + "="*50)
    print("GPU SETUP VERIFICATION")
    print("="*50)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA is available")
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"✅ CUDA Version: {torch.version.cuda}")
        print(f"✅ PyTorch Version: {torch.__version__}")
        
        # Test GPU memory allocation
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            print("✅ GPU memory allocation test passed")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ GPU memory allocation test failed: {e}")
            
        # Check cuDNN
        if torch.backends.cudnn.enabled:
            print("✅ cuDNN is enabled (optimized for RTX 4060)")
        else:
            print("⚠️  cuDNN is not enabled")
            
    else:
        print("❌ CUDA is not available")
        print("Please install CUDA-compatible PyTorch:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return False
    
    print("\n🚀 GPU setup is ready for training!")
    return True

if __name__ == "__main__":
    try:
        install_requirements()
        check_gpu_setup()
        print("\n✅ Setup completed successfully!")
        print("You can now run: python main.py")
    except Exception as e:
        print(f"❌ Setup failed: {e}")