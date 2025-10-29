"""
Trust and run the Jupyter notebook
"""
import subprocess
import sys

print("🔧 Trusting the notebook...")
try:
    # Trust the notebook
    result = subprocess.run(
        ["python", "-m", "jupyter", "trust", "anti_overfitting_training.ipynb"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode == 0:
        print("✅ Notebook is now trusted!")
    else:
        print(f"⚠️ Warning: {result.stderr}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*60)
print("📝 The notebook is ready to run in Jupyter Lab!")
print("="*60)
print("\n🌐 Open this URL in your browser:")
print("http://localhost:8888/lab")
print("\nThen:")
print("1. Click on 'anti_overfitting_training.ipynb'")
print("2. Press Shift+Enter to run each cell")
print("3. Or click Run → Run All Cells")
print("\n✅ All cells should now execute without issues!")
