"""
Script to execute Jupyter notebook and save results
"""
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import sys

def run_notebook(notebook_path):
    """Execute a notebook and save the results"""
    print(f"📂 Loading notebook: {notebook_path}")
    
    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    print("🚀 Executing notebook cells...")
    print("=" * 60)
    
    # Configure the executor
    ep = ExecutePreprocessor(
        timeout=3600,  # 1 hour timeout
        kernel_name='python3',
        allow_errors=False  # Stop on errors
    )
    
    try:
        # Execute the notebook
        ep.preprocess(nb, {'metadata': {'path': './'}})
        
        print("\n" + "=" * 60)
        print("✅ Notebook executed successfully!")
        
        # Save the executed notebook
        output_path = notebook_path.replace('.ipynb', '_executed.ipynb')
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        
        print(f"💾 Results saved to: {output_path}")
        print("\n🎉 All cells executed with outputs!")
        
    except Exception as e:
        print(f"\n❌ Error executing notebook: {str(e)}")
        
        # Save partial results
        output_path = notebook_path.replace('.ipynb', '_partial.ipynb')
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        
        print(f"💾 Partial results saved to: {output_path}")
        sys.exit(1)

if __name__ == "__main__":
    notebook_file = "anti_overfitting_training.ipynb"
    run_notebook(notebook_file)
