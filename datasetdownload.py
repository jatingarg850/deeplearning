import kagglehub
import shutil
import os

# Download latest version
path = kagglehub.dataset_download("mdhadiuzzaman/ff-data")
print("Downloaded to:", path)

# Create deeplearning folder in current workspace
workspace_path = "deeplearning"
if not os.path.exists(workspace_path):
    os.makedirs(workspace_path)

# Copy files from cache to workspace deeplearning folder
print(f"Copying files to {workspace_path}...")
for item in os.listdir(path):
    source = os.path.join(path, item)
    destination = os.path.join(workspace_path, item)
    if os.path.isdir(source):
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        shutil.copy2(source, destination)

print(f"Dataset files copied to: {os.path.abspath(workspace_path)}")