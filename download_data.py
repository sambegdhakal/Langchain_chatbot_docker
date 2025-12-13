import kagglehub
import shutil
import os

def download():
    # Download latest version
    path = kagglehub.dataset_download("excel4soccer/espn-soccer-data", force_download=True)

    DEST_DIR = r"C:\Users\Sambeg\OneDrive\Desktop\lang_chain_project\datasets"

    #removing existing directory
    if os.path.exists(DEST_DIR):
        shutil.rmtree(DEST_DIR)

    shutil.copytree(path, DEST_DIR)

    print("Path to dataset files:", path)