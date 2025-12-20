import os
import shutil
import kagglehub
from config import CONFIG

def setup_dataset():
    # 1. Use the directory from your central config
    local_data_dir = CONFIG["root_dir"] 
    
    # Download source from Kaggle
    print("Downloading dataset from Kaggle...")
    download_path = kagglehub.dataset_download("lakshaymiddha/crack-segmentation-dataset")
    true_source_dir = os.path.join(download_path, "crack_segmentation_dataset")
    
    # Create the directory if it doesn't exist
    if not os.path.exists(local_data_dir):
        os.makedirs(local_data_dir)
        print(f"Created directory: {local_data_dir}")

    # 2. Move folders to the location specified in CONFIG
    for folder in ['train', 'test']:
        source = os.path.join(true_source_dir, folder)
        destination = os.path.join(local_data_dir, folder)
        
        if os.path.exists(source) and not os.path.exists(destination):
            shutil.move(source, destination)
            print(f"Successfully moved: {folder} to {destination}")
        else:
            print(f"Folder {folder} already exists in destination, skipping move.")

    # Clean up Kaggle cache
    try:
        shutil.rmtree(download_path)
        print("Kaggle cache cleared.")
    except Exception as e:
        print(f"Cache cleanup skipped: {e}")

if __name__ == "__main__":
    setup_dataset()