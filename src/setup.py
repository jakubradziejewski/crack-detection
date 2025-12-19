import os
import shutil
import kagglehub

def setup_dataset():
    # Download source
    download_path = kagglehub.dataset_download("lakshaymiddha/crack-segmentation-dataset")
    true_source_dir = os.path.join(download_path, "crack_segmentation_dataset")
    
    local_data_dir = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(local_data_dir):
        os.makedirs(local_data_dir)

    # Move only folders with split data
    for folder in ['train', 'test']:
        source = os.path.join(true_source_dir, folder)
        destination = os.path.join(local_data_dir, folder)
        
        if os.path.exists(source) and not os.path.exists(destination):
            shutil.move(source, destination)
            print(f"Successfully moved: {folder}")

    # Clean up unmoved files
    try:
        shutil.rmtree(download_path)
        print("Kaggle cache cleared.")
    except Exception as e:
        print(f"Cache cleanup skipped: {e}")

if __name__ == "__main__":
    setup_dataset()