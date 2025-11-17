import os
from ultralytics import YOLO

# --- 1. Configuration ---
side_composites = r"C:\Users\uaeor\Downloads\Duncan\classifier_dataset"
up_down_composites = r"C:\Users\uaeor\Downloads\Duncan\classifier2_dataset"
individual_images =  r"C:\Users\uaeor\Downloads\Duncan\experiment2_individual_images"
DATA_DIR = side_composites

IMG_SIZE = 1024 #size of all images is benchmarked at 1024
BATCH_SIZE = 8       # How many images to process at once.
EPOCHS = 10         # How many times to show the model the full dataset.

def main():
    
    # --- 2. Load a Pretrained Classifier Model ---
    # We load 'yolov8n-cls.pt'. This is the smallest and fastest model, please change when you try on HBC
    model = YOLO("yolov8n-cls.pt") 

    # --- 3. Train the Model ---
    print("--- Starting Training ---")
    # We point the model to our data directory.
    # Ultralytics automatically understands the 'accept'/'reject'
    # folder structure and will handle the 80/20 train/validation split.
    results = model.train(
        data=DATA_DIR,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name='vila_accept_reject_classifier' # A name for the results folder
    )
    
    print("--- Training Complete ---")
    print("\n--- Validating Best Model ---")
    
    # 'results.save_dir' points to the new 'runs/classify/vila_accept_reject_classifier'
    # folder, which contains the 'best.pt' weights.
    best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
    
    # Load the model we just trained
    model = YOLO(best_model_path)
    
    # Run validation
    metrics = model.val()
    
    print(f"\nModel saved to: {results.save_dir}")
    print(f"Top-1 Accuracy: {metrics.top1:.4f}")

            
main()