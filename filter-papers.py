import os
import glob
from ultralytics import YOLO

# --- Configuration ---
# 1. Source: The folder you want to clean
SOURCE_DIR = r"C:\Users\uaeor\Downloads\Duncan\experiment2_individual_images"

# 2. Model: Points to your trained 'fast' model
MODEL_PATH = r"C:\Users\uaeor\runs\classify\yolo_tidy_classifier_fast\weights\best.pt"

CONFIDENCE = 0.4
# ---------------------

def main():
    print(f"Loading Janitor Model: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print("Error: Model file not found. Check the path.")
        return

    model = YOLO(MODEL_PATH)

    # Find all images
    print(f"Scanning {SOURCE_DIR}...")
    image_files = glob.glob(os.path.join(SOURCE_DIR, "**", "*.jpeg"), recursive=True)
    image_files += glob.glob(os.path.join(SOURCE_DIR, "**", "*.png"), recursive=True)

    print(f"Found {len(image_files)} images. Cleaning...")
    
    removed_count = 0
    BATCH_SIZE = 32 

    for i in range(0, len(image_files), BATCH_SIZE):
        batch = image_files[i:i+BATCH_SIZE]
        
        # Run inference
        results = model(batch, verbose=False)

        for img_path, result in zip(batch, results):
            top_class = result.names[result.probs.top1]
            conf = result.probs.top1conf.item()

            # Logic: If Table/Formula ('target') -> PERMANENTLY DELETE
            if top_class == 'target' and conf > CONFIDENCE:
                try:
                    # [!!!] This line deletes the file
                    os.remove(img_path) 
                    
                    filename = os.path.basename(img_path)
                    print(f"[DELETED] {filename} ({conf:.2f})")
                    removed_count += 1
                except Exception as e:
                    print(f"Error deleting {img_path}: {e}")

    print(f"\n--- Cleanup Complete ---")
    print(f"Permanently deleted {removed_count} tables/formulas.")
    print(f"The folder '{SOURCE_DIR}' is now clean.")

if __name__ == "__main__":
    main()