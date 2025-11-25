import os
import shutil
import glob
import random

# --- Configuration ---
DATASET_DIR = r"C:\Users\uaeor\Downloads\Duncan\dataset_tidy"
VAL_SPLIT = 0.2  # Move 20% of images to validation
# ---------------------

def main():
    train_dir = os.path.join(DATASET_DIR, "train")
    val_dir = os.path.join(DATASET_DIR, "val")

    # Check if train exists
    if not os.path.exists(train_dir):
        print(f"Error: Could not find training folder at {train_dir}")
        return

    classes = ['target', 'other']

    print(f"Splitting dataset at {DATASET_DIR}...")

    for class_name in classes:
        # Define paths
        src_path = os.path.join(train_dir, class_name)
        dest_path = os.path.join(val_dir, class_name)

        # Create destination folder
        os.makedirs(dest_path, exist_ok=True)

        # Find all images
        images = glob.glob(os.path.join(src_path, "*.*"))
        
        # Calculate how many to move
        num_to_move = int(len(images) * VAL_SPLIT)
        
        if num_to_move == 0:
            print(f"  - Class '{class_name}': Not enough images to split.")
            continue

        # Select random images
        images_to_move = random.sample(images, num_to_move)

        print(f"  - Class '{class_name}': Moving {num_to_move} images to 'val'...")

        # Move them
        for img in images_to_move:
            try:
                shutil.move(img, os.path.join(dest_path, os.path.basename(img)))
            except Exception as e:
                print(f"Error moving {img}: {e}")

    print("\n--- Split Complete ---")
    print("You can now run your training script.")

if __name__ == "__main__":
    main()