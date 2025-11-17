import pandas as pd
import os
import shutil
import glob
import zipfile
import io
from PIL import Image
from sklearn.model_selection import train_test_split

# --- 1. Configuration ---
CANVAS_SIZE = (1024, 1024)
CANVAS_BG_COLOR = (255, 255, 255) # White
OUTPUT_FORMAT = "JPEG"
OUTPUT_QUALITY = 90

DECISION_MAP = {
    1: 'accept', '1': 'accept',
    0: 'reject', '0': 'reject'
}

VALIDATION_SPLIT_SIZE = 0.20

# --- 2. Paths ---
DECISION_FILE_PATH = r"X:\preiss_group\Shared\openreview\2017no_duplicates"
# This is your SOURCE of zips
ZIP_DIR = r"X:\preiss_group\Shared\openreview\images_vila"
# This is your new DESTINATION for the padded images
OUTPUT_DATASET_DIR = r"C:\Users\uaeor\Downloads\Duncan\experiment2_individual_images"

# --- End Configuration ---

def load_decision_file(base_path):
    """Tries to load the decision file as a .csv or .xlsx."""
    csv_path = base_path + ".csv"
    xlsx_path = base_path + ".xlsx"
    if os.path.exists(csv_path): return pd.read_csv(csv_path)
    if os.path.exists(xlsx_path): return pd.read_excel(xlsx_path)
    print(f"ERROR: Could not find file at {csv_path} or {xlsx_path}")
    return None

def create_padded_image(img_data, canvas_size):
    """Puts a single image onto a fixed-size white canvas."""
    try:
        with Image.open(io.BytesIO(img_data)) as img:
            img.thumbnail(canvas_size, Image.Resampling.LANCZOS)
            
            canvas = Image.new('RGB', canvas_size, CANVAS_BG_COLOR)
            
            paste_x = (canvas_size[0] - img.width) // 2
            paste_y = (canvas_size[1] - img.height) // 2
            
            canvas.paste(img, (paste_x, paste_y))
            return canvas
    except Exception as e:
        print(f"   -> ERROR: Could not pad image: {e}")
        return None

def main():
    # 1. Load decision file and create lookup
    df = load_decision_file(DECISION_FILE_PATH)
    if df is None: return

    if 'Paper_id' not in df.columns or 'Decision' not in df.columns:
        print(f"ERROR: File must contain 'Paper_id' and 'Decision' columns.")
        return
        
    try:
        decision_lookup = dict(zip(df['Paper_id'].str.strip(), df['Decision']))
    except AttributeError:
        decision_lookup = dict(zip(df['Paper_id'], df['Decision']))
    print(f"Loaded {len(decision_lookup)} paper decisions.")

    # 2. Get all zip files and match them to decisions
    all_zip_files = glob.glob(os.path.join(ZIP_DIR, "*.zip"))
    
    # Filter to only the 100 zips you processed before
    # (or all zips that have a decision)
    files_to_process = []
    for zip_path in all_zip_files:
        paper_id = os.path.basename(zip_path).replace('.vila.zip', '')
        raw_decision = decision_lookup.get(paper_id)
        
        if raw_decision is not None:
            final_class = DECISION_MAP.get(raw_decision)
            if final_class:
                files_to_process.append((zip_path, paper_id, final_class))
    
    # We'll use the first 100 mappable files, as in your first experiment
    files_to_process = files_to_process[:100]
    print(f"Found {len(files_to_process)} papers with decisions to process.")

    # 3. Split papers into train and validation sets
    try:
        labels = [item[2] for item in files_to_process]
        train_papers, val_papers = train_test_split(
            files_to_process, test_size=VALIDATION_SPLIT_SIZE, random_state=42, stratify=labels)
    except ValueError:
        train_papers, val_papers = train_test_split(
            files_to_process, test_size=VALIDATION_SPLIT_SIZE, random_state=42)
            
    print(f"Splitting into {len(train_papers)} train papers and {len(val_papers)} val papers.")

    # 4. Process all papers and save padded images
    total_images_saved = 0
    
    # Helper function to process a list of papers
    def process_paper_list(paper_list, split_name):
        img_count = 0
        for zip_path, paper_id, final_class in paper_list:
            
            # This is the save directory, e.g., .../train/accept
            save_dir = os.path.join(OUTPUT_DATASET_DIR, split_name, final_class)
            os.makedirs(save_dir, exist_ok=True)
            
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    image_filenames = [f for f in zf.namelist() if f.endswith('.png')]
                    
                    for i, file_name in enumerate(image_filenames):
                        with zf.open(file_name) as f:
                            padded_image = create_padded_image(f.read(), CANVAS_SIZE)
                            
                            if padded_image:
                                # Save the new image
                                # Filename e.g., _2CLeIIYMPd_fig1.jpg
                                save_name = f"{paper_id}_img{i+1}.{OUTPUT_FORMAT.lower()}"
                                save_path = os.path.join(save_dir, save_name)
                                padded_image.save(save_path, OUTPUT_FORMAT, quality=OUTPUT_QUALITY)
                                img_count += 1
            except Exception as e:
                print(f"ERROR processing zip {zip_path}: {e}")
        return img_count

    print("Processing training files...")
    train_img_count = process_paper_list(train_papers, 'train')
    
    print("Processing validation files...")
    val_img_count = process_paper_list(val_papers, 'val')
    
    print("\n--- Padded Image Dataset Creation Complete ---")
    print(f"Saved {train_img_count} training images.")
    print(f"Saved {val_img_count} validation images.")
    print(f"Total images: {train_img_count + val_img_count}")
    print(f"Dataset created at: {OUTPUT_DATASET_DIR}")


if __name__ == "__main__":
    # Ensure all required libraries are present
    try: import pandas; import PIL; import sklearn
    except ImportError:
        print("ERROR: Missing libraries. Please run:")
        print("pip install pandas openpyxl Pillow scikit-learn")
        exit()
        
    main()