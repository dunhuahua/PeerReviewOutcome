import pandas as pd
import os
import shutil
import glob
from sklearn.model_selection import train_test_split

# --- 1. Config ---

DECISION_MAP = {
    1: 'accept', '1': 'accept',
    0: 'reject', '0': 'reject'
}

# --- 2. Paths ---
# choose which to do
side_composites = r"C:\Users\uaeor\Downloads\Duncan\experiment1_composites"
up_down_composites = r"C:\Users\uaeor\Downloads\Duncan\experiment1_up_down_composites"

DECISION_FILE_PATH = r"X:\preiss_group\Shared\openreview\2017no_duplicates"
COMPOSITE_DIR = up_down_composites
OUTPUT_DATASET_DIR = r"C:\Users\uaeor\Downloads\Duncan\classifier2_dataset"

VALIDATION_SPLIT_SIZE = 0.20


def load_decision_file(base_path):
    """Tries to load the decision file as a .csv or .xlsx."""
    csv_path = base_path + ".csv"
    xlsx_path = base_path + ".xlsx"
    if os.path.exists(csv_path):
        print(f"Found and loading {csv_path}...")
        return pd.read_csv(csv_path)
    elif os.path.exists(xlsx_path):
        print(f"Found and loading {xlsx_path}...")
        return pd.read_excel(xlsx_path)
    else:
        print(f"ERROR: Could not find file at {csv_path} or {xlsx_path}")
        return None

def main():
    # 1. Load decision file
    df = load_decision_file(DECISION_FILE_PATH)
    if df is None: return

    # --- [CHANGED] ---
    if 'Paper_id' not in df.columns or 'Decision' not in df.columns:
        print(f"ERROR: File must contain 'Paper_id' and 'Decision' columns.")
    # --- [END CHANGED] ---
        print(f"Found columns: {df.columns.to_list()}")
        return

    # 3. Create the lookup dictionary
    try:
        # --- [CHANGED] ---
        decision_lookup = dict(zip(df['Paper_id'].str.strip(), df['Decision']))
    except AttributeError:
        # Handle non-string Paper_Id just in case
        # --- [CHANGED] ---
        decision_lookup = dict(zip(df['Paper_id'], df['Decision']))
        
    print(f"Loaded {len(decision_lookup)} paper decisions.")

    # 2. Find all composite images
    image_files = glob.glob(os.path.join(COMPOSITE_DIR, "*.jpeg"))
    if not image_files:
        print(f"ERROR: No .jpeg images found in {COMPOSITE_DIR}")
        return
    print(f"Found {len(image_files)} images to process...")

    # 3. Create a list of (source_path, paper_id, class_name)
    all_files_to_sort = []
    skipped_count = 0

    for img_path in image_files:
        filename = os.path.basename(img_path)
        paper_id = filename.split('.composite.jpeg')[0]
        
        raw_decision = decision_lookup.get(paper_id)
        
        if raw_decision is None:
            skipped_count += 1
            continue
            
        final_class = DECISION_MAP.get(raw_decision)
        
        if final_class:
            all_files_to_sort.append((img_path, final_class))
        else:
            print(f"Skipping {paper_id}: Decision '{raw_decision}' not in DECISION_MAP.")
            skipped_count += 1

    print(f"Found {len(all_files_to_sort)} mappable images.")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} images (no decision or unmapped).")

    # 4. Split the list into training and validation sets
    if not all_files_to_sort:
        print("No files to sort. Exiting.")
        return
        
    try:
        labels = [item[1] for item in all_files_to_sort]
        train_files, val_files = train_test_split(
            all_files_to_sort,
            test_size=VALIDATION_SPLIT_SIZE,
            random_state=42,  # for reproducible results
            stratify=labels
        )
    except ValueError:
        print("Not enough samples to stratify, splitting randomly.")
        train_files, val_files = train_test_split(
            all_files_to_sort,
            test_size=VALIDATION_SPLIT_SIZE,
            random_state=42
        )

    print(f"Splitting into {len(train_files)} training files and {len(val_files)} validation files.")

    # 5. Create new folder structure and copy files
    
    def copy_files(file_list, split_name):
        count = 0
        for img_path, final_class in file_list:
            dest_dir = os.path.join(OUTPUT_DATASET_DIR, split_name, final_class)
            os.makedirs(dest_dir, exist_ok=True)
            
            dest_path = os.path.join(dest_dir, os.path.basename(img_path))
            shutil.copyfile(img_path, dest_path)
            count += 1
        print(f"Copied {count} files to '{split_name}' folder.")

    # Copy training files
    copy_files(train_files, 'train')
    
    # Copy validation files
    copy_files(val_files, 'val')

    print("\n--- Auto-Sorting Complete ---")
    print(f"New dataset created at: {OUTPUT_DATASET_DIR}")

if __name__ == "__main__":
    try:
        import pandas
    except ImportError:
        print("ERROR: 'pandas' library not found. Please run: pip install pandas openpyxl")
        exit()
    try:
        import sklearn
    except ImportError:
        print("ERROR: 'scikit-learn' library not found. Please run: pip install scikit-learn")
        exit()
        
    main()