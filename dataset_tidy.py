import os
import gzip
import json
import pymupdf  # pip install pymupdf
import shutil
import glob
import random

# --- 1. CONFIGURATION ---

# Paths for extracting "BAD" examples (Tables/Formulas)
VILA_DIR = r"X:\preiss_group\Shared\openreview\extracted_mmda"
PDF_DIR  = r"X:\preiss_group\Shared\openreview\pdfs"

# Path for copying "GOOD" examples (Scientific Figures)
# This is your mixed folder from before
GOOD_SOURCE_DIR = r"C:\Users\uaeor\Downloads\Duncan\experiment2_individual_images"

# Where to build the final dataset
OUTPUT_DIR = r"C:\Users\uaeor\Downloads\Duncan\dataset_tidy"

# How many images to collect per class (Balanced dataset)
TARGET_COUNT = 1000 
VAL_SPLIT = 0.2      # 20% for validation
# ------------------------

def setup_directories():
    """Creates the empty YOLO folder structure."""
    if os.path.exists(OUTPUT_DIR):
        print(f"Cleaning up old {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)
    
    for split in ['train', 'val']:
        for cls in ['target', 'other']:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)
    print("Created empty dataset structure.")

def extract_bad_examples(limit):
    """Extracts Tables and Equations from PDFs using VILA."""
    print(f"\n--- Step 1: Extracting {limit} Tables/Formulas ---")
    
    dest_dir = os.path.join(OUTPUT_DIR, 'train', 'target')
    files = [f for f in os.listdir(VILA_DIR) if f.endswith('.json.gz')]
    target_types = ['Table', 'Equation', 'Formula']
    
    count = 0
    for vf in files:
        if count >= limit: break
        
        paper_id = vf.replace('.json.gz', '')
        pdf_path = os.path.join(PDF_DIR, paper_id + ".pdf")
        if not os.path.exists(pdf_path): continue

        try:
            with gzip.open(os.path.join(VILA_DIR, vf), 'rt', encoding='utf-8') as f:
                data = json.load(f)
            doc = pymupdf.open(pdf_path)
            
            # Find targets
            targets = [e for e in data.get('layout', []) 
                       if e['box_group']['metadata']['type'] in target_types]

            for i, entry in enumerate(targets):
                if count >= limit: break
                
                # Crop logic
                box = entry['box_group']['boxes'][0]
                page = doc[int(box['page'])]
                rect = page.rect
                x0, y0 = float(box['left'])*rect.width, float(box['top'])*rect.height
                x1, y1 = x0 + float(box['width'])*rect.width, y0 + float(box['height'])*rect.height
                
                if (x1-x0) < 50 or (y1-y0) < 50: continue # Skip tiny specks

                pix = page.get_pixmap(clip=pymupdf.Rect(x0, y0, x1, y1))
                pix.save(os.path.join(dest_dir, f"{paper_id}_{i}.png"))
                count += 1
            doc.close()
        except: continue
        
        if count % 100 == 0: print(f"  Extracted {count}...")

    print(f"Done. Saved {count} bad examples.")
    return count

def copy_good_examples(limit):
    """Copies random charts/plots from your existing folder."""
    print(f"\n--- Step 2: Copying {limit} Good Figures ---")
    
    dest_dir = os.path.join(OUTPUT_DIR, 'train', 'other')
    
    # gather all images
    all_imgs = glob.glob(os.path.join(GOOD_SOURCE_DIR, "**", "*.jpeg"), recursive=True)
    all_imgs += glob.glob(os.path.join(GOOD_SOURCE_DIR, "**", "*.png"), recursive=True)
    
    if not all_imgs:
        print("ERROR: No source images found!")
        return 0

    # Pick random samples
    selected = random.sample(all_imgs, min(len(all_imgs), limit))
    
    count = 0
    for img in selected:
        try:
            shutil.copy(img, os.path.join(dest_dir, os.path.basename(img)))
            count += 1
        except: pass
        
    print(f"Done. Copied {count} good examples.")
    return count

def create_validation_split():
    """Moves 20% of images from train to val."""
    print(f"\n--- Step 3: Creating Validation Split ({VAL_SPLIT*100}%) ---")
    
    for cls in ['target', 'other']:
        train_path = os.path.join(OUTPUT_DIR, 'train', cls)
        val_path = os.path.join(OUTPUT_DIR, 'val', cls)
        
        images = glob.glob(os.path.join(train_path, "*.*"))
        move_count = int(len(images) * VAL_SPLIT)
        
        to_move = random.sample(images, move_count)
        
        for img in to_move:
            shutil.move(img, os.path.join(val_path, os.path.basename(img)))
            
        print(f"  Moved {move_count} '{cls}' images to validation.")

def main():
    setup_directories()
    
    # 1. Get Bad Data
    num_bad = extract_bad_examples(TARGET_COUNT)
    
    # 2. Get Good Data (Try to match the number of bad data for balance)
    copy_good_examples(num_bad)
    
    # 3. Create Validation Set (Fixes the crash error)
    create_validation_split()
    
    print("\n=== DATASET READY ===")
    print(f"Location: {OUTPUT_DIR}")
    print("You can now run 'train_tidy_classifier.py' without errors.")

if __name__ == "__main__":
    main()