import os
import glob
import math
import pandas as pd
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split

# --- Configuration ---
# 1. Clean Source (The folder you just deleted tables from)
CLEAN_SOURCE_DIR = r"C:\Users\uaeor\Downloads\Duncan\experiment2_individual_images"

# 2. Decision File (To know if they are accept/reject)
DECISION_FILE = r"X:\preiss_group\Shared\openreview\2017no_duplicates"

# 3. Output Directories
OUT_SIDE = r"C:\Users\uaeor\Downloads\Duncan\final_dataset_side_by_side"
OUT_UP   = r"C:\Users\uaeor\Downloads\Duncan\final_dataset_up_down"

CANVAS_SIZE = (1024, 1024)
BG_COLOR = (255, 255, 255)
VAL_SPLIT = 0.2
# ---------------------

def load_decisions():
    csv_path = DECISION_FILE + ".csv"
    xlsx_path = DECISION_FILE + ".xlsx"
    if os.path.exists(csv_path): df = pd.read_csv(csv_path)
    elif os.path.exists(xlsx_path): df = pd.read_excel(xlsx_path)
    else: return None
    
    # Create mapping: PaperID -> 'accept' or 'reject'
    mapping = {}
    for _, row in df.iterrows():
        pid = str(row['Paper_id']).strip()
        dec = str(row['Decision']).strip()
        if dec == '1': mapping[pid] = 'accept'
        elif dec == '0': mapping[pid] = 'reject'
    return mapping

def create_grid(images):
    if not images: return Image.new('RGB', CANVAS_SIZE, BG_COLOR)
    canvas = Image.new('RGB', CANVAS_SIZE, BG_COLOR)
    cols = int(math.ceil(math.sqrt(len(images))))
    rows = int(math.ceil(len(images) / float(cols)))
    w, h = CANVAS_SIZE[0] // cols, CANVAS_SIZE[1] // rows
    
    for i, img in enumerate(images):
        img = img.copy()
        img.thumbnail((w, h), Image.Resampling.LANCZOS)
        px = (i % cols) * w + (w - img.width) // 2
        py = (i // cols) * h + (h - img.height) // 2
        canvas.paste(img, (px, py))
    return canvas

def create_stack(images):
    if not images: return Image.new('RGB', CANVAS_SIZE, BG_COLOR)
    canvas = Image.new('RGB', CANVAS_SIZE, BG_COLOR)
    max_h = CANVAS_SIZE[1] // len(images)
    target_w = CANVAS_SIZE[0]
    current_y = 0
    
    for img in images:
        img = img.copy()
        aspect = img.height / img.width
        new_h = int(target_w * aspect)
        if new_h > max_h:
            new_h = max_h
            new_w = int(new_h / aspect)
        else: new_w = target_w
        
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        px = (target_w - img.width) // 2
        py = current_y + (max_h - img.height) // 2
        canvas.paste(img, (px, py))
        current_y += max_h
    return canvas

def main():
    print("Loading decisions...")
    decisions = load_decisions()
    if not decisions: return

    print("Grouping images by Paper ID...")
    # Find all clean images
    all_files = glob.glob(os.path.join(CLEAN_SOURCE_DIR, "**", "*.jpeg"), recursive=True)
    all_files += glob.glob(os.path.join(CLEAN_SOURCE_DIR, "**", "*.png"), recursive=True)

    images_by_paper = {}
    for f in all_files:
        fname = os.path.basename(f)
        # Extract ID (e.g. "_2CLeII_img0.jpg" -> "_2CLeII")
        if "_img" in fname: paper_id = fname.rsplit("_img", 1)[0]
        elif "_" in fname:  paper_id = fname.split('_')[0]
        else: continue
        
        if paper_id in decisions:
            if paper_id not in images_by_paper: images_by_paper[paper_id] = []
            images_by_paper[paper_id].append(f)

    # Convert dictionary to list for splitting
    paper_ids = list(images_by_paper.keys())
    labels = [decisions[pid] for pid in paper_ids]

    # Split Papers into Train/Val
    train_ids, val_ids = train_test_split(paper_ids, test_size=VAL_SPLIT, stratify=labels, random_state=42)
    
    print(f"Processing {len(train_ids)} Train papers and {len(val_ids)} Val papers...")

    # Helper to build and save
    def process_batch(ids, split_name):
        for pid in ids:
            label = decisions[pid] # 'accept' or 'reject'
            img_paths = images_by_paper[pid]
            
            # Load images
            imgs = []
            for p in img_paths:
                try: imgs.append(Image.open(p).convert('RGB'))
                except: pass
            if not imgs: continue

            # 1. Side-by-Side
            dest_side = os.path.join(OUT_SIDE, split_name, label)
            os.makedirs(dest_side, exist_ok=True)
            create_grid(imgs).save(os.path.join(dest_side, f"{pid}.jpg"), quality=90)

            # 2. Up-Down
            dest_up = os.path.join(OUT_UP, split_name, label)
            os.makedirs(dest_up, exist_ok=True)
            create_stack(imgs).save(os.path.join(dest_up, f"{pid}.jpg"), quality=90)

    process_batch(train_ids, 'train')
    process_batch(val_ids, 'val')

    print("\n--- FINAL DATASETS READY ---")
    print(f"Side-by-Side: {OUT_SIDE}")
    print(f"Up-Down:      {OUT_UP}")
    print("Individual:   (Already Clean in Source Dir)")

if __name__ == "__main__":
    main()