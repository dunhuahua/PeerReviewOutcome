import os
import zipfile
import io
import pandas as pd
import glob
from PIL import Image
from ultralytics import YOLO

# --- 1. Configuration ---
# Path to your BEST trained model
MODEL_PATH = r"C:\Users\uaeor\runs\classify\vila_accept_reject_classifier5\weights\best.pt"
CANVAS_SIZE = (1024, 1024)
CANVAS_BG_COLOR = (255, 255, 255) # White

# --- 2. Paths to Paper Data ---
DECISION_FILE_PATH = r"X:\preiss_group\Shared\openreview\2017no_duplicates"
ZIP_DIR = r"X:\preiss_group\Shared\openreview\images_vila"

DECISION_MAP = {
    1: 'accept', '1': 'accept',
    0: 'reject', '0': 'reject'
}

# --- 3. Helper Functions ---
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
    except Exception: return None

def get_papers_to_predict():
    """Gets the list of 100 papers and their true labels."""
    df = load_decision_file(DECISION_FILE_PATH)
    if df is None: return []

    if 'Paper_id' not in df.columns or 'Decision' not in df.columns:
        print(f"ERROR: File must contain 'Paper_id' and 'Decision' columns.")
        return []
        
    try:
        decision_lookup = dict(zip(df['Paper_id'].str.strip(), df['Decision']))
    except AttributeError:
        decision_lookup = dict(zip(df['Paper_id'], df['Decision']))
    
    papers_to_predict = []
    all_zip_files = glob.glob(os.path.join(ZIP_DIR, "*.zip")) # Get all zips
    
    for zip_path in all_zip_files:
        paper_id = os.path.basename(zip_path).replace('.vila.zip', '')
        raw_decision = decision_lookup.get(paper_id)
        
        if raw_decision is not None:
            final_class = DECISION_MAP.get(raw_decision)
            if final_class:
                papers_to_predict.append((zip_path, paper_id, final_class))
    
    # Return the first 100 mappable papers, just like your other scripts
    return papers_to_predict[:-50]

# --- 4. Main Prediction ---
def main():
    # 1. Load the trained classifier
    print(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print("ERROR: Model file not found. Did you train it first?")
        return
        
    model = YOLO(MODEL_PATH)
    class_names = model.names
    print(f"Model loaded. Classes: {class_names}")

    # 2. Get the list of 100 papers
    paper_list = get_papers_to_predict()
    if not paper_list:
        print("ERROR: Could not find any papers to predict.")
        return
        
    print(f"\n--- Found {len(paper_list)} papers to predict ---")
    
    total_correct = 0
    
    # 3. Loop through every paper
    for paper_zip_path, paper_id, true_label in paper_list:
        
        predictions = []
        
        try:
            with zipfile.ZipFile(paper_zip_path, 'r') as zf:
                image_filenames = [f for f in zf.namelist() if f.endswith('.png')]
                
                for i, file_name in enumerate(image_filenames):
                    with zf.open(file_name) as f:
                        padded_image = create_padded_image(f.read(), CANVAS_SIZE)
                        
                        if padded_image:
                            results = model(padded_image, verbose=False)
                            pred_index = results[0].probs.top1
                            pred_class = class_names[pred_index]
                            predictions.append(pred_class)
                            
        except Exception as e:
            print(f"ERROR processing zip {paper_zip_path}: {e}")
            continue # Skip to the next paper

        # 4. Tally the votes for this paper
        if not predictions:
            print(f"Paper {paper_id}: SKIPPED (No images found)")
            continue
            
        accept_votes = predictions.count('accept')
        reject_votes = predictions.count('reject')
        total_votes = accept_votes + reject_votes
        
        # --- [NEW LOGIC] ---
        # Calculate the acceptance percentage
        # (float() ensures this is not integer division)
        accept_ratio = float(accept_votes) / float(total_votes)
        
        # Set our threshold (e.g., 25%)
        # You can experiment with this number!
        VOTING_THRESHOLD = 0.25 
        
        if accept_ratio >= VOTING_THRESHOLD:
            final_decision = "accept"
        else:
            final_decision = "reject"
        # --- [END NEW LOGIC]... Rest of loop is the same ---
            
        # 5. Compare prediction to the true label
        is_correct = (final_decision == true_label)
        if is_correct:
            total_correct += 1
            
        print(f"Paper {paper_id}: Votes (A:{accept_votes}, R:{reject_votes}) -> Ratio: {accept_ratio:.2f}")
        print(f"  -> PREDICTION: {final_decision.upper()} | TRUE_LABEL: {true_label.upper()} | Result: {'CORRECT' if is_correct else 'WRONG'}")

    # 6. Print final accuracy
    print("\n--- Overall Results ---")
    accuracy = (total_correct / len(paper_list)) * 100
    print(f"Total papers predicted: {len(paper_list)}")
    print(f"Total correct: {total_correct}")
    print(f"Final Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    # Import necessary libraries
    try: import pandas; import PIL; from ultralytics import YOLO
    except ImportError:
        print("ERROR: Missing libraries. Please run:")
        print("pip install pandas Pillow ultralytics")
        exit()
        
    main()