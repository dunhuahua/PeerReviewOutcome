import os
import gzip
import json
import pymupdf  # Ensure pymupdf is imported
import zipfile
import tempfile
import shutil
import glob

def is_pixmap_mostly_white(pix, threshold=0.99):
    """
    Checks if a pixmap is > threshold% white.
    A simple, fast check on a sample of pixels.
    """
    try:
        # Get all pixel data
        samples = pix.samples
        total_bytes = len(samples)
        
        # We assume 3 bytes per pixel (RGB) for this check.
        # This isn't perfect, but it's fast and good enough for white.
        if total_bytes == 0:
            return True # Empty image is "white"

        # Count how many bytes are 255 (white component)
        white_bytes = samples.count(255)
        
        if (white_bytes / total_bytes) > threshold:
            return True
        return False
    except Exception:
        # In case of any error (e.g., strange format), assume it's not white
        return False

# --- [!!! FINAL ROBUST FUNCTION v3 (FIXED) !!!] ---
def extract_vila_images(vila_dir, pdf_dir, out_vila, paper_ids_list):
    """
    Extracts images from PDFs based on VILA layout data.
    
    [FIXED LOGIC v3]
    1. De-duplicates VILA list based on *significant overlap* (>=80%), 
       using a manual, epsilon-tolerant intersection check.
    2. Tries to find a matching raw (bitmap) image.
    3. If no raw image is found, falls back to "screenshot" (get_pixmap).
    4. Checks screenshot to ensure it's not just a white box.
    """
    if not os.path.exists(out_vila):
        os.makedirs(out_vila)
        print(f"Created output directory: {out_vila}")
    
    files = os.listdir(vila_dir)
    
    for vf in files:
        if not vf.endswith('.json.gz'):
            continue
        
        paper_id = vf.split('/')[-1].replace('.json.gz', '')
        if paper_id not in paper_ids_list:
            continue

        print(f'\nProcessing paper (VILA): {paper_id}')
        
        temp_paper_dir = tempfile.mkdtemp()
        print(f"   -> Created temp dir: {temp_paper_dir}")
        
        vila_path = os.path.join(vila_dir, vf)
        
        try:
            with gzip.open(vila_path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"   -> ERROR: Failed to read or parse JSON {vila_path}. Reason: {e}. Skipping.")
            shutil.rmtree(temp_paper_dir)
            continue

        pdf_path = os.path.join(pdf_dir, paper_id + ".pdf")
        
        if not os.path.exists(pdf_path):
            print(f"   -> ERROR: PDF not found at {pdf_path}. Skipping paper.")
            shutil.rmtree(temp_paper_dir)
            continue
            
        try:
            doc = pymupdf.open(pdf_path)
        except Exception as e:
            print(f"   -> ERROR: Failed to open PDF {pdf_path}. Reason: {e}. Skipping paper.")
            shutil.rmtree(temp_paper_dir)
            continue
            
        figure_entries = [e for e in data['layout'] if e['box_group']['metadata']['type'] == 'Figure']
        
        # --- [STEP 1: DE-DUPLICATE VILA ENTRIES by OVERLAP] ---
        indices_of_children_to_discard = set()
        
        # Use a more relaxed 80% overlap threshold to catch tricky sub-figures
        overlap_threshold = 0.80

        # Pre-calculate boxes to avoid redundant work
        figure_boxes = {}
        for i, entry in enumerate(figure_entries):
            if not entry['box_group']['boxes']:
                continue
            box_data = entry['box_group']['boxes'][0]
            try:
                page = int(box_data['page'])
                x0 = float(box_data['left'])
                y0 = float(box_data['top'])
                width = float(box_data['width'])
                height = float(box_data['height'])
                if width <= 0 or height <= 0:
                    continue
                # Use PDF page coordinates (0->1) for VILA logic
                rect = pymupdf.Rect(x0, y0, x0 + width, y0 + height)
                figure_boxes[i] = (page, rect, rect.width * rect.height)
            except Exception:
                continue # Skip if box data is invalid

        print(f"   -> Built {len(figure_boxes)} valid figure boxes. Starting de-duplication...")
        
        # Loop through all combinations and check for overlap
        for i in figure_boxes:
            if i in indices_of_children_to_discard:
                continue
            
            page_i, rect_i, area_i = figure_boxes[i]

            for j in figure_boxes:
                if i == j or j in indices_of_children_to_discard:
                    continue
                
                page_j, rect_j, area_j = figure_boxes[j]

                if page_i != page_j:
                    continue
                
                int_x0 = max(rect_i.x0, rect_j.x0)
                int_y0 = max(rect_i.y0, rect_j.y0)
                int_x1 = min(rect_i.x1, rect_j.x1)
                int_y1 = min(rect_i.y1, rect_j.y1)

                # Add a small tolerance (epsilon) for float comparison
                epsilon = 1e-6 
                
                if (int_x1 - int_x0) < epsilon or (int_y1 - int_y0) < epsilon:
                    # Overlap is non-existent or too small (e.g., a line/point)
                    continue
                    
                intersection_area = (int_x1 - int_x0) * (int_y1 - int_y0)
                # --- [END FIXED INTERSECTION LOGIC] ---
                
                # Calculate Intersection over Smaller Area (IoS)
                min_area = min(area_i, area_j)
                if min_area < 1e-6: # Avoid division by zero
                    continue
                    
                overlap_ratio = intersection_area / min_area
                
                if overlap_ratio > overlap_threshold:
                    # These boxes are effectively duplicates or parent/child.
                    # Discard the smaller one.
                    if area_i < area_j:
                        indices_of_children_to_discard.add(i)
                        break # 'i' is discarded, move to next 'i'
                    elif area_j < area_i:
                        indices_of_children_to_discard.add(j)
                    else:
                        # Areas are identical, discard the one with the higher index
                        if j > i:
                            indices_of_children_to_discard.add(j)
                        else:
                            indices_of_children_to_discard.add(i)
                            break # 'i' is discarded, move to next 'i'

        print(f"   -> Found {len(indices_of_children_to_discard)} sub-figures to discard (overlap method).")
        # --- [END STEP 1] ---


        # Keep track of saved raw image XREFs to avoid saving duplicates
        saved_image_xrefs = set()
        figures_saved_count = 0
        padding = 10 # Use padding for the screenshot fallback
        
        for figure_index, entry in enumerate(figure_entries):
            
            # --- CHECK IF FLAGGED FOR DISCARD ---
            if figure_index in indices_of_children_to_discard:
                print(f"     -> Skipping figure {figure_index + 1} (it's a sub-image or duplicate).")
                continue
            
            # --- Check if box was valid and processed in STEP 1 ---
            if figure_index not in figure_boxes:
                 print(f"     -> Skipping figure {figure_index + 1} (box was invalid or empty).")
                 continue
            
            box = entry['box_group']['boxes'][0]
            zero_based_page_index = int(box['page'])
            human_readable_page_number = zero_based_page_index + 1
            human_readable_figure_number = figure_index + 1

            if zero_based_page_index >= len(doc):
                print(f"   -> ERROR: Invalid page index {zero_based_page_index} for fig {human_readable_figure_number}. Max page is {len(doc)-1}.")
                continue
                
            page = doc[zero_based_page_index]
            page_rect = page.rect
            
            # --- Convert VILA coordinates (0->1) to PDF page coordinates ---
            try:
                vila_x0 = float(box['left']) * page_rect.width
                vila_y0 = float(box['top']) * page_rect.height
                vila_x1 = vila_x0 + (float(box['width']) * page_rect.width)
                vila_y1 = vila_y0 + (float(box['height']) * page_rect.height)
                vila_rect = pymupdf.Rect(vila_x0, vila_y0, vila_x1, vila_y1)
                
                if vila_rect.is_empty or vila_rect.width <= 0 or vila_rect.height <= 0:
                    print(f"     -> Skipping figure {human_readable_figure_number} (invalid VILA box dimensions).")
                    continue
            except Exception as e:
                print(f"     -> Skipping figure {human_readable_figure_number} (invalid VILA box data: {e}).")
                continue

            # --- [STEP 2: TRY TO FIND A RAW BITMAP IMAGE] ---
            intersecting_images = []
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                img_name = img_info[7]
                if not img_name:
                    continue
                try:
                    img_bbox = page.get_image_bbox(img_name)
                    img_rect = pymupdf.Rect(img_bbox) 
                    if img_rect.intersects(vila_rect) and not img_rect.is_empty:
                        intersecting_images.append((xref, img_rect))
                except Exception:
                    continue # Ignore images we can't get a bbox for
            
            best_xref = None
            if intersecting_images:
                # Find the *best* matching image (largest intersection area)
                max_intersection_area = 0.0
                for xref, img_rect in intersecting_images:
                    intersection = img_rect.intersect(vila_rect)
                    intersection_area = intersection.width * intersection.height
                    if intersection_area > max_intersection_area:
                        max_intersection_area = intersection_area
                        best_xref = xref

            if best_xref:
                # --- SUCCESS: FOUND A RAW IMAGE ---
                if best_xref in saved_image_xrefs:
                    print(f"     -> Skipping figure {human_readable_figure_number} (raw image {best_xref} already saved).")
                    continue
                
                try:
                    pix = pymupdf.Pixmap(doc, best_xref)
                    if pix.n - pix.alpha > 3:
                        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                    
                    output_filename = f"{paper_id}.page{human_readable_page_number}.fig{human_readable_figure_number}.(xref{best_xref}).png"
                    output_path = os.path.join(temp_paper_dir, output_filename)
                    
                    pix.save(output_path)
                    print(f"     -> Saved (RAW) figure to temp file: {output_filename}")
                    
                    figures_saved_count += 1
                    saved_image_xrefs.add(best_xref)
                    pix = None
                    continue # Go to the next figure
                    
                except Exception as e:
                    print(f"     -> ERROR: Failed to extract raw image xref {best_xref} for fig {human_readable_figure_number}. Reason: {e}")
            
            # --- [STEP 3: FALLBACK TO SCREENSHOT FOR VECTORS] ---
            # We only get here if best_xref is None
            print(f"     -> Info: No raw image for fig {human_readable_figure_number}. Trying screenshot (vector fallback)...")

            try:
                # Use padded coordinates for the screenshot
                padded_x0 = vila_rect.x0 - padding
                padded_y0 = vila_rect.y0 - padding
                padded_x1 = vila_rect.x1 + padding
                padded_y1 = vila_rect.y1 + padding
                
                final_x0 = max(page_rect.x0, padded_x0)
                final_y0 = max(page_rect.y0, padded_y0)
                final_x1 = min(page_rect.x1, padded_x1)
                final_y1 = min(page_rect.y1, padded_y1)
                
                clip_rect = pymupdf.Rect(final_x0, final_y0, final_x1, final_y1)
                
                if clip_rect.is_empty:
                    print(f"     -> Skipping fig {human_readable_figure_number} (empty clip rect for fallback).")
                    continue
                    
                pix = page.get_pixmap(clip=clip_rect)
                
                # --- [STEP 4: CHECK IF IT'S A WHITE BOX] ---
                if is_pixmap_mostly_white(pix):
                    print(f"     -> Skipping fig {human_readable_figure_number} (fallback screenshot was just a white box).")
                    pix = None
                    continue

                # --- SUCCESS: SAVED THE VECTOR RENDER ---
                output_filename = f"{paper_id}.page{human_readable_page_number}.fig{human_readable_figure_number}.(vector_render).png"
                output_path = os.path.join(temp_paper_dir, output_filename)
                
                pix.save(output_path)
                print(f"     -> Saved (VECTOR) figure to temp file: {output_filename}")
                figures_saved_count += 1
                pix = None
                
            except Exception as e:
                print(f"     -> ERROR: Failed fallback screenshot for fig {human_readable_figure_number}. Reason: {e}")
                
        doc.close()
        
        # --- (Rest of the zipping and cleanup logic is the same) ---
        if figures_saved_count > 0:
            zip_filename = f"{paper_id}.vila.zip"
            zip_filepath = os.path.join(out_vila, zip_filename)
            
            print(f"   -> Zipping {figures_saved_count} images to {zip_filepath}")
            with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
                for file_path in glob.glob(os.path.join(temp_paper_dir, '*.png')):
                    zf.write(file_path, arcname=os.path.basename(file_path))
            print(f"   -> Successfully created zip file: {zip_filepath}")
        else:
            print(f"   -> No figures found/saved to zip for {paper_id}.")

        try:
            shutil.rmtree(temp_paper_dir)
            print(f"   -> Cleaned up temp dir: {temp_paper_dir}")
        except Exception as e:
            print(f"   -> ERROR: Could not remove temp dir {temp_paper_dir}. Reason: {e}")

            
if __name__ == "__main__":
    
    # !!! IMPORTANT: UPDATE THESE PATHS !!!
    openreview_dir = "X:\\preiss_group\\Shared\\openreview"
    if not os.path.exists(openreview_dir):
        print(f"ERROR: Base directory not found: {openreview_dir}")
        exit()

    vila_dir = os.path.join(openreview_dir, "extracted_mmda")
    if not os.path.exists(vila_dir):
        print(f"ERROR: VILA directory not found: {vila_dir}")
        exit()
        
    pdf_dir = os.path.join(openreview_dir, "pdfs")
    if not os.path.exists(pdf_dir):
        print(f"ERROR: PDF directory not found: {pdf_dir}")
        exit()

    # Create separate output directories to avoid confusion
    out_pymupdf = os.path.join(r"C:\Users\uaeor\Downloads\Duncan", "pymupdf_output")
    out_vila = os.path.join(r"C:\Users\uaeor\Downloads\Duncan", "vila_output")
    
    if not os.path.exists(out_pymupdf):
        os.makedirs(out_pymupdf)
    if not os.path.exists(out_vila):
        os.makedirs(out_vila)

    # Process only the problematic paper
    papers_to_process = ['_2CLeIIYMPd']
    
    print("--- Starting VILA Image Extraction")
    extract_vila_images(vila_dir, pdf_dir, out_vila, papers_to_process)
    print("--- VILA Image Extraction Complete ---")
