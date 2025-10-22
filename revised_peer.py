import os
import gzip
import json
import pymupdf
import zipfile
import tempfile
import shutil
import glob

def extract_all_images_using_pymupdf(pdf_path, output_path, paper_id):
    """
    Opens a single PDF, extracts all images page by page, converts them to 
    RGB/PNG format, and saves them to the output directory.
    """
    try:
        doc = pymupdf.open(pdf_path) # open a document
    except Exception as e:
        print(f"  -> ERROR: Failed to open PDF {pdf_path}. Reason: {e}")
        return

    for page_index in range(len(doc)): # iterate over pdf pages
        page = doc[page_index] # get the page
        image_list = page.get_images()

        # print the number of images found on the page
        if image_list:
            print(f"  -> Found {len(image_list)} images on page {page_index}")
        else:
            print(f"  -> No images found on page {page_index}")

        for image_index, img in enumerate(image_list, start=1): # enumerate the image list
            xref = img[0] # get the XREF of the image
            try:
                pix = pymupdf.Pixmap(doc, xref) # create a Pixmap
            except Exception as e:
                print(f"  -> ERROR: Failed to extract image {image_index} on page {page_index}. Reason: {e}")
                continue # Skip this image

            if pix.n - pix.alpha > 3: # CMYK: convert to RGB first
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)

            # Construct the output filename: paper_id.p[page_index].i[image_index].png
            output_filename = f"{paper_id}.p{page_index}.i{image_index}.png"
            pix.save(os.path.join(output_path, output_filename))
            pix = None # free memory
    
    doc.close() # Always close the document when done
    print(f"  -> Image extraction complete for {paper_id}.")

# ----------------------------------------------------------------------

def extract_pymupdf_images(pdf_dir, out_pymupdf, paper_ids_list):
    """
    Iterates through a directory of PDF files, filters for specific paper IDs,
    extracts all images, and saves them into a single ZIP file per paper.
    """
    if not os.path.exists(out_pymupdf):
        os.makedirs(out_pymupdf)
        print(f"Created output directory: {out_pymupdf}")

    files = os.listdir(pdf_dir)

    for pf in files:
        if not pf.endswith('.pdf'):
            continue

        paper_id = pf.replace('.pdf', '')

        if paper_id not in paper_ids_list:
            continue

        print(f'\nProcessing paper (PyMuPDF): {paper_id}')
        
        # Create a temporary directory for this paper's images
        temp_paper_dir = tempfile.mkdtemp()
        print(f"  -> Created temp dir: {temp_paper_dir}")
        
        pdf_path = os.path.join(pdf_dir, pf)
        
        # Extract images into the temporary directory
        extract_all_images_using_pymupdf(pdf_path, temp_paper_dir, paper_id)
        
        # Zip the contents of the temporary directory
        zip_filename = f"{paper_id}.pymupdf.zip"
        zip_filepath = os.path.join(out_pymupdf, zip_filename)
        
        saved_files = glob.glob(os.path.join(temp_paper_dir, '*.png'))
        
        if saved_files:
            print(f"  -> Zipping {len(saved_files)} images to {zip_filepath}")
            with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
                for file_path in saved_files:
                    zf.write(file_path, arcname=os.path.basename(file_path))
            print(f"  -> Successfully created zip file: {zip_filepath}")
        else:
            print(f"  -> No images found to zip for {paper_id}.")
            
        # Clean up the temporary directory
        try:
            shutil.rmtree(temp_paper_dir)
            print(f"  -> Cleaned up temp dir: {temp_paper_dir}")
        except Exception as e:
            print(f"  -> ERROR: Could not remove temp dir {temp_paper_dir}. Reason: {e}")

    

def extract_vila_images(vila_dir, pdf_dir, out_vila, paper_ids_list):
    """
    Extracts images from PDFs based on VILA layout data and saves them
    into a single ZIP file per paper.
    
    [MODIFIED] This version now checks for nested bounding boxes and
    discards the "child" sub-images, keeping only the main "parent" container.
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
        print(f"  -> Created temp dir: {temp_paper_dir}")
        
        vila_path = os.path.join(vila_dir, vf)
        
        with gzip.open(vila_path, 'rt', encoding='utf-8') as f:
            data = json.load(f)

        pdf_path = os.path.join(pdf_dir, paper_id + ".pdf")
        
        if not os.path.exists(pdf_path):
            print(f"  -> ERROR: PDF not found at {pdf_path}. Skipping paper.")
            shutil.rmtree(temp_paper_dir)
            continue
            
        try:
            doc = pymupdf.open(pdf_path)
        except Exception as e:
            print(f"  -> ERROR: Failed to open PDF {pdf_path}. Reason: {e}. Skipping paper.")
            shutil.rmtree(temp_paper_dir)
            continue
            
        figure_entries = [e for e in data['layout'] if e['box_group']['metadata']['type'] == 'Figure']
        padding = 10 
        figures_saved_count = 0
        
        # --- NEW LOGIC TO FIND CHILD BOXES ---
        indices_of_children_to_discard = set()
        for i, entry_a in enumerate(figure_entries):
            box_a_data = entry_a['box_group']['boxes'][0]
            a_page = int(box_a_data['page'])
            a_x0 = float(box_a_data['left'])
            a_y0 = float(box_a_data['top'])
            a_x1 = a_x0 + float(box_a_data['width'])
            a_y1 = a_y0 + float(box_a_data['height'])

            for j, entry_b in enumerate(figure_entries):
                if i == j:
                    continue
                
                box_b_data = entry_b['box_group']['boxes'][0]
                b_page = int(box_b_data['page'])
                
                if a_page != b_page:
                    continue

                b_x0 = float(box_b_data['left'])
                b_y0 = float(box_b_data['top'])
                b_x1 = b_x0 + float(box_b_data['width'])
                b_y1 = b_y0 + float(box_b_data['height'])

                tolerance = 1e-5
                if (a_x0 - tolerance <= b_x0) and (a_y0 - tolerance <= b_y0) and \
                   (a_x1 + tolerance >= b_x1) and (a_y1 + tolerance >= b_y1):
                    
                    # Box A contains Box B. Flag Box B (the child) for discarding.
                    indices_of_children_to_discard.add(j)
                    # We don't break here, as Box A might contain other children
        # --- END NEW LOGIC ---

        
        for figure_index, entry in enumerate(figure_entries):
            
            # --- NEW CHECK ---
            # Skip this entry if it was flagged as a child
            if figure_index in indices_of_children_to_discard:
                print(f"    -> Skipping figure {figure_index + 1} (it's a sub-image).")
                continue
            # --- END NEW CHECK ---
            
            box = entry['box_group']['boxes'][0]
            zero_based_page_index = int(box['page'])

            if zero_based_page_index >= len(doc):
                print(f"  -> ERROR: Invalid page index {zero_based_page_index} for paper {paper_id}. Max page is {len(doc)-1}.")
                continue
                
            page = doc[zero_based_page_index]
            page_rect = page.rect
            
            x0 = float(box['left']) * page_rect.width
            y0 = float(box['top']) * page_rect.height
            x1 = x0 + (float(box['width']) * page_rect.width)
            y1 = y0 + (float(box['height']) * page_rect.height)

            padded_x0 = x0 - padding
            padded_y0 = y0 - padding
            padded_x1 = x1 + padding
            padded_y1 = y1 + padding
            
            final_x0 = max(page_rect.x0, padded_x0)
            final_y0 = max(page_rect.y0, padded_y0)
            final_x1 = min(page_rect.x1, padded_x1)
            final_y1 = min(page_rect.y1, padded_y1)

            clip_rect = pymupdf.Rect(final_x0, final_y0, final_x1, final_y1)
            
            if not clip_rect.is_empty:
                pix = page.get_pixmap(clip=clip_rect)
                
                human_readable_figure_number = figure_index + 1
                human_readable_page_number = zero_based_page_index + 1
                
                output_filename = f"{paper_id}.page{human_readable_page_number}.fig{human_readable_figure_number}.png"
                output_path = os.path.join(temp_paper_dir, output_filename)
                
                pix.save(output_path)
                print(f"    -> Saved figure to temp file: {output_filename}")
                figures_saved_count += 1
                pix = None

        doc.close()
        
        if figures_saved_count > 0:
            zip_filename = f"{paper_id}.vila.zip"
            zip_filepath = os.path.join(out_vila, zip_filename)
            
            print(f"  -> Zipping {figures_saved_count} images to {zip_filepath}")
            with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
                for file_path in glob.glob(os.path.join(temp_paper_dir, '*.png')):
                    zf.write(file_path, arcname=os.path.basename(file_path))
            print(f"  -> Successfully created zip file: {zip_filepath}")
        else:
            print(f"  -> No figures found/saved to zip for {paper_id}.")

        try:
            shutil.rmtree(temp_paper_dir)
            print(f"  -> Cleaned up temp dir: {temp_paper_dir}")
        except Exception as e:
            print(f"  -> ERROR: Could not remove temp dir {temp_paper_dir}. Reason: {e}")

                
if __name__ == "__main__":

    openreview_dir = "X:\\preiss_group\\Shared\\openreview"
    assert os.path.exists(openreview_dir)

    vila_dir = os.path.join(openreview_dir, "extracted_mmda")
    assert os.path.exists(vila_dir)
    pdf_dir = os.path.join(openreview_dir, "pdfs")
    assert os.path.exists(pdf_dir)

    out_pymupdf = os.path.join(r"C:\Users\uaeor\Downloads\Duncan")
    out_vila = os.path.join(r"C:\Users\uaeor\Downloads\Duncan")

    papers_to_process = ['_0kaDkv3dVf', '_2CLeIIYMPd', '__ObYt4753c','_8EQ_gMAHFy','_IM-AfFhna9', '_Ko4kT3ckWy', '_kxlwvhOodK', '_LNdXw0BSx', '0BaWDGvCa5p', '0DecTiJFbm']
    
    # extract_pymupdf_images(pdf_dir, out_pymupdf, papers_to_process)
    extract_vila_images(vila_dir, pdf_dir, out_vila, papers_to_process)