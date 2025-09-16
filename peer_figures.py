import os
import gzip
import json
import pymupdf


def extract_all_images_using_pymupdf(pdf_path, output_path, paper_id):

    doc = pymupdf.open(pdf_path) # open a document

    for page_index in range(len(doc)): # iterate over pdf pages
        page = doc[page_index] # get the page
        image_list = page.get_images()

        # print the number of images found on the page
        if image_list:
            print(f"Found {len(image_list)} images on page {page_index}")
        else:
            print("No images found on page", page_index)        

        for image_index, img in enumerate(image_list, start=1): # enumerate the image list
            xref = img[0] # get the XREF of the image
            pix = pymupdf.Pixmap(doc, xref) # create a Pixmap

            if pix.n - pix.alpha > 3: # CMYK: convert to RGB first
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)

            pix.save(os.path.join(output_dir, paper_id + ".p" + str(page_index) + ".i" + str(image_index) + "png"))
            pix = None
    

def extract_pymupdf_images(pdf_dir, out_pymupdf):

    files = os.listdir(pdf_dir)

    for pf in files:

        if not pf.endswith('.pdf'):
            continue

        paper_id = pf.split('/')[-1].replace('.pdf', '')

        if not paper_id in ['34k1OWJWtDW']:
            continue

        print('Processing paper: %s' % paper_id)
        
        pdf_path = os.path.join(pdf_dir, pf)
        extract_all_images_using_pymupdf(pdf_path, out_pymupdf, paper_id)

    
def extract_vila_images(vila_dir, pdf_dir, out_vila):
    
    files = os.listdir(vila_dir)
    
    for vf in files:

        if not vf.endswith('.json.gz'):
            continue
        
        paper_id = vf.split('/')[-1].replace('.json.gz', '')

        if not paper_id in ['34k1OWJWtDW']:
            continue

        print('Processing paper: %s' % paper_id)
        
        vila_path = os.path.join(vila_dir, vf)
        
        with gzip.open(vila_path, 'rt', encoding='utf-8') as f:
            data = json.load(f)

        pdf_path = os.path.join(pdf_dir, paper_id + ".pdf")
        doc = pymupdf.open(pdf_path) # open document
            
        for entry in data['layout']:
            # Check if this is figure
            if entry['box_group']['metadata']['type'] == 'Figure':
                print('Figure found: %s' % entry['box_group']['boxes'])
                page_number = int(entry['box_group']['boxes'][0]['page'])
                print('- page number: %s' % page_number)
                page = doc[page_number]
                left = float(entry['box_group']['boxes'][0]['left'])
                top = float(entry['box_group']['boxes'][0]['top'])
                width = float(entry['box_group']['boxes'][0]['width'])
                height = float(entry['box_group']['boxes'][0]['height'])

                left_adjustment = 0.008
                top_adjustment = -0.015
                left += left_adjustment
                top += top_adjustment
                
                a4_width = 595.0
                a4_height = 842.0
                
                page.set_cropbox(pymupdf.Rect(left * a4_width, top * a4_height, (left + width) * a4_width, (top + height) * a4_height)) # set a cropbox for the page
                doc.save("cropped-page-1.pdf")
                print('Saved cropped-page-1.pdf')
                
                
if __name__ == "__main__":

    openreview_dir = os.path.join("/mnt", "xdrive", "preiss_group", "Shared", "openreview")
    assert os.path.exists(openreview_dir)

    vila_dir = os.path.join(openreview_dir, "extracted_mmda")
    assert os.path.exists(vila_dir)
    pdf_dir = os.path.join(openreview_dir, "pdfs")
    assert os.path.exists(pdf_dir)

    out_pymupdf = os.path.join(openreview_dir, "images_pymupdf")
    out_vila = os.path.join(openreview_dir, "images_vila")

    #extract_pymupdf_images(pdf_dir, out_pymupdf)
    extract_vila_images(vila_dir, pdf_dir, out_vila)
