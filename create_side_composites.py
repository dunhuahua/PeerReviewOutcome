import os
import glob
import zipfile
import math
import io
from PIL import Image

# --- Configuration ---

# [!!!] This is the most important setting for your experiment.
# All output images will be this size, e.g., (1024, 1024) or (512, 512).
CANVAS_SIZE = (1024, 1024)

CANVAS_BG_COLOR = (255, 255, 255) # White background
OUTPUT_FORMAT = "JPEG" # Use JPEG for smaller file sizes, good for classifiers
OUTPUT_QUALITY = 90    # JPEG quality

def create_composite_image_grid(image_files, canvas_size):
    """
    Takes a list of PIL Image objects and arranges them in a grid on
    a new, fixed-size canvas.
    """
    num_images = len(image_files)
    
    # 1. Create the blank canvas
    canvas = Image.new('RGB', canvas_size, CANVAS_BG_COLOR)
    
    if num_images == 0:
        print("   -> No images to paste. Returning blank canvas.")
        return canvas

    # 2. Calculate the grid dimensions (e.g., 9 images -> 3x3 grid)
    # This creates a grid that is as "square" as possible
    cols = int(math.ceil(math.sqrt(num_images)))
    rows = int(math.ceil(num_images / float(cols)))
    
    # 3. Calculate the size of each "cell" in the grid
    cell_width = canvas_size[0] // cols
    cell_height = canvas_size[1] // rows
    
    if cell_width == 0 or cell_height == 0:
        print("   -> ERROR: Canvas size is too small for the grid. Returning blank canvas.")
        return canvas

    print(f"   -> Arranging {num_images} images into a {rows}x{cols} grid...")

    for i, img in enumerate(image_files):
        # Ensure image is RGB (to match canvas)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Resize the image to fit *within* the cell, maintaining aspect ratio
        img.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
        
        # Create a new, blank "cell" image
        # We do this to easily center the resized image
        cell = Image.new('RGB', (cell_width, cell_height), CANVAS_BG_COLOR)
        
        # Calculate position to center the image within the cell
        paste_x = (cell_width - img.width) // 2
        paste_y = (cell_height - img.height) // 2
        
        # Paste the resized image onto the (white) cell
        cell.paste(img, (paste_x, paste_y))
        
        # Calculate where this cell goes on the main canvas
        grid_x = (i % cols) * cell_width
        grid_y = (i // cols) * cell_height
        
        # Paste the cell (with the image) onto the main canvas
        canvas.paste(cell, (grid_x, grid_y))
        
    return canvas


def process_zip_files(input_dir, output_dir):
    """
    Main function to find all zips, extract images, and create composites.
    """
    zip_files = glob.glob(os.path.join(input_dir, "*.zip"))
    
    if not zip_files:
        print(f"ERROR: No .zip files found in {input_dir}")
        return

    # CHANGE
    # Slice the list to get only the first 100 files
    zip_files_to_process = zip_files[:100]
    print(f"Found {len(zip_files)} total zip files. Processing the first {len(zip_files_to_process)}.")
    
    for zip_path in zip_files_to_process:
        paper_id = os.path.basename(zip_path).replace('.vila.zip', '')
        print(f"\nProcessing {paper_id}...")
        
        images_in_memory = []
        
        try:
            # 1. Open the zip file
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Find all .png files inside the zip
                image_filenames = [f for f in zf.namelist() if f.endswith('.png')]
                
                # 2. Load all images into a list in memory
                for file_name in image_filenames:
                    with zf.open(file_name) as f:
                        # Read image data into a memory buffer
                        image_data = io.BytesIO(f.read())
                        # Open the image with PIL
                        img = Image.open(image_data)
                        # Keep the PIL object
                        images_in_memory.append(img)
                        
            # 3. Create the composite image
            composite_image = create_composite_image_grid(images_in_memory, CANVAS_SIZE)
            
            # 4. Save the final image
            output_filename = f"{paper_id}.composite.{OUTPUT_FORMAT.lower()}"
            output_path = os.path.join(output_dir, output_filename)
            
            composite_image.save(output_path, OUTPUT_FORMAT, quality=OUTPUT_QUALITY)
            print(f"   -> Saved composite image to {output_path}")

        except Exception as e:
            print(f"   -> ERROR processing {paper_id}: {e}")
        finally:
            # 5. Clean up memory
            for img in images_in_memory:
                img.close()

if __name__ == "__main__":
    
    # --- Paths ---
    
    # [!!! CHANGED as requested !!!]
    # This now points directly to your VILA output directory
    input_zip_dir = r"X:\preiss_group\Shared\openreview\images_vila"
    
    # This is the base directory for your outputs
    output_base_dir = r"C:\Users\uaeor\Downloads\Duncan"
    
    # This is the new directory for your composite images
    output_composite_dir = os.path.join(output_base_dir, "experiment1_composites")
    
    # --- End Paths ---

    # Ensure the output directory exists
    if not os.path.exists(output_composite_dir):
        os.makedirs(output_composite_dir)
        print(f"Created output directory: {output_composite_dir}")

    # Check for Pillow
    try:
        from PIL import Image
    except ImportError:
        print("ERROR: Pillow library not found.")
        print("Please install it by running: pip install Pillow")
        exit()
        
    # Check if the new input path exists
    if not os.path.exists(input_zip_dir):
        print(f"ERROR: Input directory not found at: {input_zip_dir}")
        print("Please check the path and try again.")
        exit()

    process_zip_files(input_zip_dir, output_composite_dir)
    print("\n--- Composite Image Generation Complete ---")