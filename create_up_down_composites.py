import os
import glob
import zipfile
import math
import io
from PIL import Image

# --- Configuration ---

# All output images will be this fixed size.
CANVAS_SIZE = (1024, 1024)

CANVAS_BG_COLOR = (255, 255, 255) # White background
OUTPUT_FORMAT = "JPEG" # Use JPEG for smaller file sizes
OUTPUT_QUALITY = 90    # JPEG quality

def create_composite_image_stack(image_files, canvas_size):
    """
    Takes a list of PIL Image objects and stacks them vertically on
    a new, fixed-size canvas.
    Each image will be resized to fit the full canvas width.
    """
    num_images = len(image_files)
    
    # 1. Create the blank canvas
    canvas = Image.new('RGB', canvas_size, CANVAS_BG_COLOR)
    
    if num_images == 0:
        print("   -> No images to paste. Returning blank canvas.")
        return canvas

    print(f"   -> Stacking {num_images} images vertically...")

    # this takes hieght, compared to the side composites which takes width
    max_image_height = canvas_size[1] // num_images
    
    # Each image will take the full width of the canvas
    target_width = canvas_size[0]
    
    current_y_offset = 0

    # 2. Paste each image into its section
    for i, img in enumerate(image_files):
        # Ensure image is RGB (to match canvas)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Calculate resize dimensions for the current image
        # It should fit the target_width and not exceed max_image_height
        
        # Calculate new height based on target_width and aspect ratio
        aspect_ratio = img.height / img.width
        resized_height = int(target_width * aspect_ratio)
        
        # Ensure it doesn't exceed the max_image_height
        if resized_height > max_image_height:
            resized_height = max_image_height
            # Recalculate width to maintain aspect ratio with new height
            resized_width = int(resized_height / aspect_ratio)
        else:
            resized_width = target_width # Use full width if height allows

        # Resize the image using calculated dimensions
        img = img.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
        
        # Create a new, blank "strip" image (for centering)
        strip = Image.new('RGB', (target_width, max_image_height), CANVAS_BG_COLOR)
        
        # Calculate position to center the image horizontally within its strip
        paste_x = (target_width - img.width) // 2
        paste_y = (max_image_height - img.height) // 2 # Center vertically within its allocated strip height
        
        strip.paste(img, (paste_x, paste_y))
        
        # Paste the strip onto the main canvas
        canvas.paste(strip, (0, current_y_offset))
        
        # Move the offset down for the next image
        current_y_offset += max_image_height
        
    return canvas


def process_zip_files(input_dir, output_dir):
    """
    Main function to find all zips, extract images, and create composites.
    """
    zip_files = glob.glob(os.path.join(input_dir, "*.zip"))
    
    if not zip_files:
        print(f"ERROR: No .zip files found in {input_dir}")
        return

    # Slice the list to get only the first 100 files, as previously agreed
    zip_files_to_process = zip_files[:100]
    print(f"Found {len(zip_files)} total zip files. Processing the first {len(zip_files_to_process)}.")

    for zip_path in zip_files_to_process:
        paper_id = os.path.basename(zip_path).replace('.vila.zip', '')
        print(f"\nProcessing {paper_id}...")
        
        images_in_memory = []
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                image_filenames = [f for f in zf.namelist() if f.endswith('.png')]
                
                for file_name in image_filenames:
                    with zf.open(file_name) as f:
                        image_data = io.BytesIO(f.read())
                        img = Image.open(image_data)
                        images_in_memory.append(img)
                        
            # --- [CHANGED] ---
            # Call the new stacking function
            composite_image = create_composite_image_stack(images_in_memory, CANVAS_SIZE)
            # --- [END CHANGED] ---
            
            output_filename = f"{paper_id}.composite.{OUTPUT_FORMAT.lower()}"
            output_path = os.path.join(output_dir, output_filename)
            
            composite_image.save(output_path, OUTPUT_FORMAT, quality=OUTPUT_QUALITY)
            print(f"   -> Saved stacked composite image to {output_path}")

        except Exception as e:
            print(f"   -> ERROR processing {paper_id}: {e}")
        finally:
            for img in images_in_memory:
                img.close()

if __name__ == "__main__":
    
    # --- Paths ---
    input_zip_dir = r"X:\preiss_group\Shared\openreview\images_vila"
    
    base_dir = r"C:\Users\uaeor\Downloads\Duncan"
    
    # --- [CHANGED] ---
    # New output directory for these stacked composites
    output_composite_dir = os.path.join(base_dir, "experiment1_up_down_composites")
    # --- [END CHANGED] ---
    
    # --- End Paths ---

    if not os.path.exists(output_composite_dir):
        os.makedirs(output_composite_dir)
        print(f"Created output directory: {output_composite_dir}")

    try:
        from PIL import Image
    except ImportError:
        print("ERROR: Pillow library not found.")
        print("Please install it by running: pip install Pillow")
        exit()
        
    if not os.path.exists(input_zip_dir):
        print(f"ERROR: Input directory not found at: {input_zip_dir}")
        print("Please check the path and try again.")
        exit()

    process_zip_files(input_zip_dir, output_composite_dir)
    print("\n--- Up-Down Composite Image Generation Complete ---")