import os
import shutil
import random
from collections import defaultdict

# --- CONFIGURATION ---
SOURCE_IMG_DIR = "raw_data/img_align_celeba/img_align_celeba" # Path to unzipped images
IDENTITY_FILE = "raw_data/identity_CelebA.txt"
DEST_DIR = "stored_images"
NUM_CELEBS = 50          # How many distinct people you want
MIN_IMAGES_PER_CELEB = 20 # Minimum photos per person (to ensure good matching)

def prepare_dataset():
    # 1. Parse the Identity File
    print("Parsing identity file...")
    celeb_map = defaultdict(list)
    
    try:
        with open(IDENTITY_FILE, "r") as f:
            for line in f:
                # File format: "000001.jpg 2880"
                parts = line.split()
                if len(parts) >= 2:
                    filename = parts[0]
                    celeb_id = parts[1]
                    celeb_map[celeb_id].append(filename)
    except FileNotFoundError:
        print(f"Error: Could not find {IDENTITY_FILE}. Did you download the dataset?")
        return

    print(f"Found {len(celeb_map)} total celebrities.")

    # 2. Filter: Keep only celebs with enough images
    valid_celebs = {cid: imgs for cid, imgs in celeb_map.items() if len(imgs) >= MIN_IMAGES_PER_CELEB}
    print(f"Found {len(valid_celebs)} celebrities with >= {MIN_IMAGES_PER_CELEB} images.")

    # 3. Sample: Pick random celebrities
    if len(valid_celebs) < NUM_CELEBS:
        print("Warning: Not enough celebs meet criteria. Using all available.")
        selected_ids = list(valid_celebs.keys())
    else:
        selected_ids = random.sample(list(valid_celebs.keys()), NUM_CELEBS)

    # 4. Move and Rename Images
    if os.path.exists(DEST_DIR):
        shutil.rmtree(DEST_DIR) # Clean start
    os.makedirs(DEST_DIR)

    count = 0
    print(f"Copying images for {len(selected_ids)} selected celebrities...")
    
    for cid in selected_ids:
        images = valid_celebs[cid]
        for img_name in images:
            src_path = os.path.join(SOURCE_IMG_DIR, img_name)
            
            # RENAME STRATEGY: CelebID_OriginalName.jpg
            # This ensures your Search Engine knows "Celeb 2880" is a unique person
            new_name = f"Celeb_{cid}_{img_name}"
            dest_path = os.path.join(DEST_DIR, new_name)
            
            try:
                shutil.copy2(src_path, dest_path)
                count += 1
            except FileNotFoundError:
                print(f"Warning: Image {src_path} not found.")

    print(f"Success! Copied {count} images to '{DEST_DIR}/'.")
    print("You can now run 'python ingestion.py'")

if __name__ == "__main__":
    prepare_dataset()