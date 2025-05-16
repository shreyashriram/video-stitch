import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm

# === CONFIG ===

FRAME_DIR = "extracted_frames/conf_video_6_1_every_1"
OUTPUT_PATH = "FINALS_SR.png"

# # === 1. Load and preprocess frames ===
# def load_frames(path_pattern):
#     frame_paths = sorted(glob(path_pattern))
#     images = []
#     for fp in tqdm(frame_paths, desc="load images"):
#         # img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
#         img = cv2.imread(fp, cv2.IMREAD_COLOR)

#         if img is None:
#             print(f"⚠️ Failed to load {fp}")
#             continue
#         images.append(img)
        
#     return images

def load_frames(path_list):
    images = []
    for fp in tqdm(path_list, desc="load images"):
        img = cv2.imread(fp, cv2.IMREAD_COLOR)
        if img is None:
            print(f"⚠️ Failed to load {fp}")
            continue
        images.append(img)
    return images


# === 3. Stitch images ===
def stitch_images(images):
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    # stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    status, pano = stitcher.stitch(images)
    if status != cv2.Stitcher_OK:
        print(f"Stitching failed: error code {status}")
        return None
    return pano

if __name__ == "__main__":
    # from multiprocessing import set_start_method
    # set_start_method("fork")  # Use "spawn" if "fork" causes issues

    def main():
        
        
        # image_list = load_frames(os.path.join(FRAME_DIR, "*.png"))  # or .png
        all_paths = sorted(glob(os.path.join(FRAME_DIR, "*.png")))

        # Only use even-numbered frames (e.g., frame_0.png, frame_2.png)
        even_paths = [p for p in all_paths if int(os.path.basename(p).split("_")[-1].split(".")[0]) % 2 == 0]

        # Load only even-numbered images
        image_list = load_frames(even_paths)

        

        if len(image_list) < 2:
            print("❗ Need at least 2 valid images to stitch.")
            return

        result = stitch_images(image_list)
        if result is not None:
            cv2.imwrite(OUTPUT_PATH, result)
            print(f"✅ Stitched image saved to {OUTPUT_PATH}")

    main()
