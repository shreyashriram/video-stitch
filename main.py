# main.py

import os
import cv2
from high_res_stitcher import HighResStitcher

VIDEO_NAME = "video1"  # Adjust to match your video folder name
FRAMES_DIR = os.path.join("extracted_frames", VIDEO_NAME)
OUTPUT_PATH = f"output/{VIDEO_NAME}_stitched.jpg"

def get_frame_paths(frames_dir):
    """Collect and return sorted paths of all frames in the extracted directory."""
    frames = sorted([
        os.path.join(frames_dir, fname)
        for fname in os.listdir(frames_dir)
        if fname.endswith('.jpg')
    ])
    return frames

def main():
    # Step 1: Get paths to all extracted frames
    frame_paths = get_frame_paths(FRAMES_DIR)
    print(f"Found {len(frame_paths)} frames to stitch.")

    # Step 2: Load images
    images = []
    for path in frame_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
        else:
            print(f"Warning: Failed to load image at {path}")

    if len(images) < 2:
        print("Need at least two valid images to stitch. Exiting.")
        return

    # Step 3: Initialize the stitcher
    stitcher = HighResStitcher(method="sift", match_ratio=0.75)

    # Step 4: Perform stitching
    print("Starting stitching process...")
    result = stitcher.stitch(images)

    if result is not None:
        print(f"Stitching complete. Saving result to {OUTPUT_PATH}")
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        cv2.imwrite(OUTPUT_PATH, result)
    else:
        print("Stitching failed.")

if __name__ == "__main__":
    main()