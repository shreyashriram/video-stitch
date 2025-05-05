import cv2
import numpy as np
import glob
from tqdm import tqdm
import os

video_name = "video2"
FRAME_DIR = f"extracted_frames/{video_name}"
FRAME_SCALE = 1
OUTPUT_IMG = f"output/panorama_{video_name}_divide_conquer.jpg"
DEBUG_DIR = f"output/debug_{video_name}"

# Create debug directory if it doesn't exist
os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_IMG), exist_ok=True)

def load_and_resize(path, scale=1.0):
    img = cv2.imread(path)
    return cv2.resize(img, (0, 0), fx=scale, fy=scale)

def detect_and_match_orb(img1, img2):
    orb = cv2.ORB_create(1500)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None:
        return None, None
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    if len(matches) < 4:
        return None, None
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    return src_pts, dst_pts

def update_corners(corners, H):
    new_corners = []
    for corner in corners:
        corner_homo = np.array([corner[0], corner[1], 1.0])
        transformed = H @ corner_homo
        transformed /= transformed[2]
        new_corners.append((transformed[0], transformed[1]))
    return new_corners

def crop_final_canvas(canvas, corners):
    xs, ys = zip(*corners)
    min_x = int(max(min(xs), 0))
    min_y = int(max(min(ys), 0))
    max_x = int(min(max(xs), canvas.shape[1]))
    max_y = int(min(max(ys), canvas.shape[0]))
    
    cropped = canvas[min_y:max_y, min_x:max_x]
    return cropped

def stitch_pair(img1, img2):
    """Stitch a pair of images and return the result with tracked corners."""
    h1, w1 = img1.shape[:2]
    
    # Create a canvas large enough for the stitched image
    canvas_height = h1 * 3
    canvas_width = w1 * 5
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    
    # Place the first image in the center of the canvas
    x_offset = w1 * 2
    y_offset = h1
    canvas[y_offset:y_offset+h1, x_offset:x_offset+w1] = img1
    
    # Initialize the homography matrix
    accumulated_H = np.eye(3)
    accumulated_H[0, 2] = x_offset
    accumulated_H[1, 2] = y_offset
    
    # Track corners of the first image
    corners = [
        (x_offset, y_offset),
        (x_offset + w1, y_offset),
        (x_offset, y_offset + h1),
        (x_offset + w1, y_offset + h1)
    ]
    
    # Detect and match features
    src_pts, dst_pts = detect_and_match_orb(img1, img2)
    
    if src_pts is None or dst_pts is None:
        print("⚠️ Poor match between images.")
        return None, None
    
    # Find homography
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        print("⚠️ Homography failed.")
        return None, None
    
    # Apply homography to place the second image
    warped_H = accumulated_H @ H
    warped = cv2.warpPerspective(img2, warped_H, (canvas.shape[1], canvas.shape[0]))
    
    # Blend the images (simple overlay where warped image has content)
    mask = (warped > 0)
    canvas[mask] = warped[mask]
    
    # Update the tracked corners
    img2_corners = [
        (0, 0),
        (img2.shape[1], 0),
        (0, img2.shape[0]),
        (img2.shape[1], img2.shape[0])
    ]
    transformed_corners = update_corners(img2_corners, warped_H)
    all_corners = corners + transformed_corners
    
    # Crop the result
    result = crop_final_canvas(canvas, all_corners)
    
    return result, all_corners

def divide_and_conquer_stitch(images, level=0, index=0):
    """Recursively stitch images using divide and conquer approach."""
    n = len(images)
    
    # Base case: single image
    if n == 1:
        return images[0], [(0, 0), (images[0].shape[1], 0), (0, images[0].shape[0]), (images[0].shape[1], images[0].shape[0])]
    
    # Base case: pair of images
    if n == 2:
        result, corners = stitch_pair(images[0], images[1])
        if result is None:
            # Fallback to just using the first image if stitching fails
            return images[0], [(0, 0), (images[0].shape[1], 0), (0, images[0].shape[0]), (images[0].shape[1], images[0].shape[0])]
        
        # Save debug image for this level
        cv2.imwrite(f"{DEBUG_DIR}/level_{level}_index_{index}.jpg", result)
        return result, corners
    
    # Recursive case: divide and conquer
    mid = n // 2
    left_img, left_corners = divide_and_conquer_stitch(images[:mid], level+1, index*2)
    right_img, right_corners = divide_and_conquer_stitch(images[mid:], level+1, index*2+1)
    
    # Merge the results
    result, corners = stitch_pair(left_img, right_img)
    # if result is None:
    #     # Fallback to just using the left image if stitching fails
    #     return left_img, left_corners
    
    # Save debug image for this level
    cv2.imwrite(f"{DEBUG_DIR}/level_{level}_index_{index}.jpg", result)
    
    return result, corners

def main():
    # Load frame files
    frame_files = sorted(glob.glob(f"{FRAME_DIR}/frame_*.jpg"))
    print(f"📂 Found {len(frame_files)} frames.")
    
    if len(frame_files) < 2:
        print("❌ Not enough frames to stitch.")
        return
    
    # Load and resize all images
    print("🔄 Loading and resizing images...")
    images = [load_and_resize(file, scale=FRAME_SCALE) for file in tqdm(frame_files)]
    
    # Perform divide and conquer stitching
    print("🔧 Stitching using divide and conquer...")
    result, _ = divide_and_conquer_stitch(images)
    
    # Save the final panorama
    cv2.imwrite(OUTPUT_IMG, result)
    print(f"✅ Panorama saved to {OUTPUT_IMG}")
    print(f"🔍 Debug images saved to {DEBUG_DIR}")

if __name__ == "__main__":
    main()