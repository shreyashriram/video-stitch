import cv2
import numpy as np
import glob
from tqdm import tqdm

video_name = "video2"

# CONFIG
FRAME_DIR = f"extracted_frames/{video_name}"
FRAME_SCALE = 0.5
OUTPUT_IMG = f"output/panorama_{video_name}.jpg"

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

def smart_crop_black(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold to binary (non-black pixels)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # Find contours around the non-black areas
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image  # fallback if empty
    
    # Get bounding box of all non-black areas
    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    cropped = image[y:y+h, x:x+w]
    return cropped

def alpha_blend(base, overlay, mask):
    """
    Blend overlay onto base where mask == True
    """
    alpha = 0.5  # You can adjust blend strength
    blended = base.copy()
    blended[mask] = cv2.addWeighted(base[mask], (1 - alpha), overlay[mask], alpha, 0)
    return blended

# MAIN
frame_files = sorted(glob.glob(f"{FRAME_DIR}/frame_*.jpg"))
print(f"📂 Found {len(frame_files)} frames.")

if len(frame_files) < 2:
    print("❌ Not enough frames to stitch.")
    exit()

# Load base image
base_img = load_and_resize(frame_files[0], scale=FRAME_SCALE)
h_base, w_base = base_img.shape[:2]

# Large initial canvas
canvas_height = h_base * 6
canvas_width = w_base * 10
canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

# Start somewhere in the middle
x_offset = w_base * 4
y_offset = h_base * 2

# Identity transform with offset
accumulated_H = np.eye(3)
accumulated_H[0, 2] = x_offset
accumulated_H[1, 2] = y_offset

# Place base image
canvas[y_offset:y_offset+h_base, x_offset:x_offset+w_base] = base_img

# List to track corners (for debugging, optional)
tracked_corners = [
    (x_offset, y_offset),
    (x_offset + w_base, y_offset),
    (x_offset, y_offset + h_base),
    (x_offset + w_base, y_offset + h_base)
]

prev_img = base_img.copy()

for i in tqdm(range(1, len(frame_files)), desc="🔧 Stitching & Blending"):
    curr_img = load_and_resize(frame_files[i], scale=FRAME_SCALE)

    src_pts, dst_pts = detect_and_match_orb(prev_img, curr_img)

    if src_pts is None or dst_pts is None:
        print(f"⚠️ Skipping frame {i} due to poor match.")
        continue

    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if H is None:
        print(f"⚠️ Homography failed for frame {i}")
        continue

    # Update the overall transform
    accumulated_H = accumulated_H @ H

    # Warp current image onto canvas
    warped = cv2.warpPerspective(curr_img, accumulated_H, (canvas_width, canvas_height))
    
    # Create mask for where the warped image has content
    mask = (warped > 0)
    mask_full = np.any(mask, axis=2)  # Single channel mask

    # Instead of overwriting, blend smoothly where overlap occurs
    overlap_mask = (mask_full) & (np.any(canvas > 0, axis=2))
    new_mask = (mask_full) & (~overlap_mask)

    # Blend overlapping regions
    canvas = alpha_blend(canvas, warped, overlap_mask)

    # Directly add non-overlapping regions
    canvas[new_mask] = warped[new_mask]

    # Optional: track corners for debugging/analysis
    h_curr, w_curr = curr_img.shape[:2]
    curr_corners = [
        (0, 0),
        (w_curr, 0),
        (0, h_curr),
        (w_curr, h_curr)
    ]
    transformed_corners = update_corners(curr_corners, accumulated_H)
    tracked_corners.extend(transformed_corners)

    # Update previous image
    prev_img = curr_img

# Smart crop to remove all black borders
final_img = smart_crop_black(canvas)

cv2.imwrite(OUTPUT_IMG, final_img)
print(f"✅ Panorama saved as {OUTPUT_IMG}")
