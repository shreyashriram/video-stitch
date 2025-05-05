import cv2
import numpy as np
import glob
from tqdm import tqdm

video_name = "video2"

# CONFIG
FRAME_DIR = f"extracted_frames/{video_name}"
FRAME_SCALE = 0.3
OUTPUT_IMG = f"output/panorama_test_{video_name}.jpg"


def load_and_resize(path, scale=1.0):
    img = cv2.imread(path)
    return cv2.resize(img, (0, 0), fx=scale, fy=scale)

def detect_and_match_akaze(img1, img2):
    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return None, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        return None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    return src_pts, dst_pts

def smart_crop_black(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    cropped = image[y:y+h, x:x+w]
    return cropped

def quick_blend(canvas, warped, overlap_mask):
    """
    Lightweight feather blending for overlapping areas.
    """
    kernel = np.ones((5,5), np.float32) / 25
    blurred_canvas = cv2.filter2D(canvas, -1, kernel)
    canvas[overlap_mask] = cv2.addWeighted(blurred_canvas[overlap_mask], 0.5, warped[overlap_mask], 0.5, 0)
    return canvas

# MAIN
frame_files = sorted(glob.glob(f"{FRAME_DIR}/frame_*.jpg"))
print(f"📂 Found {len(frame_files)} frames.")

if len(frame_files) < 2:
    print("❌ Not enough frames to stitch.")
    exit()

# Load base image
base_img = load_and_resize(frame_files[0], scale=FRAME_SCALE)
h_base, w_base = base_img.shape[:2]

# Large canvas
canvas_height = h_base * 5
canvas_width = w_base * 8
canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

x_offset = w_base * 3
y_offset = h_base * 2

accumulated_H = np.eye(3)
accumulated_H[0, 2] = x_offset
accumulated_H[1, 2] = y_offset

canvas[y_offset:y_offset+h_base, x_offset:x_offset+w_base] = base_img

prev_img = base_img.copy()

for i in tqdm(range(1, len(frame_files)), desc="🔧 Fast Stitching (AKAZE)"):
    curr_img = load_and_resize(frame_files[i], scale=FRAME_SCALE)
    src_pts, dst_pts = detect_and_match_akaze(prev_img, curr_img)

    if src_pts is None or dst_pts is None:
        print(f"⚠️ Skipping frame {i} due to poor match.")
        continue

    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if H is None:
        print(f"⚠️ Homography failed for frame {i}")
        continue

    accumulated_H = accumulated_H @ H

    warped = cv2.warpPerspective(curr_img, accumulated_H, (canvas_width, canvas_height))
    mask = (np.sum(warped, axis=2) > 0).astype(np.uint8)

    overlap_mask = (mask == 1) & (np.sum(canvas, axis=2) > 0)

    # Light blending for overlaps
    if np.any(overlap_mask):
        canvas = quick_blend(canvas, warped, overlap_mask)

    # Add non-overlapping regions directly
    canvas[(mask == 1) & (np.sum(canvas, axis=2) == 0)] = warped[(mask == 1) & (np.sum(canvas, axis=2) == 0)]

    prev_img = curr_img

# Crop to final image
final_img = smart_crop_black(canvas)
cv2.imwrite(OUTPUT_IMG, final_img)
print(f"✅ Panorama saved as {OUTPUT_IMG}")
