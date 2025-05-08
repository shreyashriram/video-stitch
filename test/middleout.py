import cv2
import numpy as np
import glob
from tqdm import tqdm
import os

video_name = "video1/three_video1"
FRAME_DIR = f"extracted_frames/{video_name}"
FRAME_SCALE = 0.5
OUTPUT_IMG = f"output/panorama_{video_name}_robust_middle_out.png"
DEBUG_DIR = f"output/debug_{video_name}_robust_middle_out"

# Create debug directory if it doesn't exist
os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_IMG), exist_ok=True)

def load_and_resize(path, scale=1.0):
    img = cv2.imread(path)
    return cv2.resize(img, (0, 0), fx=scale, fy=scale)

def detect_and_match_features(img1, img2, method="orb", min_matches=10):
    """
    Enhanced feature detection and matching with multiple methods.
    
    Args:
        img1, img2: Input images
        method: One of "sift", "orb", "combined"
        min_matches: Minimum number of good matches required
        
    Returns:
        src_pts, dst_pts: Matched feature points or None if insufficient matches
    """
    # Convert images to grayscale for feature detection
    if len(img1.shape) == 3 and img1.shape[2] == 4:  # BGRA
        gray1 = cv2.cvtColor(img1[:,:,:3], cv2.COLOR_BGR2GRAY)
    else:  # BGR
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        
    if len(img2.shape) == 3 and img2.shape[2] == 4:  # BGRA
        gray2 = cv2.cvtColor(img2[:,:,:3], cv2.COLOR_BGR2GRAY)
    else:  # BGR
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization to improve feature detection
    gray1 = cv2.equalizeHist(gray1)
    gray2 = cv2.equalizeHist(gray2)
    
    # Initialize output
    src_pts, dst_pts = None, None
    
    if method == "sift" or method == "combined":
        # Try SIFT features (better quality but slower)
        sift = cv2.SIFT_create(nfeatures=2000)
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        
        if des1 is not None and des2 is not None and len(kp1) > min_matches and len(kp2) > min_matches:
            # Use FLANN matcher for SIFT (faster for high-dimensional features)
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            # Get 2 best matches for each feature and apply ratio test
            matches = flann.knnMatch(des1, des2, k=2)
            
            # Apply Lowe's ratio test to filter good matches
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:  # Ratio test
                    good_matches.append(m)
            
            if len(good_matches) >= min_matches:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                print(f"SIFT found {len(good_matches)} good matches")
                return src_pts, dst_pts
    
    if method == "orb" or (method == "combined" and src_pts is None):
        # Try ORB features as fallback (faster but sometimes lower quality)
        orb = cv2.ORB_create(nfeatures=3000, scaleFactor=1.2, nlevels=8, edgeThreshold=31)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        
        if des1 is not None and des2 is not None and len(kp1) > min_matches and len(kp2) > min_matches:
            # Use BFMatcher with Hamming distance (appropriate for binary descriptors like ORB)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            
            # Get 2 best matches for each feature
            matches = bf.knnMatch(des1, des2, k=2)
            
            # Apply ratio test
            good_matches = []
            for pair in matches:
                if len(pair) == 2:  # Sometimes knnMatch returns only 1 match
                    m, n = pair
                    if m.distance < 0.8 * n.distance:  # Slightly more permissive for ORB
                        good_matches.append(m)
                elif len(pair) == 1:
                    good_matches.append(pair[0])
            
            if len(good_matches) >= min_matches:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                print(f"ORB found {len(good_matches)} good matches")
                return src_pts, dst_pts
    
    # If we reach here, neither method found enough matches
    print("⚠️ Insufficient matches found with any method")
    return None, None

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

def stitch_pair(img1, img2, attempt=0):
    """
    Stitch a pair of images with robust feature matching and error handling.
    Attempts multiple matching methods if the first one fails.
    """
    # Check if img1 is BGR or BGRA
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        h1, w1 = img1.shape[:2]
        img1_alpha = cv2.cvtColor(img1, cv2.COLOR_BGR2BGRA)
        img1_alpha[:, :, 3] = 255  # Set alpha to fully opaques
    else:
        h1, w1 = img1.shape[:2]
        img1_alpha = img1.copy()
    
    # Create a canvas large enough for the stitched image with alpha channel
    canvas_height = h1 * 3
    canvas_width = w1 * 5
    canvas = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)
    
    # Place the first image in the center of the canvas
    x_offset = w1 * 2
    y_offset = h1
    canvas[y_offset:y_offset+h1, x_offset:x_offset+w1] = img1_alpha
    
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
    
    # Try feature detection methods in order of preference
    methods = ["combined", "sift", "orb"]
    method = methods[min(attempt, len(methods)-1)]
    
    # Make sure img2 is in the right format for feature detection
    if len(img2.shape) == 3 and img2.shape[2] == 4:
        img2_for_detection = cv2.cvtColor(img2, cv2.COLOR_BGRA2BGR)
    else:
        img2_for_detection = img2
        
    # Similarly for img1
    if len(img1.shape) == 3 and img1.shape[2] == 4:
        img1_for_detection = cv2.cvtColor(img1, cv2.COLOR_BGRA2BGR)
    else:
        img1_for_detection = img1
    
    # Detect and match features
    src_pts, dst_pts = detect_and_match_features(img1_for_detection, img2_for_detection, method=method)
    
    if src_pts is None or dst_pts is None:
        print(f"⚠️ Poor match between images using {method}.")
        
        # If still on the first attempt, try with a different method
        if attempt < len(methods) - 1:
            print(f"Trying with alternate method...")
            return stitch_pair(img1, img2, attempt + 1)
        return None, None
    
    # Find homography with RANSAC for robustness
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    
    # Check if homography is valid
    if H is None:
        print("⚠️ Homography failed.")
        return None, None
    
    # Additional check for degenerate homography
    if np.abs(np.linalg.det(H)) < 1e-10:
        print("⚠️ Degenerate homography detected (nearly singular matrix).")
        return None, None
    
    # Check if homography produces reasonable scaling
    h_scale = np.sqrt(H[0, 0]**2 + H[0, 1]**2)
    v_scale = np.sqrt(H[1, 0]**2 + H[1, 1]**2)
    
    if h_scale > 4.0 or h_scale < 0.25 or v_scale > 4.0 or v_scale < 0.25:
        print(f"⚠️ Unreasonable scaling detected in homography: h_scale={h_scale}, v_scale={v_scale}")
        
        # Try again with a different method if available
        if attempt < len(methods) - 1:
            print(f"Trying with alternate method...")
            return stitch_pair(img1, img2, attempt + 1)
        return None, None
    
    # Convert img2 to BGRA before warping
    if len(img2.shape) == 3 and img2.shape[2] == 3:
        img2_alpha = cv2.cvtColor(img2, cv2.COLOR_BGR2BGRA)
        img2_alpha[:, :, 3] = 255  # Set alpha to fully opaque
    else:
        img2_alpha = img2.copy()
    
    # Apply homography to place the second image
    warped_H = accumulated_H @ H
    warped = cv2.warpPerspective(img2_alpha, warped_H, (canvas.shape[1], canvas.shape[0]))
    
    # Use a more sophisticated blending approach
    # Create a mask where the warped image has content
    mask = (warped[:, :, 3] > 0).astype(np.uint8) * 255
    
    # Create a mask where the first image has content
    first_img_mask = np.zeros_like(canvas[:,:,0], dtype=np.uint8)
    first_img_mask[y_offset:y_offset+h1, x_offset:x_offset+w1] = 255
    
    # Find the overlap region
    overlap = (mask > 0) & (first_img_mask > 0)
    
    # Create a combined canvas for blending
    blended_canvas = canvas.copy()
    
    # In the non-overlapping regions, just use the corresponding image
    non_overlap_mask = (mask > 0) & (~overlap)
    blended_canvas[non_overlap_mask] = warped[non_overlap_mask]
    
    # In the overlap region, use a weighted average for smoother transition
    if np.any(overlap):
        # Create a gradient weight for the overlap region
        overlap_coords = np.where(overlap)
        for y, x in zip(overlap_coords[0], overlap_coords[1]):
            # Simple alpha blending (0.5 weight to each image)
            blended_canvas[y, x, :3] = canvas[y, x, :3] * 0.5 + warped[y, x, :3] * 0.5
            blended_canvas[y, x, 3] = 255  # Set alpha to fully opaque in overlap
    
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
    result = crop_final_canvas(blended_canvas, all_corners)
    
    return result, all_corners

def middle_out_stitch(images, level=0):
    """Middle-out approach for stitching with enhanced robustness."""
    n = len(images)
    
    # Base case: single image
    if n == 1:
        # Convert to BGRA
        if len(images[0].shape) == 3 and images[0].shape[2] == 3:
            img_alpha = cv2.cvtColor(images[0], cv2.COLOR_BGR2BGRA)
            img_alpha[:, :, 3] = 255  # Set alpha to fully opaque
        else:
            img_alpha = images[0].copy()
        return img_alpha, [(0, 0), (images[0].shape[1], 0), (0, images[0].shape[0]), (images[0].shape[1], images[0].shape[0])]
    
    # Base case: pair of images
    if n == 2:
        result, corners = stitch_pair(images[0], images[1])
        if result is None:
            # Fallback to just using the first image if stitching fails
            if len(images[0].shape) == 3 and images[0].shape[2] == 3:
                img_alpha = cv2.cvtColor(images[0], cv2.COLOR_BGR2BGRA)
                img_alpha[:, :, 3] = 255  # Set alpha to fully opaque
            else:
                img_alpha = images[0].copy()
            return img_alpha, [(0, 0), (images[0].shape[1], 0), (0, images[0].shape[0]), (images[0].shape[1], images[0].shape[0])]
        
        # Save debug image as PNG
        cv2.imwrite(f"{DEBUG_DIR}/level_{level}_pair_{n}.png", result)
        return result, corners
    
    # Find the middle point
    mid = n // 2
    
    # Initialize with the middle image
    if len(images[mid].shape) == 3 and images[mid].shape[2] == 3:
        middle_img = cv2.cvtColor(images[mid], cv2.COLOR_BGR2BGRA)
        middle_img[:, :, 3] = 255  # Set alpha to fully opaque
    else:
        middle_img = images[mid].copy()
        
    current_img = middle_img
    current_corners = [(0, 0), (current_img.shape[1], 0), (0, current_img.shape[0]), (current_img.shape[1], current_img.shape[0])]
    
    # Save the initial middle image for debugging
    cv2.imwrite(f"{DEBUG_DIR}/level_{level}_start_middle.png", current_img)
    
    # Process alternating images from the middle outward
    left_index = mid - 1
    right_index = mid + 1
    step = 0
    
    # Track failed stitches to know when to stop trying
    consecutive_failures = 0
    max_consecutive_failures = 3
    
    while (left_index >= 0 or right_index < n) and consecutive_failures < max_consecutive_failures:
        stitching_succeeded = False
        
        # Stitch left image if available
        if left_index >= 0:
            print(f"Stitching left image {left_index} with current result")
            left_result, left_corners = stitch_pair(current_img, images[left_index])
            
            if left_result is not None:
                current_img = left_result
                current_corners = left_corners
                cv2.imwrite(f"{DEBUG_DIR}/level_{level}_step_{step}_left.png", current_img)
                stitching_succeeded = True
                consecutive_failures = 0
            else:
                print(f"⚠️ Failed to stitch left image {left_index}")
                consecutive_failures += 1
            
            left_index -= 1
            step += 1
        
        # Stitch right image if available
        if right_index < n:
            print(f"Stitching right image {right_index} with current result")
            right_result, right_corners = stitch_pair(current_img, images[right_index])
            
            if right_result is not None:
                current_img = right_result
                current_corners = right_corners
                cv2.imwrite(f"{DEBUG_DIR}/level_{level}_step_{step}_right.png", current_img)
                stitching_succeeded = True
                consecutive_failures = 0
            else:
                print(f"⚠️ Failed to stitch right image {right_index}")
                consecutive_failures += 1
            
            right_index += 1
            step += 1
        
        # If both left and right stitching failed, increment the failure counter
        if not stitching_succeeded:
            consecutive_failures += 1
    
    if consecutive_failures >= max_consecutive_failures:
        print(f"⚠️ Stopping stitching after {max_consecutive_failures} consecutive failures")
    
    return current_img, current_corners

def main():
    # Load frame files
    frame_files = sorted(glob.glob(f"{FRAME_DIR}/sr_rgb_*.png"))
    print(f"📂 Found {len(frame_files)} frames.")
    
    if len(frame_files) < 2:
        print("❌ Not enough frames to stitch.")
        return
    
    # Load and resize all images
    print("🔄 Loading and resizing images...")
    images = [load_and_resize(file, scale=FRAME_SCALE) for file in tqdm(frame_files)]
    
    # Perform middle-out stitching
    print("🔧 Stitching using robust middle-out approach...")
    result, _ = middle_out_stitch(images)
    
    # Save the final panorama as PNG
    cv2.imwrite(OUTPUT_IMG, result)
    print(f"✅ Panorama saved to {OUTPUT_IMG}")
    print(f"🔍 Debug images saved to {DEBUG_DIR}")

if __name__ == "__main__":
    main()