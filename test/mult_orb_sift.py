import cv2
import numpy as np
import glob
import os
from tqdm import tqdm

video_name = "video11"
FRAME_DIR = f"extracted_frames/{video_name}"
FRAME_SCALE = .5
OUTPUT_DIR = f"output/{video_name}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_resize(path, scale=1.0):
    img = cv2.imread(path)
    return cv2.resize(img, (0, 0), fx=scale, fy=scale)

def detect_and_match_orb(img1, img2):
    orb = cv2.ORB_create(3000)  # Increased feature count for better matching
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None:
        return None, None
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Require more matches for robust stitching
    if len(matches) < 10:
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

def create_panorama(frame_files, suffix=""):
    if len(frame_files) < 2:
        print("❌ Not enough frames to stitch.")
        return None
    
    base_img = load_and_resize(frame_files[0], scale=FRAME_SCALE)
    h_base, w_base = base_img.shape[:2]
    
    # Create a larger canvas for the panorama
    canvas_height = h_base * 6
    canvas_width = w_base * 10
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    
    # Place the first image in the middle
    x_offset = w_base * 4
    y_offset = h_base * 2
    
    accumulated_H = np.eye(3)
    accumulated_H[0, 2] = x_offset
    accumulated_H[1, 2] = y_offset
    
    canvas[y_offset:y_offset+h_base, x_offset:x_offset+w_base] = base_img
    
    tracked_corners = [
        (x_offset, y_offset),
        (x_offset + w_base, y_offset),
        (x_offset, y_offset + h_base),
        (x_offset + w_base, y_offset + h_base)
    ]
    
    prev_img = base_img.copy()
    
    for i in tqdm(range(1, len(frame_files)), desc=f"🔧 Stitching {suffix}"):
        curr_img = load_and_resize(frame_files[i], scale=FRAME_SCALE)
        
        src_pts, dst_pts = detect_and_match_orb(prev_img, curr_img)
        if src_pts is None or dst_pts is None:
            print(f"⚠️ Skipping frame {i} due to poor match.")
            continue
        
        H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if H is None:
            print(f"⚠️ Homography failed for frame {i}")
            continue
        
        accumulated_H = accumulated_H @ H
        
        warped = cv2.warpPerspective(curr_img, accumulated_H, (canvas.shape[1], canvas.shape[0]))
        
        # Blend the images together
        mask = (warped > 0)
        canvas[mask] = warped[mask]
        
        corners = [
            (0, 0),
            (curr_img.shape[1], 0),
            (0, curr_img.shape[0]),
            (curr_img.shape[1], curr_img.shape[0])
        ]
        
        transformed_corners = update_corners(corners, accumulated_H)
        tracked_corners.extend(transformed_corners)
        
        prev_img = curr_img
    
    cropped = crop_final_canvas(canvas, tracked_corners)
    output_path = f"{OUTPUT_DIR}/panorama_{video_name}_{suffix}.jpg"
    cv2.imwrite(output_path, cropped)
    print(f"✅ Panorama saved to {output_path}")
    
    return cropped

def merge_panoramas(panoramas):
    """Merge multiple panorama images using feature matching"""
    if len(panoramas) < 2:
        return panoramas[0]
    
    # Start with first panorama
    base_pano = panoramas[0]
    
    # Create a large canvas
    canvas_height = base_pano.shape[0] * 3
    canvas_width = base_pano.shape[1] * 3
    merged_canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    
    # Place first panorama in center
    y_offset = (canvas_height - base_pano.shape[0]) // 2
    x_offset = (canvas_width - base_pano.shape[1]) // 2
    
    merged_canvas[y_offset:y_offset+base_pano.shape[0], 
                x_offset:x_offset+base_pano.shape[1]] = base_pano
    
    tracked_corners = [
        (x_offset, y_offset),
        (x_offset + base_pano.shape[1], y_offset),
        (x_offset, y_offset + base_pano.shape[0]),
        (x_offset + base_pano.shape[1], y_offset + base_pano.shape[0])
    ]
    
    # Set up initial homography
    initial_H = np.eye(3)
    initial_H[0, 2] = x_offset
    initial_H[1, 2] = y_offset
    
    # Keep track of the accumulated homography
    accumulated_H = initial_H.copy()
    
    # Merge remaining panoramas
    for i, pano in enumerate(panoramas[1:], 1):
        print(f"🔄 Merging panorama {i+1}/{len(panoramas)}...")
        
        # Use SIFT for better feature matching between panoramas
        sift = cv2.SIFT_create()
        
        # Get keypoints and descriptors for the current merged canvas
        # We need to convert non-zero parts of the canvas to grayscale
        mask = np.all(merged_canvas > 0, axis=2).astype(np.uint8) * 255
        cv2.imwrite(f"{OUTPUT_DIR}/mask_{i}.jpg", mask)  # Debug
        
        merged_gray = cv2.cvtColor(merged_canvas, cv2.COLOR_BGR2GRAY)
        merged_gray[mask == 0] = 0
        
        kp1, des1 = sift.detectAndCompute(merged_gray, mask.astype(np.uint8))
        
        # Get keypoints and descriptors for the next panorama
        pano_gray = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
        kp2, des2 = sift.detectAndCompute(pano_gray, None)
        
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            print(f"⚠️ Not enough features in panorama {i+1}, using simple positioning")
            # Position the panorama below the existing merged canvas
            next_y = tracked_corners[-1][1] + 20
            next_x = x_offset
            
            h, w = pano.shape[:2]
            merged_canvas[next_y:next_y+h, next_x:next_x+w] = pano
            
            # Update tracked corners
            tracked_corners.extend([
                (next_x, next_y),
                (next_x + w, next_y),
                (next_x, next_y + h),
                (next_x + w, next_y + h)
            ])
            continue
        
        # Match features using FLANN
        flann_index_params = dict(algorithm=1, trees=5)
        flann_search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(flann_index_params, flann_search_params)
        
        matches = flann.knnMatch(des2, des1, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        if len(good_matches) > 10:
            # Get corresponding points
            src_pts = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Find homography
            H, mask_homography = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if H is not None:
                # Warp the panorama onto the canvas
                warped = cv2.warpPerspective(pano, H, (canvas_width, canvas_height))
                
                # Blend the images
                mask_warped = np.all(warped > 0, axis=2)
                merged_canvas[mask_warped] = warped[mask_warped]
                
                # Update corners
                corners = [
                    (0, 0),
                    (pano.shape[1], 0),
                    (0, pano.shape[0]),
                    (pano.shape[1], pano.shape[0])
                ]
                
                transformed_corners = []
                for corner in corners:
                    corner_homo = np.array([corner[0], corner[1], 1.0])
                    transformed = H @ corner_homo
                    transformed /= transformed[2]
                    transformed_corners.append((transformed[0], transformed[1]))
                
                tracked_corners.extend(transformed_corners)
                continue
            
        # If we reach here, the matching failed - use simple positioning
        print(f"⚠️ Feature matching failed for panorama {i+1}, using simple positioning")
        next_y = tracked_corners[-1][1] + 20
        next_x = x_offset
        
        h, w = pano.shape[:2]
        merged_canvas[next_y:next_y+h, next_x:next_x+w] = pano
        
        # Update tracked corners
        tracked_corners.extend([
            (next_x, next_y),
            (next_x + w, next_y),
            (next_x, next_y + h),
            (next_x + w, next_y + h)
        ])
    
    # Crop the final merged canvas
    final_merged = crop_final_canvas(merged_canvas, tracked_corners)
    output_merged_path = f"{OUTPUT_DIR}/panorama_{video_name}_merged.jpg"
    cv2.imwrite(output_merged_path, final_merged)
    print(f"✅ Final merged panorama saved to {output_merged_path}")
    
    return final_merged

def main():
    # Process each clip
    panoramas = []
    
    for clip_num in range(1, 5):
        clip_path = f"{FRAME_DIR}/clip{clip_num}"
        if not os.path.exists(clip_path):
            print(f"⚠️ Clip path {clip_path} not found, skipping.")
            continue
            
        frame_files = sorted(glob.glob(f"{clip_path}/frame_*.jpg"))
        print(f"📂 Found {len(frame_files)} frames in clip{clip_num}.")
        
        panorama = create_panorama(frame_files, suffix=f"clip{clip_num}")
        if panorama is not None:
            panoramas.append(panorama)
    
    # Merge all panoramas if we have more than one
    if len(panoramas) > 1:
        merged_panorama = merge_panoramas(panoramas)
    elif len(panoramas) == 1:
        print("✅ Only one clip processed, using single panorama as result.")
        merged_panorama = panoramas[0]
        output_merged_path = f"{OUTPUT_DIR}/panorama_{video_name}_merged.jpg"
        cv2.imwrite(output_merged_path, merged_panorama)
    else:
        print("❌ No panoramas were created.")

if __name__ == "__main__":
    main()