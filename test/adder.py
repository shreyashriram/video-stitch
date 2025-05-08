import cv2
import numpy as np
import os
import argparse

def stitch_images_with_opencv(image_paths, output_path, method='stitcher'):
    """
    Stitches multiple images together using OpenCV.
    
    Parameters:
    - image_paths: List of paths to the images
    - output_path: Path where the stitched image will be saved
    - method: 'stitcher', 'sift', or 'orb' for different feature matching approaches
    
    Returns:
    - Path to the saved stitched image
    """
    if not image_paths:
        raise ValueError("No image paths provided")
        
    # Load all images
    images = []
    for path in image_paths:
        if os.path.exists(path):
            try:
                img = cv2.imread(path)
                if img is None:
                    print(f"Error: Could not read image at {path}")
                    continue
                images.append(img)
            except Exception as e:
                print(f"Error opening {path}: {e}")
        else:
            print(f"Warning: Image path {path} does not exist")
    
    if not images:
        raise ValueError("No valid images found")
    
    # Choose stitching method
    if method == 'stitcher':
        return stitch_with_stitcher(images, output_path)
    elif method == 'sift':
        return stitch_with_sift(images, output_path)
    elif method == 'orb':
        return stitch_with_orb(images, output_path)
    else:
        raise ValueError("Method must be 'stitcher', 'sift', or 'orb'")

def stitch_with_stitcher(images, output_path):
    """Use OpenCV's Stitcher class for panorama creation"""
    # Create a Stitcher object
    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    
    # Stitch the images
    status, stitched = stitcher.stitch(images)
    
    if status != cv2.Stitcher_OK:
        error_messages = {
            cv2.Stitcher_ERR_NEED_MORE_IMGS: "Not enough images for stitching",
            cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Homography estimation failed",
            cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera parameter adjustment failed"
        }
        error_msg = error_messages.get(status, f"Unknown error (code {status})")
        raise RuntimeError(f"Stitching failed: {error_msg}")
    
    # Save result
    cv2.imwrite(output_path, stitched)
    return output_path

def stitch_with_sift(images, output_path):
    """Stitch images using SIFT feature detection and matching"""
    if len(images) < 2:
        raise ValueError("Need at least 2 images for SIFT stitching")
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Use the first image as the base
    result = images[0]
    
    for idx, img in enumerate(images[1:], 1):
        # Convert images to grayscale for feature detection
        gray1 = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find keypoints and descriptors
        keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
        
        # Match features using FLANN
        flann_index_kdtree = 1
        index_params = dict(algorithm=flann_index_kdtree, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        
        # Apply ratio test to find good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        if len(good_matches) < 4:
            print(f"Not enough matches found between image {idx-1} and {idx}")
            continue
            
        # Extract matched keypoints
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography matrix
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            print(f"Could not find homography between image {idx-1} and {idx}")
            continue
            
        # Warp image and combine with result
        h, w = result.shape[:2]
        h2, w2 = img.shape[:2]
        
        # Calculate new dimensions after warping
        pts = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, H)
        
        # Find corners of both images in the common coordinate system
        corners = np.concatenate([
            np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2),
            dst
        ])
        
        [x_min, y_min] = np.int32(corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(corners.max(axis=0).ravel() + 0.5)
        
        # Adjust transformation to account for translation
        translation = [-x_min, -y_min]
        H_translation = np.array([
            [1, 0, translation[0]],
            [0, 1, translation[1]],
            [0, 0, 1]
        ])
        H_combined = H_translation @ H
        
        # Create a canvas large enough to hold both images
        warped_img = cv2.warpPerspective(img, H_combined, (x_max - x_min, y_max - y_min))
        
        # Adjust the position of the first image on the canvas
        canvas = np.zeros_like(warped_img)
        canvas[translation[1]:translation[1] + h, translation[0]:translation[0] + w] = result
        
        # Combine images
        # Create a mask where the warped image has non-zero pixels
        mask = (warped_img != 0)
        # Copy values from the warped image to the canvas where the mask is True
        canvas[mask] = warped_img[mask]
        
        result = canvas
    
    # Save result
    cv2.imwrite(output_path, result)
    return output_path

def stitch_with_orb(images, output_path):
    """Stitch images using ORB feature detection and matching"""
    if len(images) < 2:
        raise ValueError("Need at least 2 images for ORB stitching")
    
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=1000)
    
    # Use the first image as the base
    result = images[0]
    
    for idx, img in enumerate(images[1:], 1):
        # Convert images to grayscale for feature detection
        gray1 = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find keypoints and descriptors
        keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)
        
        # Match features using Brute Force matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Take top 80% of matches
        good_matches = matches[:int(len(matches) * 0.8)]
        
        if len(good_matches) < 4:
            print(f"Not enough matches found between image {idx-1} and {idx}")
            continue
        
        # Extract matched keypoints
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography matrix
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            print(f"Could not find homography between image {idx-1} and {idx}")
            continue
            
        # Warp image and combine with result
        h, w = result.shape[:2]
        h2, w2 = img.shape[:2]
        
        # Calculate new dimensions after warping
        pts = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, H)
        
        # Find corners of both images in the common coordinate system
        corners = np.concatenate([
            np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2),
            dst
        ])
        
        [x_min, y_min] = np.int32(corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(corners.max(axis=0).ravel() + 0.5)
        
        # Adjust transformation to account for translation
        translation = [-x_min, -y_min]
        H_translation = np.array([
            [1, 0, translation[0]],
            [0, 1, translation[1]],
            [0, 0, 1]
        ])
        H_combined = H_translation @ H
        
        # Create a canvas large enough to hold both images
        warped_img = cv2.warpPerspective(img, H_combined, (x_max - x_min, y_max - y_min))
        
        # Adjust the position of the first image on the canvas
        canvas = np.zeros_like(warped_img)
        canvas[translation[1]:translation[1] + h, translation[0]:translation[0] + w] = result
        
        # Combine images
        # Create a mask where the warped image has non-zero pixels
        mask = (warped_img != 0)
        # Copy values from the warped image to the canvas where the mask is True
        canvas[mask] = warped_img[mask]
        
        result = canvas
    
    # Save result
    cv2.imwrite(output_path, result)
    return output_path

if __name__ == "__main__":
    # Your images here
    image_paths = [
        "output/debug_video5/forest1_robust_middle_out/level_0_step_14_left.png",
        "output/debug_video5/forest2_robust_middle_out/level_0_step_18_left.png",
    ]
    
    # Choose your method: 'stitcher', 'sift', or 'orb'
    output_path = f"output/forest_stitched_panorama.jpg"
    method = "sift"
    
    try:
        result_path = stitch_images_with_opencv(image_paths, output_path, method)
        print(f"Stitched image saved to {result_path}")
    except Exception as e:
        print(f"Error: {e}")