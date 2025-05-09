
import cv2
import os

video_number = 12
clip_count = 3  # Adjust this to the number of clips you have
frame_rate = 10 # Save every tenth frame
export_as_png = False  # Set to False to export as JPEG


for clip_index in range(1, clip_count + 1):
    video_path = f"video_data/video{video_number}/clip{clip_index}.mp4"
    output_dir = f"extracted_frames/video{video_number}/clip{clip_index}"
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open {video_path}")
        continue

    count = 0
    print(f"🎬 Processing {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if int(cap.get(1)) % frame_rate == 0:
            if export_as_png:
                frame_filename = f"{output_dir}/frame_{count:03d}.png"
                success = cv2.imwrite(frame_filename, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # 0 = no compression (best quality)
            else:
                frame_filename = f"{output_dir}/frame_{count:03d}.jpg"
                success = cv2.imwrite(frame_filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])

            if success:
                print(f"✅ Saved {frame_filename}")
            else:
                print(f"⚠️ Failed to save {frame_filename}")

            count += 1

    cap.release()
    print(f"✔️ Finished extracting frames from {video_path}")


# === Multi Improved Fallback Section ===
import cv2
import numpy as np
import glob
import os
from tqdm import tqdm
import time

video_name = "video12"
FRAME_DIR = f"extracted_frames/{video_name}"
FRAME_SCALE = 1
OUTPUT_DIR = f"output/{video_name}"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_resize(path, scale=1.0):
    img = cv2.imread(path)
    if img is None:
        print(f"⚠️ Failed to load image at {path}")
        return None
    return cv2.resize(img, (0, 0), fx=scale, fy=scale)

# Matching functions

# def detect_and_match_orb(img1, img2):
#     orb = cv2.ORB_create(3000)
#     kp1, des1 = orb.detectAndCompute(img1, None)
#     kp2, des2 = orb.detectAndCompute(img2, None)
#     if des1 is None or des2 is None:
#         return None, None
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING)
#     matches = bf.knnMatch(des1, des2, k=2)
#     good = [m for m, n in matches if m.distance < 0.75 * n.distance]
#     if len(good) < 10:
#         return None, None
#     src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
#     dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
#     return src, dst

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

def detect_and_match_akaze(img1, img2):
    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return None, None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]
    if len(good) < 10:
        return None, None
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    return src, dst

def detect_and_match_sift(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return None, None
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 10:
        return None, None
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    return src, dst

# Blending with tight feathering

def blend_on_canvas(canvas, warped):
    mask = (warped.sum(axis=2) > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask_eroded = cv2.erode(mask, kernel, iterations=1)
    dist = cv2.distanceTransform(mask_eroded, cv2.DIST_L2, 5)
    alpha = dist / dist.max() if dist.max() > 0 else dist
    alpha = np.clip(alpha, 0, 1)
    alpha = cv2.GaussianBlur(alpha, (21,21), 0)[..., np.newaxis]
    return (canvas * (1 - alpha) + warped * alpha).astype(np.uint8)

# Utilities

def update_corners(corners, H):
    new = []
    for x, y in corners:
        v = np.array([x, y, 1.0])
        vt = H @ v
        vt /= vt[2]
        new.append((vt[0], vt[1]))
    return new

def crop_canvas(canvas, all_corners):
    xs = [p[0] for p in all_corners]
    ys = [p[1] for p in all_corners]
    min_x = int(max(min(xs), 0)); max_x = int(min(max(xs), canvas.shape[1]))
    min_y = int(max(min(ys), 0)); max_y = int(min(max(ys), canvas.shape[0]))
    return canvas[min_y:max_y, min_x:max_x]

# Stitch frames into a panorama

def create_panorama(frame_files, suffix=""):
    if len(frame_files) < 2:
        print("❌ Not enough frames to stitch.")
        return None
    base = load_and_resize(frame_files[0], FRAME_SCALE)
    if base is None: return None
    h0, w0 = base.shape[:2]
    ch, cw = h0 * 6, w0 * 10
    canvas = np.zeros((ch, cw, 3), dtype=np.uint8)
    x_off, y_off = w0 * 4, h0 * 2
    H_acc = np.eye(3); H_acc[0,2], H_acc[1,2] = x_off, y_off
    canvas[y_off:y_off+h0, x_off:x_off+w0] = base
    corners = [(x_off, y_off), (x_off+w0, y_off), (x_off, y_off+h0), (x_off+w0, y_off+h0)]
    prev = base.copy()
    for i in tqdm(range(1, len(frame_files)), desc=f"🔧 Stitching {suffix}"):
        curr = load_and_resize(frame_files[i], FRAME_SCALE)
        if curr is None: continue
        src, dst = detect_and_match_sift(prev, curr)
        if src is None:
            print(f"⚠️ ORB failed for frame {i}, trying AKAZE...")
            src, dst = detect_and_match_akaze(prev, curr)
        if src is None:
            print(f"⚠️ AKAZE failed for frame {i}, trying SIFT...")
            src, dst = detect_and_match_sift(prev, curr)
        if src is None:
            print(f"⚠️ All matchers failed for frame {i}, skipping.")
            continue
        H, _ = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)
        if H is None:
            print(f"⚠️ Homography failed for frame {i}")
            continue
        H_acc = H_acc @ H
        warped = cv2.warpPerspective(curr, H_acc, (cw, ch))
        canvas = blend_on_canvas(canvas, warped)
        corners += update_corners([(0,0),(curr.shape[1],0),(0,curr.shape[0]),(curr.shape[1],curr.shape[0])], H_acc)
        prev = curr
    result = crop_canvas(canvas, corners)
    out_path = f"{OUTPUT_DIR}/panorama_{video_name}_{suffix}_improved2.jpg"
    cv2.imwrite(out_path, result)
    print(f"✅ Saved panorama {suffix} to {out_path}")
    return result

# def merge_panoramas(panoramas):
#     """
#     Sequentially stitch each clip-panorama into one final panorama.
#     First try cv2.Stitcher, fallback to manual feature matching if it fails.
#     """
#     if not panoramas:
#         print("❌ No panoramas to merge.")
#         return None

#     print(f"🔄 Merging {len(panoramas)} panoramas sequentially…")
#     merged = panoramas[0]

#     for idx, pano in enumerate(panoramas[1:], start=1):
#         print(f"   · Stitching panorama {idx+1} into the final image")

#         # First, try cv2.Stitcher
#         stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
#         status, result = stitcher.stitch([merged, pano])

#         if status == cv2.Stitcher_OK:
#             merged = result
#             continue

#         print(f"⚠️  Stitcher failed on pano {idx+1} (status={status}), falling back to manual merge…")

#         # Fallback: manual feature matching + homography
#         src, dst = detect_and_match_orb(merged, pano)
#         if src is None:
#             print("⚠️ SIFT failed, trying ORB fallback...")
#             src, dst = detect_and_match_orb(merged, pano)
#         if src is None:
#             print("⚠️ ORB failed, trying AKAZE fallback...")
#             src, dst = detect_and_match_akaze(merged, pano)
#         if src is None:
#             print(f"❌ All matchers failed to merge pano {idx+1}, skipping.")
#             continue

#         H, _ = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)
#         if H is None:
#             print(f"❌ Homography estimation failed for pano {idx+1}, skipping.")
#             continue

#         # Prepare canvas large enough to hold both images
#         h1, w1 = merged.shape[:2]
#         h2, w2 = pano.shape[:2]
#         out_w, out_h = max(w1, w2) * 2, max(h1, h2) * 2
#         warped_pano = cv2.warpPerspective(pano, H, (out_w, out_h))
#         canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
#         canvas[:h1, :w1] = merged

#         merged = blend_on_canvas(canvas, warped_pano)

#     out_path = f"{OUTPUT_DIR}/panorama_{video_name}_final.jpg"
#     cv2.imwrite(out_path, merged)
#     print(f"✅ Saved final merged panorama to {out_path}")
#     return merged

def merge_panoramas(panoramas):
    """
    Sequentially stitch each clip-panorama into one final panorama.
    Always uses cv2.Stitcher, even if it reports failure.
    """
    if not panoramas:
        print("❌ No panoramas to merge.")
        return None

    print(f"🔄 Merging {len(panoramas)} panoramas sequentially…")
    merged = panoramas[0]

    for idx, pano in enumerate(panoramas[1:], start=1):
        print(f"   · Forcing Stitcher on panorama {idx+1}")

        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        status, result = stitcher.stitch([merged, pano])

        if result is not None:
            merged = result
            print(f"✅ Stitcher ran on panorama {idx+1} (status={status}) — result used regardless.")
        else:
            print(f"⚠️ Stitcher returned None on panorama {idx+1}, keeping previous merged image.")

    out_path = f"{OUTPUT_DIR}/panorama_{video_name}_final.jpg"
    cv2.imwrite(out_path, merged)
    print(f"✅ Saved final merged panorama to {out_path}")
    return merged
# 1) Blurriness detector
def is_blurry(img, thresh=100.0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < thresh

# 2) Fallback: OpenCV Stitcher
def create_panorama_stitcher(frame_files, suffix=""):
    imgs = []
    for p in frame_files:
        im = load_and_resize(p, FRAME_SCALE)
        if im is not None:
            imgs.append(im)
    if len(imgs) < 2:
        print("❌ Not enough frames for Stitcher.")
        return None
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, pano = stitcher.stitch(imgs)
    if status != cv2.Stitcher_OK:
        print(f"⚠️ Stitcher failed ({status}) for {suffix}")
        return None
    out_path = f"{OUTPUT_DIR}/panorama_{video_name}_{suffix}_stitcher.jpg"
    cv2.imwrite(out_path, pano)
    print(f"✅ Saved Stitcher fallback {suffix} to {out_path}")
    return pano

# 3) In your main loop, wrap create_panorama with blur check + fallback
def main():
    start_time = time.time()
    print(f"🚀 Starting panorama stitching for '{video_name}'")
    panoramas = []
    for clip in range(1, 5):
        path = f"{FRAME_DIR}/clip{clip}"
        if not os.path.isdir(path):
            print(f"⚠️ Clip path {path} not found, skipping.")
            continue
        frames = glob.glob(f"{path}/frame_*.jpg")
        
        print(f"📂 Found {len(frames)} frames in {path}")

        # stitch every clip with the built-in Stitcher
        # for using OpenCVStitcher
        pano = create_panorama_stitcher(frames, suffix=f"clip{clip}")

        # for using custom stitching
        # pano = create_panorama(frames, suffix=f"clip{clip}")
        if pano is not None:
            panoramas.append(pano)

    # Merge or save single panorama as before
    if panoramas:
        merge_panoramas(panoramas)
    else:
        print("❌ No panoramas were created.")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"✅ Completed stitching workflow in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()
# import cv2
# import numpy as np
# import glob
# import os
# from tqdm import tqdm
# import time

# video_name = "video12"
# FRAME_DIR = f"extracted_frames/{video_name}"
# FRAME_SCALE = .5
# OUTPUT_DIR = f"output/{video_name}"
# os.makedirs(OUTPUT_DIR, exist_ok=True)


# def load_and_resize(path, scale=1.0):
#     img = cv2.imread(path)
#     if img is None:
#         print(f"⚠️ Failed to load image at {path}")
#         return None
#     return cv2.resize(img, (0, 0), fx=scale, fy=scale)


# def detect_and_match_orb(img1, img2):
#     orb = cv2.ORB_create(1500)
#     kp1, des1 = orb.detectAndCompute(img1, None)
#     kp2, des2 = orb.detectAndCompute(img2, None)

#     if des1 is None or des2 is None:
#         return None, None

#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     matches = bf.match(des1, des2)
#     matches = sorted(matches, key=lambda x: x.distance)

#     if len(matches) < 4:
#         return None, None

#     src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
#     dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

#     return src_pts, dst_pts


# def detect_and_match_akaze(img1, img2):
#     akaze = cv2.AKAZE_create()
#     kp1, des1 = akaze.detectAndCompute(img1, None)
#     kp2, des2 = akaze.detectAndCompute(img2, None)
#     if des1 is None or des2 is None:
#         return None, None
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING)
#     matches = bf.knnMatch(des1, des2, k=2)
#     good = [m for m, n in matches if m.distance < 0.7 * n.distance]
#     if len(good) < 10:
#         return None, None
#     src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
#     dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
#     return src, dst


# def detect_and_match_sift(img1, img2):
#     sift = cv2.SIFT_create()
#     kp1, des1 = sift.detectAndCompute(img1, None)
#     kp2, des2 = sift.detectAndCompute(img2, None)
#     if des1 is None or des2 is None:
#         return None, None
#     bf = cv2.BFMatcher(cv2.NORM_L2)
#     matches = bf.knnMatch(des1, des2, k=2)
#     good = [m for m, n in matches if m.distance < 0.75 * n.distance]
#     if len(good) < 10:
#         return None, None
#     src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
#     dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
#     return src, dst


# def blend_on_canvas(canvas, warped):
#     mask = (warped.sum(axis=2) > 0).astype(np.uint8)
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
#     mask_eroded = cv2.erode(mask, kernel, iterations=1)
#     dist = cv2.distanceTransform(mask_eroded, cv2.DIST_L2, 5)
#     alpha = dist / dist.max() if dist.max() > 0 else dist
#     alpha = np.clip(alpha, 0, 1)
#     alpha = cv2.GaussianBlur(alpha, (21,21), 0)[..., np.newaxis]
#     return (canvas * (1 - alpha) + warped * alpha).astype(np.uint8)


# def update_corners(corners, H):
#     new = []
#     for x, y in corners:
#         v = np.array([x, y, 1.0])
#         vt = H @ v
#         vt /= vt[2]
#         new.append((vt[0], vt[1]))
#     return new


# def crop_canvas(canvas, all_corners):
#     xs = [p[0] for p in all_corners]
#     ys = [p[1] for p in all_corners]
#     min_x = int(max(min(xs), 0)); max_x = int(min(max(xs), canvas.shape[1]))
#     min_y = int(max(min(ys), 0)); max_y = int(min(max(ys), canvas.shape[0]))
#     return canvas[min_y:max_y, min_x:max_x]


# def create_panorama_stitcher(frame_files, suffix=""):
#     imgs = []
#     for p in frame_files:
#         im = load_and_resize(p, FRAME_SCALE)
#         if im is not None:
#             imgs.append(im)
#     if len(imgs) < 2:
#         print("❌ Not enough frames for Stitcher.")
#         return None
#     stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
#     status, pano = stitcher.stitch(imgs)
#     if status != cv2.Stitcher_OK:
#         print(f"⚠️ Stitcher failed ({status}) for {suffix}")
#         return None
#     out_path = f"{OUTPUT_DIR}/panorama_{video_name}_{suffix}_stitcher.jpg"
#     cv2.imwrite(out_path, pano)
#     print(f"✅ Saved Stitcher fallback {suffix} to {out_path}")
#     return pano


# def merge_panoramas(panoramas):
#     if not panoramas:
#         print("❌ No panoramas to merge.")
#         return None

#     print(f"🔄 Merging {len(panoramas)} panoramas sequentially…")
#     merged = panoramas[0]
#     total = len(panoramas)

#     for idx, pano in enumerate(panoramas[1:], start=1):
#         step_start = time.time()
#         print(f"   · Stitching panorama {idx+1}/{total}")

#         stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
#         status, result = stitcher.stitch([merged, pano])

#         if status == cv2.Stitcher_OK:
#             merged = result
#         else:
#             print(f"⚠️  Stitcher failed on pano {idx+1}, falling back to manual merge…")
#             src, dst = detect_and_match_sift(merged, pano)
#             if src is None:
#                 src, dst = detect_and_match_orb(merged, pano)
#             if src is None:
#                 src, dst = detect_and_match_akaze(merged, pano)
#             if src is None:
#                 print(f"❌ All matchers failed to merge pano {idx+1}, skipping.")
#                 continue
#             H, _ = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)
#             if H is None:
#                 print(f"❌ Homography estimation failed for pano {idx+1}, skipping.")
#                 continue
#             h1, w1 = merged.shape[:2]
#             h2, w2 = pano.shape[:2]
#             out_w, out_h = max(w1, w2) * 2, max(h1, h2) * 2
#             warped_pano = cv2.warpPerspective(pano, H, (out_w, out_h))
#             canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
#             canvas[:h1, :w1] = merged
#             merged = blend_on_canvas(canvas, warped_pano)

#         step_end = time.time()
#         print(f"⏱️  Panorama {idx+1} stitched in {step_end - step_start:.2f} sec")

#     out_path = f"{OUTPUT_DIR}/panorama_{video_name}_final.jpg"
#     cv2.imwrite(out_path, merged)
#     print(f"✅ Saved final merged panorama to {out_path}")
#     return merged


# def main():
#     total_start = time.time()
#     print(f"🚀 Starting panorama stitching for '{video_name}'")
#     panoramas = []

#     clips = [f"{FRAME_DIR}/clip{i}" for i in range(1, 5)]
#     total_clips = len(clips)

#     for clip_idx, path in enumerate(clips, start=1):
#         clip_start = time.time()
#         if not os.path.isdir(path):
#             print(f"⚠️ Clip path {path} not found, skipping.")
#             continue

#         frames = glob.glob(f"{path}/frame_*.png")
#         frames = frames[::3]
#         print(f"📂 Found {len(frames)} frames in {path}")

#         pano = create_panorama_stitcher(frames, suffix=f"clip{clip_idx}")
#         if pano is not None:
#             panoramas.append(pano)

#         clip_end = time.time()
#         elapsed = clip_end - clip_start
#         avg_time = (clip_end - total_start) / clip_idx
#         est_total = avg_time * total_clips
#         est_remaining = est_total - (clip_end - total_start)
#         print(f"⏱️ Clip {clip_idx}/{total_clips} done in {elapsed:.2f} sec; estimated {est_remaining:.2f} sec remaining")

#     if panoramas:
#         merge_start = time.time()
#         merge_panoramas(panoramas)
#         merge_end = time.time()
#         print(f"⏱️ Merging panoramas took {merge_end - merge_start:.2f} sec")
#     else:
#         print("❌ No panoramas were created.")

#     total_end = time.time()
#     total_elapsed = total_end - total_start
#     print(f"✅ Completed stitching workflow in {total_elapsed:.2f} seconds.")

# if __name__ == "__main__":
#     main()


# === Adder Merge Section ===
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
        "output/video12/panorama_video12_clip1_stitcher.jpg",
"output/video12/panorama_video12_clip2_stitcher.jpg",
"output/video12/panorama_video12_clip3_stitcher.jpg"
    ]
    
    # Choose your method: 'stitcher', 'sift', or 'orb'
    output_path = f"output/forest_stitched_panorama.jpg"
    method = "sift"
    
    try:
        result_path = stitch_images_with_opencv(image_paths, output_path, method)
        print(f"Stitched image saved to {result_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    # Step 1: Extract frames
    print("🚀 Starting frame extraction...")
    extract_all_videos()  # Assumes extract_frames.py has a function like this

    # Step 2: Process each clip into panorama
    print("🔧 Stitching panoramas for each clip...")
    panoramas = []
    for clip_folder in sorted(os.listdir('extracted_frames')):
        pano = process_clip_folder(clip_folder)  # Assumes mult_fallback has this function
        if pano is not None:
            panoramas.append(pano)
    
    if not panoramas:
        print("❌ No panoramas generated. Exiting.")
        exit(1)

    # Step 3: Merge all panoramas using adder logic
    print("🔗 Merging all panoramas using adder logic...")
    final_pano = final_merge_with_adder(panoramas)  # Assumes adder.py has this function

    output_path = 'output/final_combined_panorama.jpg'
    cv2.imwrite(output_path, final_pano)
    print(f"✅ Final panorama saved at {output_path}")
