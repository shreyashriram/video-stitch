import cv2
import numpy as np
import glob
import os
from tqdm import tqdm
import time

video_name = "video11"
FRAME_DIR = f"extracted_frames/{video_name}"
FRAME_SCALE = 0.5
OUTPUT_DIR = f"output/{video_name}"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_resize(path, scale=1.0):
    img = cv2.imread(path)
    if img is None:
        print(f"⚠️ Failed to load image at {path}")
        return None
    return cv2.resize(img, (0, 0), fx=scale, fy=scale)

# Matching functions

def detect_and_match_orb(img1, img2):
    orb = cv2.ORB_create(3000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return None, None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 10:
        return None, None
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    return src, dst

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

# stitch panoramas into a super high quality panorama using sift
# def merge_panoramas(panoramas):
#     if len(panoramas) < 2:
#         print("❌ Not enough panoramas to merge.")
#         return None
#     base = panoramas[0]
#     h0, w0 = base.shape[:2]
#     ch, cw = h0 * 6, w0 * 10
#     canvas = np.zeros((ch, cw, 3), dtype=np.uint8)
#     x_off, y_off = w0 * 4, h0 * 2
#     H_acc = np.eye(3); H_acc[0,2], H_acc[1,2] = x_off, y_off
#     canvas[y_off:y_off+h0, x_off:x_off+w0] = base
#     corners = [(x_off, y_off), (x_off+w0, y_off), (x_off, y_off+h0), (x_off+w0, y_off+h0)]
#     prev = base.copy()
#     for i, pano in enumerate(panoramas[1:], 1):
#         src, dst = detect_and_match_sift(prev, pano)
#         if src is None:
#             print(f"⚠️ SIFT failed for panorama {i}, skipping.")
#             continue
#         H, _ = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)
#         if H is None:
#             print(f"⚠️ Homography failed for panorama {i}")
#             continue
#         H_acc = H_acc @ H
#         warped = cv2.warpPerspective(pano, H_acc, (cw, ch))
#         canvas = blend_on_canvas(canvas, warped)
#         corners += update_corners([(0,0),(pano.shape[1],0),(0,pano.shape[0]),(pano.shape[1],pano.shape[0])], H_acc)
#         prev = pano
#     result = crop_canvas(canvas, corners)
#     out_path = f"{OUTPUT_DIR}/panorama_{video_name}_merged_sift.jpg"
#     cv2.imwrite(out_path, result)
#     print(f"✅ Saved merged panorama to {out_path}")
#     return result
# replace your existing merge_panoramas() with this:
def merge_panoramas(panoramas):
    """
    Sequentially stitch each clip-panorama into one final panorama.
    """
    if not panoramas:
        print("❌ No panoramas to merge.")
        return None

    print(f"🔄 Merging {len(panoramas)} panoramas sequentially…")
    # Start with the first clip’s panorama
    merged = panoramas[0]

    # Stitch each subsequent panorama into `merged`
    for idx, pano in enumerate(panoramas[1:], start=1):
        print(f"   · Stitching panorama {idx+1} into the final image")
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        status, result = stitcher.stitch([merged, pano])
        if status != cv2.Stitcher_OK:
            print(f"⚠️  Stitcher failed on pano {idx+1} (status={status}), skipping it.")
            continue
        merged = result

    # Save out the fully merged panorama
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
        # pano = create_panorama_stitcher(frames, suffix=f"clip{clip}")
        # for using custom stitching
        pano = create_panorama(frames, suffix=f"clip{clip}")
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