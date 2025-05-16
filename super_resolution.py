import cv2
import numpy as np
import glob
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import time

from util import rgb_to_ycbcr, show_ycbcr_channels, ycbcr_to_rgb

FRAME_DIR = "extracted_frames/video_6_3_every_1"
FRAME_SCALE = 1.0


# ========= FRAME EXTRACTION =========
def extract_frames(video_path, output_dir, frame_rate=1):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if int(cap.get(1)) % frame_rate == 0:
            frame_path = os.path.join(output_dir, f"frame_{count:03d}.jpg")
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            print(f"📸 Extracted frame {count}")
            count += 1
    cap.release()
    print(f"✅ Done extracting {count} frames.")

def load_frame(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    Y, Cb, Cr = rgb_to_ycbcr(img)
    return Y, Cb, Cr

def align_frames_ecc(reference, y_frames, ecc_threshold=0.80):
# Aligns and filters Y-channel frames based on ECC threshold.
# Args:
#     reference: reference frame (Y[0]) as float32 (H, W)
#     frames: list of Y frames [Y[1], Y[2], ..., Y[N]]
#     ecc_thresh: minimum ECC value to keep the frame
#     max_frames: maximum number of frames to keep
# Returns:
#     aligned_frames: list of aligned Y-channel frames (including reference)

    aligned_frames = [reference]
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6)

    for i, frame in enumerate(y_frames):

        # Downscale for alignment
        scale = 0.25
        small_ref = cv2.resize(reference, (0, 0), fx=scale, fy=scale)
        small_input = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        ref_blur = cv2.GaussianBlur(small_ref, (5, 5), 1.5)
        frame_blur = cv2.GaussianBlur(small_input, (5, 5), 1.5) 

        # warping
        warp_matrix = np.eye(2, 3, dtype=np.float32)

        try:
            cc, warp_matrix = cv2.findTransformECC(ref_blur, frame_blur, warp_matrix,
                                                   motionType=cv2.MOTION_AFFINE,
                                                   criteria=criteria)
            
            # Scale warp back to original resolution
            warp_matrix[0, 2] /= scale
            warp_matrix[1, 2] /= scale

            aligned = cv2.warpAffine(frame, warp_matrix,
                                     (reference.shape[1], reference.shape[0]),
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

            ecc_score = cv2.computeECC(reference, aligned)
            print(f"ECC: {ecc_score:.4f}")

            if ecc_score > ecc_threshold:
                aligned_frames.append(aligned)

            else:
                print(f"⚠️ Frame {i+1} rejected (ECC = {ecc_score:.3f})")
                break

        except cv2.error as e:
            print(f"❌ Frame {i+1} alignment failed: {e}")

    return aligned_frames, i+1

def back_projection_sr_confidence(aligned_y, scale=2, num_iters=5):
    base_lr = aligned_y[0]
    h, w = base_lr.shape

    hr = cv2.resize(base_lr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # Confidence map: count of valid contributions
    confidence = np.zeros_like(hr, dtype=np.float32)

    for it in range(num_iters):
        correction = np.zeros_like(hr, dtype=np.float32)
        confidence[:] = 0  # reset confidence for this iteration

        for i, lr in enumerate(aligned_y):
            simulated_lr = cv2.resize(hr, (w, h), interpolation=cv2.INTER_LINEAR)
            residual = lr - simulated_lr
            residual_up = cv2.resize(residual, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)

            # Where is this frame contributing? (not black)
            mask = (lr > 1e-3).astype(np.float32)
            mask_up = cv2.resize(mask, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

            correction += residual_up 
            confidence += mask_up

        # Avoid divide-by-zero by clamping confidence
        safe_confidence = np.clip(confidence, 1e-3, None)
        hr += correction / safe_confidence

    hr = np.clip(hr, 0.0, 1.0)
    return hr, confidence

def back_projection_sr(aligned_y, scale=2, num_iters=5):
# Perform super-resolution on aligned Y frames using iterative back-projection.
# Args:
#     aligned_y: list of aligned Y-channel frames (float32, shape H×W)
#     scale: upscaling factor (e.g., 2)
#     num_iters: number of IBP refinement iterations

# Returns:
#     High-resolution Y image (float32, shape H*scale × W*scale)

    base_lr = aligned_y[0]
    h, w = base_lr.shape

    # Step 1: Initial HR guess from the reference
    hr = cv2.resize(base_lr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    for it in range(num_iters):
        correction = np.zeros_like(hr)

        for i, lr in enumerate(aligned_y):
            # Simulate LR from current HR guess
            simulated_lr = cv2.resize(hr, (w, h), interpolation=cv2.INTER_LINEAR)

            # Compute residual between real and simulated LR
            residual = lr - simulated_lr

            # Upsample residual and accumulate
            residual_up = cv2.resize(residual, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)
            correction += residual_up

        # Apply averaged correction
        hr += correction / len(aligned_y)

    # Clip to valid range [0, 1]
    hr = np.clip(hr, 0.0, 1.0)
    return hr

def crop_to_full_contribution_region(sr_y, confidence_maps, num_frames, shrink=20):
    """
    Crops sr_y to the smallest region where all pixels have full contribution (max confidence),
    with optional inward shrinking (negative padding).

    Args:
        sr_y (np.ndarray): High-resolution Y image.
        confidence_maps (np.ndarray): Raw confidence values.
        num_frames (int): Total number of input frames.
        shrink (int): How much to shrink the bounding box inward from each side.

    Returns:
        cropped (np.ndarray): Cropped high-confidence-only region.
        bbox (tuple): (x, y, w, h) of the final crop.
    """
    normalized_conf = confidence_maps / num_frames
    max_conf = normalized_conf.max()

    # Mask only the pixels with exact full contribution
    mask = (normalized_conf >= max_conf - 1e-4).astype(np.uint8)

    coords = cv2.findNonZero(mask)
    if coords is None:
        raise ValueError("No fully confident region found.")

    x, y, w, h = cv2.boundingRect(coords)

    # Shrink inward (safe crop inside the bbox)
    x_new = x + shrink
    y_new = y + shrink
    x_end = x + w - shrink
    y_end = y + h - shrink

    if x_end <= x_new or y_end <= y_new:
        raise ValueError("Shrink value too large: resulting box has zero or negative size.")

    cropped = sr_y[y_new:y_end, x_new:x_end]
    return cropped, (x_new, y_new, x_end - x_new, y_end - y_new)

def show_confidence_with_crop(conf_map, x, y, w, h):
    """
    Display normalized confidence map with a visible crop box using NumPy.
    Args:
        conf_map (np.ndarray): Normalized confidence map [0, 1].
        x, y, w, h (int): Crop box coordinates.
    """
    # Create RGB heatmap from confidence map
    norm_conf_rgb = plt.cm.hot(conf_map)[:, :, :3]  # shape H×W×3, RGB

    # Draw rectangle using numpy (set border pixels to a cyan color)
    border_color = [0.0, 1.0, 1.0]  # cyan

    norm_conf_rgb[y:y+1, x:x+w] = border_color      # top
    norm_conf_rgb[y+h-1:y+h, x:x+w] = border_color  # bottom
    norm_conf_rgb[y:y+h, x:x+1] = border_color      # left
    norm_conf_rgb[y:y+h, x+w-1:x+w] = border_color  # right

    plt.imshow(norm_conf_rgb)
    plt.title("Normalized Confidence Map with Crop Box")
    plt.axis("off")
    plt.show()

# ========= YCbCr Loading =========
def load_frame(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    Y, Cb, Cr = rgb_to_ycbcr(img)
    return Y, Cb, Cr

def main():
    frame_files = sorted(glob.glob(f"{FRAME_DIR}/frame_*.jpg"))
    print(f"📂 Found {len(frame_files)} frames.")

    if len(frame_files) < 2:
        print("Not enough frames to stitch.")
        return

    print("🔄 Loading and converting images...")
    ys, cbs, crs = [], [], []

    for file in tqdm(frame_files):
        y, cb, cr = load_frame(file)
        ys.append(y)
        cbs.append(cb)
        crs.append(cr)

    print(f"Loaded {len(ys)} YCbCr frames.")
    
    base_frame = 0
    batch = 0

    while (base_frame < len(ys) - 4):
        start = time.time()

        aligned_y, num = align_frames_ecc(ys[base_frame], ys[base_frame+1:])
        print(f"Batch {batch}: {len(aligned_y)} frames aligned. New Base Frame: {num+base_frame}")
        
        sr_y, confidence_maps = back_projection_sr_confidence(aligned_y, scale=2, num_iters=5)
        cropped_y, (x, y, w, h) = crop_to_full_contribution_region(sr_y, confidence_maps, num_frames=len(aligned_y))
       

        cb_ref = cbs[base_frame]
        cr_ref = crs[base_frame]

        # Resize Cb and Cr to match full sr_y first
        sr_cb_full = cv2.resize(cb_ref, sr_y.shape[::-1], interpolation=cv2.INTER_CUBIC)
        sr_cr_full = cv2.resize(cr_ref, sr_y.shape[::-1], interpolation=cv2.INTER_CUBIC)

        # Then crop to match cropped_y
        sr_cb = sr_cb_full[y:y+h, x:x+w]
        sr_cr = sr_cr_full[y:y+h, x:x+w]

        # sr_rgb = ycbcr_to_rgb(sr_y, sr_cb, sr_cr)
        sr_rgb = ycbcr_to_rgb(cropped_y, sr_cb, sr_cr)

        sr_rgb_8bit = (sr_rgb * 255.0).astype(np.uint8)
        os.makedirs(f"output/CONFIDENCE_HR_{FRAME_DIR}", exist_ok=True)
        cv2.imwrite(f"output/CONFIDENCE_HR_{FRAME_DIR}/sr_rgb_{batch:03d}.png", cv2.cvtColor(sr_rgb_8bit, cv2.COLOR_RGB2BGR))

        print(f"     output/HR_{FRAME_DIR}/sr_rgb_{batch}.png")
        end = time.time()
        print(f"     {end - start} Seconds")

        base_frame = base_frame + num
        batch += 1

if __name__ == "__main__":
    main()
