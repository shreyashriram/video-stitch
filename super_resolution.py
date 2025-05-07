import cv2
import numpy as np
import glob
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from util import rgb_to_ycbcr, show_ycbcr_channels


FRAME_DIR = "extracted_frames/every_3_video1_max30"
FRAME_SCALE = 1.0


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
    

def main():
    frame_files = sorted(glob.glob(f"{FRAME_DIR}/frame_*.jpg"))
    print(f"📂 Found {len(frame_files)} frames.")

    if len(frame_files) < 2:
        print("❌ Not enough frames to stitch.")
        return

    print("🔄 Loading and converting images...")
    ys, cbs, crs = [], [], []

    for file in tqdm(frame_files):
        y, cb, cr = load_frame(file)
        ys.append(y)
        cbs.append(cb)
        crs.append(cr)

    print(f"✅ Loaded {len(ys)} YCbCr frames.")

    aligned_y, num = align_frames_ecc(ys[0], ys[1:])
    print(f"{len(aligned_y)} frames aligned.")

    sr_y = back_projection_sr(aligned_y, scale=5, num_iters=5)
    print("Original LR shape:", aligned_y[0].shape)
    print("SR Y shape:", sr_y.shape)

    # Step: Upsample Cb and Cr from reference
    cb_ref = cbs[0]
    cr_ref = crs[0]

    sr_cb = cv2.resize(cb_ref, sr_y.shape[::-1], interpolation=cv2.INTER_CUBIC)
    sr_cr = cv2.resize(cr_ref, sr_y.shape[::-1], interpolation=cv2.INTER_CUBIC)

    # Step: Convert to RGB
    from util import ycbcr_to_rgb
    sr_rgb = ycbcr_to_rgb(sr_y, sr_cb, sr_cr)

    # Step: Convert to 8-bit and save
    sr_rgb_8bit = (sr_rgb * 255.0).astype(np.uint8)
    os.makedirs("output", exist_ok=True)
    cv2.imwrite("output/sr_rgb_5.png", cv2.cvtColor(sr_rgb_8bit, cv2.COLOR_RGB2BGR))

    # Step: Show final image
    plt.imshow(sr_rgb)
    plt.title("Final Super-Resolved RGB Image")
    plt.axis('off')
    plt.show()

    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(cv2.resize(aligned_y[0], sr_y.shape[::-1], interpolation=cv2.INTER_CUBIC), cmap='gray')
    # plt.title("Bicubic Upscale (Baseline)")
    # plt.axis('off')

    # plt.subplot(1, 2, 2)
    # plt.imshow(sr_y, cmap='gray')
    # plt.title("Super-Resolved (Back-Projection)")
    # plt.axis('off')
    # plt.show()

    # plt.imshow(sr_y, cmap='gray')
    # plt.title("Super-Resolved Y Image (x2)")
    # plt.axis('off')
    # plt.show()

    # stack = np.stack(aligned_y, axis=0)  # shape: (N, H, W)
    # std_map = np.std(stack, axis=0)

    # print(f"{len(aligned_y)} frames aligned.")

    # plt.imshow(std_map, cmap='hot')
    # plt.title("Pixel-wise Std Dev (alignment jitter)")
    # plt.colorbar()
    # plt.show()

if __name__ == "__main__":
    main()
