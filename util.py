import cv2
import matplotlib.pyplot as plt

import numpy as np

def show_ycbcr_channels(ys, cbs, crs, index=0):
    """
    Display the Y, Cb, and Cr channels of a given frame index.
    
    Args:
        ys, cbs, crs: Lists of (H, W) float32 images
        index: Index of the frame to display
    """
    if index < 0 or index >= len(ys):
        print(f"❌ Index {index} out of range.")
        return

    y, cb, cr = ys[index], cbs[index], crs[index]

    plt.figure(figsize=(12, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(y, cmap='gray')
    plt.title(f"Y (Luminance) | Frame {index}")

    plt.subplot(1, 3, 2)
    plt.imshow(cb, cmap='gray')
    plt.title("Cb (Blue-diff)")

    plt.subplot(1, 3, 3)
    plt.imshow(cr, cmap='gray')
    plt.title("Cr (Red-diff)")

    plt.tight_layout()
    plt.show()

def rgb_to_ycbcr(image_rgb):
    R = image_rgb[:, :, 0]
    G = image_rgb[:, :, 1]
    B = image_rgb[:, :, 2]

    Y  = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.169 * R - 0.331 * G + 0.5 * B + 0.5
    Cr = 0.5 * R - 0.419 * G - 0.081 * B + 0.5

    return Y, Cb, Cr

def ycbcr_to_rgb(Y, Cb, Cr):
    """
    Convert YCbCr channels to RGB.
    Inputs:
        Y, Cb, Cr: float32 arrays, shape (H, W), values in [0, 1]
    Returns:
        RGB image: float32 array, shape (H, W, 3), values in [0, 1]
    """
    Cb_shift = Cb - 0.5
    Cr_shift = Cr - 0.5

    R = Y + 1.402 * Cr_shift
    G = Y - 0.344136 * Cb_shift - 0.714136 * Cr_shift
    B = Y + 1.772 * Cb_shift

    rgb = np.stack([R, G, B], axis=2)
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb