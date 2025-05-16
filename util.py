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
import cv2
import numpy as np


def visualize_sift_keypoints_with_orientations(image, keypoints, radius=10, color=(0, 0, 255)):
    """
    Visualize SIFT keypoints with red circles and orientation arrows.
    
    Args:
        image: Input image (BGR or grayscale)
        keypoints: List of cv2.KeyPoint objects
        radius: Radius of keypoint circle
        color: BGR color for keypoints (default: red)
        
    Returns:
        vis_image: Image with keypoints visualized
    """
    if len(image.shape) == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_image = image.copy()

    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        angle = np.deg2rad(kp.angle)
        length = int(radius * 2)

        # Draw circle
        cv2.circle(vis_image, (x, y), radius, color, 1, cv2.LINE_AA)

        # Draw orientation arrow
        dx = int(length * np.cos(angle))
        dy = int(length * np.sin(angle))
        cv2.arrowedLine(vis_image, (x, y), (x + dx, y + dy), color, 1, cv2.LINE_AA, tipLength=0.3)

    return vis_image



def visualize_sift_matches(img1, kp1, img2, kp2, matches, name1="img1", name2="img2", num_matches=50):
    """
    Visualize SIFT feature matches between two images.

    Args:
        img1: First image (grayscale or BGR)
        kp1: Keypoints from image 1
        img2: Second image
        kp2: Keypoints from image 2
        matches: List of cv2.DMatch objects
        name1: Label or filename for image 1
        name2: Label or filename for image 2
        num_matches: Number of top matches to draw
    """
    if not matches:
        print(f"No matches between {name1} and {name2}.")
        return

    matches_to_draw = matches[:num_matches]
    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, matches_to_draw, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    window_name = f"Matches: {name1} <-> {name2}"
    cv2.imshow(window_name, img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
