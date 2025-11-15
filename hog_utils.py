import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
from skimage import exposure

from skimage import data, exposure

# ---------------------------------------------------------------
# HOG Utilities for Hand Detection
# ---------------------------------------------------------------

# Default HOG parameters (can be tweaked for your project)
DEFAULT_WIN_SIZE = (64, 64)  # detection window size
DEFAULT_BLOCK_SIZE = (16, 16)  # 2x2 cells
DEFAULT_BLOCK_STRIDE = (8, 8)  # slide block by 1 cell
DEFAULT_CELL_SIZE = (8, 8)  # cell size
DEFAULT_NBINS = 9  # 9 histogram bins


def create_hog(
        win_size=None,
        block_size=None,
        block_stride=None,
        cell_size=None,
        nbins=None
):
    """
    Creates and returns a configured HOG descriptor.

    All parameters are optional; if omitted, defaults are used.

    Parameters:
        win_size     (tuple): (width, height) of detection window
        block_size   (tuple): (width, height) of HOG block
        block_stride (tuple): (stride_x, stride_y) between blocks
        cell_size    (tuple): (width, height) of HOG cell
        nbins        (int)  : number of orientation bins

    Returns:
        hog (cv2.HOGDescriptor)
    """

    win_size = win_size or DEFAULT_WIN_SIZE
    block_size = block_size or DEFAULT_BLOCK_SIZE
    block_stride = block_stride or DEFAULT_BLOCK_STRIDE
    cell_size = cell_size or DEFAULT_CELL_SIZE
    nbins = nbins or DEFAULT_NBINS

    hog = cv2.HOGDescriptor(
        _winSize=win_size,
        _blockSize=block_size,
        _blockStride=block_stride,
        _cellSize=cell_size,
        _nbins=nbins
    )
    return hog


def preprocess_patch(patch, hog):
    """
    Ensures the input patch is resized and grayscale,
    using the window size from the given HOG descriptor.

    Parameters:
        patch (numpy array): BGR or grayscale image patch
        hog   (cv2.HOGDescriptor): descriptor whose winSize is used

    Returns:
        gray_resized (numpy array): processed patch ready for HOG
    """
    win_w, win_h = hog.winSize  # use the actual HOG window size

    # Resize patch to HOG window size
    resized = cv2.resize(patch, (win_w, win_h))

    # Convert to grayscale if needed
    if len(resized.shape) == 3:  # BGR image
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized

    return gray


def compute_hog(patch, hog=None):
    """
    Computes HOG features for a given image patch.

    Parameters:
        patch (numpy array): image patch
        hog   (cv2.HOGDescriptor): Optional; reuse a pre-created HOG object

    Returns:
        features (numpy array): flattened HOG feature vector
    """
    if hog is None:
        hog = create_hog()

    # Preprocess patch (resize + grayscale according to hog.winSize)
    gray = preprocess_patch(patch, hog)

    # Compute HOG descriptor
    descriptor = hog.compute(gray)

    # Flatten to 1D
    return descriptor.flatten()


def sliding_windows(image, hog=None, step=16):
    """
    Generates sliding window patches across the image.

    If hog is provided, uses hog.winSize; otherwise uses default.
    """
    if hog is not None:
        win_w, win_h = hog.winSize
    else:
        win_w, win_h = DEFAULT_WIN_SIZE

    H, W = image.shape[:2]

    for y in range(0, H - win_h + 1, step):
        for x in range(0, W - win_w + 1, step):
            patch = image[y:y + win_h, x:x + win_w]
            yield (x, y, patch)


if __name__ == "__main__":
    image_path = "hand.png"
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise SystemExit(f"Could not read {image_path}")

    # 1) custom HOG
    hog_cv = create_hog(
        win_size=(80, 160),
        block_size=(16, 16),
        block_stride=(8, 8),
        cell_size=(8, 8),
        nbins=9
    )

    # 2) features
    features = compute_hog(img_bgr, hog_cv)
    print("HOG feature length:", features.shape[0])

    # 3) histogram figure
    plt.figure(figsize=(7, 4))
    feat_norm = features / (np.max(features) + 1e-6)
    plt.plot(feat_norm, linewidth=1)
    plt.title("HOG Feature Histogram")
    plt.xlabel("Feature index")
    plt.ylabel("Normalized magnitude")
    plt.grid(True)
    plt.tight_layout()

    # 4) HOG visualization figure
    gray = preprocess_patch(img_bgr, hog_cv)

    cells_per_block = (
        hog_cv.blockSize[0] // hog_cv.cellSize[0],
        hog_cv.blockSize[1] // hog_cv.cellSize[1],
    )

    fd, hog_image = hog(
        gray,
        orientations=hog_cv.nbins,
        pixels_per_cell=hog_cv.cellSize,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        visualize=True,
        feature_vector=True,
        channel_axis=None,
    )

    hog_image_rescaled = exposure.rescale_intensity(
        hog_image, in_range=(0, hog_image.max())
    )

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    img_rgb_resized = cv2.cvtColor(
        cv2.resize(img_bgr, hog_cv.winSize), cv2.COLOR_BGR2RGB
    )
    plt.imshow(img_rgb_resized)
    plt.title("Input patch")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(hog_image_rescaled, cmap="gray")
    plt.title("HOG visualization")
    plt.axis("off")
    plt.tight_layout()

    # 5) show EVERYTHING once
    plt.show()
