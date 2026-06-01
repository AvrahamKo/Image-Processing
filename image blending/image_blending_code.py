import cv2
import numpy as np
from matplotlib import pyplot as plt


def build_gaussian_pyramid(img, levels):
    """
    Build a Gaussian pyramid for a color or grayscale image.

    Each level is downsampled by a factor of two using cv2.pyrDown.

    Args:
        img (np.ndarray): Input image (color or grayscale).
        levels (int): Number of pyramid levels to construct.

    Returns:
        List[np.ndarray]: Gaussian pyramid from original size to smallest.
    """
    pyramid = [img]
    for i in range(levels):
        img = cv2.pyrDown(img)
        pyramid.append(img)
    return pyramid


def build_laplacian_pyramid(img, levels):
    """
    Build a Laplacian pyramid for a color or grayscale image.

    Each Laplacian level is the difference between a Gaussian level and
the expanded next Gaussian level. The final level is the smallest Gaussian.

    Args:
        img (np.ndarray): Input image (color or grayscale).
        levels (int): Number of pyramid levels to construct.

    Returns:
        List[np.ndarray]: Laplacian pyramid from largest to smallest.
    """
    gaussian_pyramid = build_gaussian_pyramid(img, levels)
    laplacian_pyramid = []
    for i in range(levels):
        next_gaussian_up = cv2.pyrUp(gaussian_pyramid[i+1])
        next_gaussian_up = cv2.resize(
            next_gaussian_up,
            (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
        )
        laplacian = cv2.subtract(gaussian_pyramid[i], next_gaussian_up)
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid


def reconstruct_from_laplacian_pyramid(laplacian_pyramid):
    """
    Reconstruct an image from its Laplacian pyramid.

    Starting from the smallest level, each level is expanded and added
to the next Laplacian level until the original size is restored.

    Args:
        laplacian_pyramid (List[np.ndarray]): Laplacian pyramid levels.

    Returns:
        np.ndarray: Reconstructed image at the original resolution.
    """
    image = laplacian_pyramid[-1]
    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        size = (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0])
        image_up = cv2.pyrUp(image)
        image_up = cv2.resize(image_up, size)
        image = cv2.add(laplacian_pyramid[i], image_up)
    return image


def normalize_mask(mask):
    """
    Normalize a mask image to the [0, 1] float range.

    Args:
        mask (np.ndarray): Single-channel 8-bit mask image.

    Returns:
        np.ndarray: Normalized mask as float32 in [0, 1].
    """
    mask_float = mask.astype(np.float32)
    mask_float /= 255.0
    return mask_float


def laplacian_blending(imgA, imgB, mask, levels=5):
    """
    Blend two images using Laplacian pyramids and a binary mask.

    Args:
        imgA (np.ndarray): First color image (foreground).
        imgB (np.ndarray): Second color image (background).
        mask (np.ndarray): Single-channel or BGR mask indicating blend regions.
        levels (int): Number of pyramid levels to use.

    Returns:
        np.ndarray: Blended color image.
    """
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    A_channels = cv2.split(imgA)
    B_channels = cv2.split(imgB)
    mask_gaussian_pyramid = build_gaussian_pyramid(normalize_mask(mask), levels)
    blended_channels = []
    for A_c, B_c in zip(A_channels, B_channels):
        LpA = build_laplacian_pyramid(A_c, levels)
        LpB = build_laplacian_pyramid(B_c, levels)
        Lp_blend = []
        for i in range(levels+1):
            mask_size = (LpA[i].shape[1], LpA[i].shape[0])
            mask_resized = cv2.resize(mask_gaussian_pyramid[i], mask_size)
            blended = LpB[i] * (1 - mask_resized) + LpA[i] * mask_resized
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            Lp_blend.append(blended)
        blended_channel = reconstruct_from_laplacian_pyramid(Lp_blend)
        blended_channels.append(blended_channel)
    blended_image = cv2.merge(blended_channels)
    return blended_image


def build_gaussian_pyramid_gray(img, levels):
    """
    Build a Gaussian pyramid for a grayscale image.

    Args:
        img (np.ndarray): Input grayscale image.
        levels (int): Number of pyramid levels to construct.

    Returns:
        List[np.ndarray]: Gaussian pyramid levels as float32 images.
    """
    pyramid = [img.astype(np.float32)]
    current = img.astype(np.float32)
    for _ in range(levels):
        current = cv2.pyrDown(current)
        pyramid.append(current)
    return pyramid


def build_laplacian_pyramid_gray(img, levels):
    """
    Build a Laplacian pyramid for a grayscale image using its Gaussian pyramid.

    Args:
        img (np.ndarray): Input grayscale image.
        levels (int): Number of pyramid levels to construct.

    Returns:
        List[np.ndarray]: Laplacian pyramid levels as float32 images.
    """
    gaussian_pyramid = build_gaussian_pyramid_gray(img, levels)
    laplacian_pyramid = []
    for i in range(levels):
        gauss_up = cv2.pyrUp(gaussian_pyramid[i + 1])
        gauss_up = cv2.resize(
            gauss_up,
            (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
        )
        lap = gaussian_pyramid[i] - gauss_up
        laplacian_pyramid.append(lap)
    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid


def reconstruct_from_laplacian_pyramid(laplacian_pyramid):
    """
    Reconstruct a grayscale image from its Laplacian pyramid.

    Args:
        laplacian_pyramid (List[np.ndarray]): Pyramid levels from highest to lowest.

    Returns:
        np.ndarray: Reconstructed image as uint8.
    """
    image = laplacian_pyramid[-1]
    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        size = (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0])
        image_up = cv2.pyrUp(image)
        image_up = cv2.resize(image_up, size)
        image = image_up + laplacian_pyramid[i]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def create_hybrid_image_laplacian_gray(imgA, imgB, levels=6):
    """
    Create a hybrid grayscale image by combining low and high frequencies via Laplacian pyramids.

    High-frequency details come from imgA, low-frequency content from imgB.

    Args:
        imgA (np.ndarray): Grayscale image providing high frequencies.
        imgB (np.ndarray): Grayscale image providing low frequencies.
        levels (int): Number of pyramid levels to use.

    Returns:
        np.ndarray: Hybrid grayscale image.
    """
    LpA = build_laplacian_pyramid_gray(imgA, levels)
    LpB = build_laplacian_pyramid_gray(imgB, levels)
    hybrid_laplacian = []
    midpoint = levels // 2
    for i in range(levels + 1):
        hybrid_laplacian.append(LpA[i] if i < midpoint else LpB[i])
    hybrid_image = reconstruct_from_laplacian_pyramid(hybrid_laplacian)
    return hybrid_image


def display_gaussian_pyramid(image_path, levels):
    """
    Display a Gaussian pyramid of an image using matplotlib.

    Args:
        image_path (str): Path to the input image file.
        levels (int): Number of pyramid levels to display.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load the image.")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gaussian_pyramid = [image]
    for _ in range(levels - 1):
        image = cv2.pyrDown(gaussian_pyramid[-1])
        gaussian_pyramid.append(image)
    plt.figure(figsize=(15, 5))
    for i, img in enumerate(gaussian_pyramid):
        plt.subplot(1, levels, i + 1)
        plt.imshow(img)
        plt.title(f"Level {i}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def display_laplacian_pyramid(image_path, levels):
    """
    Display a Laplacian pyramid of an image using matplotlib.

    Args:
        image_path (str): Path to the input image file.
        levels (int): Number of pyramid levels to display.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load the image.")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gaussian_pyramid = [image]
    for _ in range(levels - 1):
        image = cv2.pyrDown(gaussian_pyramid[-1])
        gaussian_pyramid.append(image)
    laplacian_pyramid = []
    for i in range(levels - 1):
        size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyramid[i], gaussian_expanded)
        laplacian = np.clip(laplacian, 0, 255)
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1])
    plt.figure(figsize=(15, 5))
    for i, img in enumerate(laplacian_pyramid):
        plt.subplot(1, levels, i + 1)
        plt.imshow(img.astype(np.uint8))
        plt.title(f"Level {i}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load two color images and the mask (same size)
    imgA = cv2.imread('image1.jpg')  # First image
    imgB = cv2.imread('image2.jpg')  # Second image
    mask = cv2.imread('mask.jpg', cv2.IMREAD_GRAYSCALE)  # Black-white mask

    # Number of pyramid levels
    levels = 5

    # Perform blending
    blended_result = laplacian_blending(imgA, imgB, mask, levels)

    # Save or display the result
    cv2.imwrite('blended_result.jpg', blended_result)
    # cv2.imshow('Blended Result', blended_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 1. Read grayscale images
    imgA = cv2.imread('image_close.jpg', cv2.IMREAD_GRAYSCALE)  # Image to be seen up close
    imgB = cv2.imread('image_far.jpg', cv2.IMREAD_GRAYSCALE)    # Image to be seen from far away

    # 2. Create hybrid image using Laplacian pyramid
    levels = 6
    hybrid_img = create_hybrid_image_laplacian_gray(imgA, imgB, levels)

    # 3. Save and display result
    cv2.imwrite('hybrid_result_gray_laplacian.jpg', hybrid_img)
    # cv2.imshow('Hybrid Image (Laplacian, Gray)', hybrid_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    image_path = 'image1.jpg'  # Replace with the path to your image
    display_gaussian_pyramid(image_path, levels=5)

    image_path = 'image1.jpg'  # Replace with the path to your image
    display_laplacian_pyramid(image_path, levels=5)
