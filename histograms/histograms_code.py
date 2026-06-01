import mediapy as media
import numpy as np
import cv2
import matplotlib.pyplot as plt


def convert_video_to_grayscale(input_video_path, video_type=None):
    """
    Convert a color video to a sequence of grayscale frames.

    Reads a video from disk, converts each RGB frame to grayscale using
    the standard luminance formula, and replicates the gray channel
    across three channels to maintain the original frame shape.

    Parameters
    ----------
    input_video_path : str
        Path to the input video file.
    video_type : str, optional
        (Unused) Type or format of the video, provided for compatibility
        or future extension.

    Returns
    -------
    List[np.ndarray]
        A list of grayscale frames, each as a 3-channel uint8 image.
    """
    frames = media.read_video(input_video_path)
    grayscale_frames = []
    rgb_to_gray_weights = np.array([0.299, 0.587, 0.114])

    for frame in frames:
        # Compute weighted sum over RGB channels to get a single gray channel
        gray_frame = np.dot(frame[..., :3], rgb_to_gray_weights)

        # Stack into three identical channels and cast to uint8
        gray_frame = np.stack([gray_frame] * 3, axis=-1).astype(np.uint8)
        grayscale_frames.append(gray_frame)

    return grayscale_frames


def compute_cumulative_histogram(frame):
    """
    Compute the normalized cumulative histogram of a single-channel image.

    Calculates the histogram of pixel intensities (0-255) for one channel,
    normalizes it to create a probability distribution, and then computes
    its cumulative sum.

    Parameters
    ----------
    frame : np.ndarray
        A single-channel (grayscale) image as a 2D array of uint8 values.

    Returns
    -------
    np.ndarray
        A 1D array of length 256 containing the cumulative distribution of pixel
        intensities.
    """
    # Compute histogram for intensity values 0-255
    hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
    hist = hist / hist.sum()  # Normalize to sum to 1

    # Compute cumulative distribution
    cumulative_hist = np.cumsum(hist)
    return cumulative_hist


def max_change_in_cumulative(gray_video):
    """
    Identify the two consecutive frames with the maximum change in
    cumulative histogram.

    Iterates through a sequence of grayscale frames, computes the
    cumulative histogram for each, and finds the pair of consecutive
    frames whose histograms differ the most in Euclidean norm.

    Parameters
    ----------
    gray_video : List[np.ndarray]
        A list of grayscale frames (3-channel images), where only the
        first channel is used for histogram computation.

    Returns
    -------
    tuple (int, int)
        A tuple containing the indices (i, i+1) of the two consecutive
        frames with the largest change in cumulative histogram.
    """
    max_frame = 0
    max_value = 0
    prev_frame = None

    for frame_idx, frame in enumerate(gray_video):
        if prev_frame is not None:
            # Use only one channel since all are identical
            cumulative_hist1 = compute_cumulative_histogram(prev_frame[..., 0])
            cumulative_hist2 = compute_cumulative_histogram(frame[..., 0])

            # Compute difference and its Euclidean norm
            hist_diff = cumulative_hist2 - cumulative_hist1
            norm = np.linalg.norm(hist_diff)

            # Update if this difference is the maximum seen so far
            if norm > max_value:
                max_value = norm
                max_frame = frame_idx - 1

        prev_frame = frame

    return (max_frame, max_frame + 1)


if __name__ == "__main__":
    # Example usage (requires a valid video path)
    video_path = "input.mp4"
    gray_seq = convert_video_to_grayscale(video_path)
    f1, f2 = max_change_in_cumulative(gray_seq)
    print(f"Max histogram change between frames {f1} and {f2}")
