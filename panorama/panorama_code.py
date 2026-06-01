import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_sift_matches_from_video(video_path, frame_idx1=0, frame_idx2=9, ratio=0.75, min_matches=4):
    """
    Loads a video, extracts two frames by index, and visualizes the SIFT matches between them.

    :param video_path: Path to the video file
    :param frame_idx1: Index of the first frame (default: 0)
    :param frame_idx2: Index of the second frame (default: 9)
    :param ratio: Lowe's ratio test threshold
    :param min_matches: Minimum number of matches to draw
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    # Get first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx1)
    ret1, img1 = cap.read()
    if not ret1:
        cap.release()
        raise IOError(f"Cannot read frame at index {frame_idx1}")

    # Get second frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx2)
    ret2, img2 = cap.read()
    cap.release()
    if not ret2:
        raise IOError(f"Cannot read frame at index {frame_idx2}")

    # Convert to grayscale and use SIFT
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=200)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # BFMatcher + Lowe’s ratio test
    bf = cv2.BFMatcher()
    knn_matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in knn_matches if m.distance < ratio * n.distance]
    if len(good) < min_matches:
        print(f"Only {len(good)} good matches found (minimum {min_matches})")

    # Draw the matches
    matched_img = cv2.drawMatches(
        img1, kp1, img2, kp2, good, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Show with matplotlib
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"{len(good)} good matches between {frame_idx1} and {frame_idx2}")
    plt.show()


def compute_direction_between_frames(gray1, gray2, ratio=0.75, min_matches=4, ransac_thresh=3.0):
    # Step 1: SIFT + ratio test
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    if des1 is None or des2 is None:
        return None, None

    bf = cv2.BFMatcher()
    knn = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    if len(good) < min_matches:
        return None, None

    # Step 2: Build arrays of points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    # Step 3: Compute affine with RANSAC
    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts,
                                             method=cv2.RANSAC,
                                             ransacReprojThreshold=ransac_thresh)
    if M is None:
        return None, None

    # Extract translation in x and y from M (M is 2x3)
    dx = M[0,2]
    dy = M[1,2]
    return dx, dy

def detect_camera_motion_direction(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video {video_path}")
        return None

    # Read first frame
    ret, f1 = cap.read()
    if not ret:
        print("Error: cannot read first frame")
        cap.release()
        return None
    gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)

    # Read the 10th frame (index 9)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
    ret, f10 = cap.read()
    cap.release()
    if not ret:
        print("Error: cannot read 10th frame")
        return None
    gray10 = cv2.cvtColor(f10, cv2.COLOR_BGR2GRAY)

    dx, dy = compute_direction_between_frames(gray1, gray10)
    if dx is None:
        print("Not enough good matches found")
        return None

    # Reverse signs to reflect camera motion
    if abs(dx) > abs(dy):
        direction = "right" if dx < 0 else "left"
    else:
        direction = "up" if dy > 0 else "down"

    print(f"The dominant direction of camera movement is: {direction}")
    return direction

def process_and_save_frames(video_path, direction, output_folder):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video {video_path}")
        return

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # Select order and processing by direction
    if direction == "right":
        processed = frames
    elif direction == "left":
        processed = frames[::-1]
    elif direction == "up":
        processed = [cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE) for f in frames]
    elif direction == "down":
        processed = [cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE) for f in frames]
    else:
        print(f"Unknown direction: {direction}")
        return

    os.makedirs(output_folder, exist_ok=True)
    for idx, frame in enumerate(processed, start=1):
        filename = os.path.join(output_folder, f"frame_{idx:05d}.png")
        cv2.imwrite(filename, frame)

    print(f"Saved {len(processed)} frames to '{output_folder}'")

def apply_rotation_only(img, M):
    """Applies only rotation around the image center, without translation."""
    h, w = img.shape[:2]
    center = (w / 2, h / 2)

    # Expand M to 3x3
    M_full = np.vstack([M, [0, 0, 1]])

    # Extract rotation components only (no translation)
    R = np.eye(3)
    R[0:2, 0:2] = M[0:2, 0:2]  # keep only the rotation

    # Rotate around center
    T1 = np.array([[1, 0, -center[0]],
                   [0, 1, -center[1]],
                   [0, 0, 1]])
    T2 = np.array([[1, 0, center[0]],
                   [0, 1, center[1]],
                   [0, 0, 1]])

    R_centered = T2 @ R @ T1

    # Apply rotation only
    rotated = cv2.warpAffine(img, R_centered[:2, :], (w, h), flags=cv2.INTER_LINEAR)
    return rotated

def align_images_sequence_rigid_centered(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if len(files) < 2:
        print("At least two images are required.")
        return

    sift = cv2.SIFT_create(nfeatures=200)
    bf = cv2.BFMatcher()

    # Read first image
    prev_img = cv2.imread(os.path.join(input_dir, files[0]))
    if prev_img is None:
        print("Error reading the first image.")
        return

    h, w = prev_img.shape[:2]
    cv2.imwrite(os.path.join(output_dir, files[0]), prev_img)

    for i in range(1, len(files)):
        curr_img = cv2.imread(os.path.join(input_dir, files[i]))
        if curr_img is None:
            print(f"Error reading {files[i]}")
            continue

        gray_prev = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

        kp1, des1 = sift.detectAndCompute(gray_prev, None)
        kp2, des2 = sift.detectAndCompute(gray_curr, None)

        if des1 is None or des2 is None:
            print(f"No features found in {files[i]}")
            continue

        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.7 * n.distance]

        if len(good) < 4:
            print(f"Not enough matches for {files[i]}")
            continue

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])

        M, _ = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC)
        if M is None:
            print(f"Alignment failed for {files[i]}")
            continue

        aligned = apply_rotation_only(curr_img, M)
        cv2.imwrite(os.path.join(output_dir, files[i]), aligned)

        prev_img = aligned
        print(f"Aligned (rotation only): {files[i]}")

    print(f"Completed rotation-only alignment for all images and saved to {output_dir}")

def compute_vertical_shift(gray_ref, gray_to_align,
                           ratio=0.75, min_matches=4, ransac_thresh=3.0):
    sift = cv2.SIFT_create(nfeatures=200)
    kp1, des1 = sift.detectAndCompute(gray_ref, None)
    kp2, des2 = sift.detectAndCompute(gray_to_align, None)
    bf = cv2.BFMatcher()
    knn = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    if len(good) < min_matches:
        raise RuntimeError(f"only {len(good)} good matches")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts,
                                             method=cv2.RANSAC,
                                             ransacReprojThreshold=ransac_thresh)
    if M is None:
        raise RuntimeError("RANSAC failed to find a transform")

    # Y translation component
    dy = M[1,2]
    return float(dy)

def stabilize_vertical_from_images(input_dir, output_dir, ratio=0.75):
    os.makedirs(output_dir, exist_ok=True)

    files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if len(files) < 2:
        print("At least two images are required.")
        return

    # Read first frame
    prev_frame = cv2.imread(os.path.join(input_dir, files[0]))
    h, w = prev_frame.shape[:2]
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    out_path = os.path.join(output_dir, f"{0:06d}.png")
    cv2.imwrite(out_path, prev_frame)
    print(f"[0] saved unmodified → {out_path}")

    for idx in range(1, len(files)):
        frame = cv2.imread(os.path.join(input_dir, files[idx]))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        try:
            dy = compute_vertical_shift(prev_gray, gray, ratio)
        except RuntimeError as e:
            print(f"[{idx}] Warning: {e}; using dy=0")
            dy = 0.0

        M = np.array([[1, 0, 0], [0, 1, -dy]], dtype=np.float32)
        aligned = cv2.warpAffine(frame, M, (w, h))

        out_path = os.path.join(output_dir, f"{idx:06d}.png")
        cv2.imwrite(out_path, aligned)
        print(f"[{idx}] dy={dy:.1f}px → saved → {out_path}")

        prev_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

    print(f"Done: {len(files)} frames written to '{output_dir}'")

def align_sequence_rotate_then_vertical(input_dir, output_dir, ratio=0.75, min_matches=4):
    """
    For each frame in the sequence:
      1. Compute rigid transform (rotation + translation) with respect to previous frame
      2. Extract only the rotation and apply it around the center
      3. Compute the relative vertical shift between the rotated frame and the previous frame
      4. Apply translation in Y axis only
      5. Save the result to output_dir
    """
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(input_dir)
                   if f.lower().endswith(('.png','jpg','jpeg')))

    if len(files) < 2:
        print("At least two images are required.")
        return

    # Read the first frame and write directly
    prev_img = cv2.imread(os.path.join(input_dir, files[0]))
    h, w = prev_img.shape[:2]
    cv2.imwrite(os.path.join(output_dir, files[0]), prev_img)
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=200)
    bf = cv2.BFMatcher()

    for i in range(1, len(files)):
        fname = files[i]
        curr = cv2.imread(os.path.join(input_dir, fname))
        gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        # 1) Estimate rigid transform (rotation+translation)
        kp1, des1 = sift.detectAndCompute(prev_gray, None)
        kp2, des2 = sift.detectAndCompute(gray,     None)
        if des1 is None or des2 is None:
            print(f"[{i}] No features matched in {fname}, skipping.")
            continue
        matches = bf.knnMatch(des1, des2, k=2)
        good   = [m for m,n in matches if m.distance < ratio*n.distance]
        if len(good) < min_matches:
            print(f"[{i}] Not enough matches in {fname} ({len(good)})")
            continue
        src = np.float32([ kp1[m.queryIdx].pt for m in good ])
        dst = np.float32([ kp2[m.trainIdx].pt for m in good ])
        M, _ = cv2.estimateAffinePartial2D(dst, src, method=cv2.RANSAC)

        # 2) Rotation only around center
        rotated = apply_rotation_only(curr, M)

        # 3) Compute dy between rotated frame and previous frame
        gray_rot = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        try:
            dy = compute_vertical_shift(prev_gray, gray_rot, ratio, min_matches)
        except RuntimeError as e:
            print(f"[{i}] Warning: {e}; using dy=0")
            dy = 0.0

        # 4) Apply vertical translation only
        T = np.array([[1, 0,   0],
                      [0, 1, -dy]], dtype=np.float32)
        final = cv2.warpAffine(rotated, T, (w, h), flags=cv2.INTER_LINEAR)

        # 5) Save and update prev_gray
        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, final)
        print(f"[{i}] Rotated + Y-shifted={-dy:.1f}px → saved → {out_path}")

        prev_gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)

    print(f"Processed {len(files)} images, results in '{output_dir}'")

def regular_pushbroom(input_folder: str, slice_width: int, offset: int, output_path: str):
    """
    Creates a panorama image using the Pushbroom method from a folder of images.

    - input_folder: Folder containing source images (sorted by filename).
    - slice_width: Width of each strip (in pixels) to take from each image.
    - offset: Pixel offset from the image center (positive is right, negative is left).
    - output_path: Output file path for the panorama.
    """
    # Get all image files in the directory
    image_paths = sorted(glob.glob(os.path.join(input_folder, '*')))
    if not image_paths:
        raise ValueError(f"No images found in '{input_folder}'")

    # Load valid images
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: unable to read '{path}', skipping.")
        else:
            images.append(img)
    if not images:
        raise ValueError("No valid images loaded.")

    h, w = images[0].shape[:2]
    n = len(images)

    # Compute slice position
    center_x = w // 2
    x = center_x + offset
    if x < 0 or x + slice_width > w:
        raise ValueError(
            f"Slice region out of bounds: x={x}, width={slice_width}, image width={w}")

    # Create canvas
    canvas_width = slice_width * n
    panorama = np.zeros((h, canvas_width, 3), dtype=images[0].dtype)

    # Fill the canvas with strips
    for i, img in enumerate(images):
        strip = img[:, x:x + slice_width]
        if strip.shape[1] != slice_width:
            strip = cv2.resize(strip, (slice_width, h))
        panorama[:, i * slice_width:(i + 1) * slice_width] = strip

    # Save
    cv2.imwrite(output_path, panorama)
    print(f"Panorama saved to '{output_path}'")

def batch_pushbrooms(input_folder: str,
                     output_folder: str,
                     slice_width: int = 5,
                     start_offset: int = -200,
                     end_offset: int = 200,
                     step: int = 50):
    """
    Runs regular_pushbroom with a range of offsets and saves each panorama to the output folder.

    - input_folder: Source folder with images.
    - output_folder: Output folder for saving results.
    - slice_width: Width of each strip (pixels).
    - start_offset, end_offset, step: Range settings for offset.
    """
    os.makedirs(output_folder, exist_ok=True)
    offsets = list(range(start_offset, end_offset + 1, step))

    for idx, offset in enumerate(offsets, start=1):
        filename = f"panorama_{idx:03d}.png"
        output_path = os.path.join(output_folder, filename)
        print(f"[{idx}/{len(offsets)}] Offset={offset} -> Saving '{filename}'")
        regular_pushbroom(input_folder, slice_width, offset, output_path)

def compute_horizontal_shift(img1, img2,
                             ratio=0.75, min_matches=4,
                             nfeatures=500, ransac_thresh=3.0):
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < ratio * n.distance]
    if len(good) < min_matches:
        return 0.0

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts,
                                             method=cv2.RANSAC,
                                             ransacReprojThreshold=ransac_thresh)
    if M is None:
        return 0.0

    dx = M[0,2]
    return float(dx)

def create_smart_pushbroom(input_folder: str,
                           output_path: str,
                           min_width: int = 1,
                           max_width: int = 100,
                           center_offset: int = 0,
                           stitch_edges: bool = True,
                           start_index: int = 0,
                           step_size: int = 2,
                           sift_features: int = 500):
    """
    Creates a "smart" panorama using the Pushbroom method with dynamic strip width and edge stitching support.
    Uses SIFT to compute horizontal shift, with a limit on number of SIFT features.
    You can select the start index and step size (for even/odd or every Nth frame).

    Parameters:
      - input_folder: Source folder with images.
      - output_path: Output file path.
      - min_width, max_width: Range of strip widths.
      - center_offset: Offset from image center (pixels).
      - stitch_edges: Whether to stitch edges of the image.
      - start_index: 0 to start from first image, 1 from second, etc.
      - step_size: Step size (2 for every other, 5 for every fifth, etc.).
      - sift_features: Number of SIFT features.
    """
    all_paths = sorted(glob.glob(os.path.join(input_folder, '*')))
    image_paths = all_paths[start_index::step_size]
    images = [cv2.imread(p) for p in image_paths]
    images = [img for img in images if img is not None]
    if not images:
        raise ValueError(f"No valid images in folder (start_index={start_index}, step_size={step_size})")

    h, w = images[0].shape[:2]
    n = len(images)

    # Compute dx between adjacent images
    dxs = []
    for i in range(n - 1):
        dx = compute_horizontal_shift(images[i], images[i+1], nfeatures=sift_features)
        dxs.append(dx)
    dxs.append(dxs[-1])  # Keep length n

    # Map abs(dx) to strip width
    abs_dxs = np.abs(dxs)
    min_dx, max_dx = abs_dxs.min(), abs_dxs.max()
    widths = []
    for dx in abs_dxs:
        if max_dx > min_dx:
            norm = (dx - min_dx) / (max_dx - min_dx)
            strip_w = int(min_width + norm * (max_width - min_width))
        else:
            strip_w = min_width
        widths.append(max(min_width, strip_w))

    # Compute cut position for each strip
    base_center = w // 2 + center_offset
    x0_list = [
        max(0, min(base_center + int(dx/2) - strip_w//2, w - strip_w))
        for dx, strip_w in zip(dxs, widths)
    ]

    # Left/right edge widths for stitching
    if stitch_edges:
        left_width = x0_list[0]
        right_width = w - (x0_list[-1] + widths[-1])
    else:
        left_width = 0
        right_width = 0

    # Create canvas
    canvas_width = left_width + sum(widths) + right_width
    print(f"Canvas size: {canvas_width}×{h}, left_edge={left_width}, right_edge={right_width}, offset={center_offset}, stitch_edges={stitch_edges}")
    panorama = np.zeros((h, canvas_width, 3), dtype=images[0].dtype)

    # Stitch left edge
    if stitch_edges and left_width > 0:
        panorama[:, :left_width] = images[0][:, :left_width]

    # Place strips
    offset_x = left_width
    for img, x0, strip_w in zip(images, x0_list, widths):
        strip = img[:, x0:x0 + strip_w]
        if strip.shape[1] != strip_w:
            strip = cv2.resize(strip, (strip_w, h))
        panorama[:, offset_x:offset_x + strip_w] = strip
        offset_x += strip_w

    # Stitch right edge
    if stitch_edges and right_width > 0:
        start = left_width + sum(widths)
        panorama[:, start:start + right_width] = images[-1][:, x0_list[-1] + widths[-1]:]

    # Save
    cv2.imwrite(output_path, panorama)
    print(f"Smart panorama saved to '{output_path}'")

def rotate_images_in_folder_inplace(folder: str, direction: str):
    """
    Rotates all images in a folder in-place according to the direction:
    - If direction == "up": rotate 90° counterclockwise
    - If direction == "down": rotate 90° clockwise
    Images are overwritten in the folder.
    """
    for fname in sorted(os.listdir(folder)):
        path = os.path.join(folder, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: not a valid image at {path}, skipping.")
            continue

        if direction == "up":
            rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif direction == "down":
            rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        else:
            rotated = img

        cv2.imwrite(path, rotated)
        print(f"Rotated '{fname}' -> {direction}")

def create_video_from_images(input_folder: str,output_video_path: str,fps: int = 30):
    # Get all images in the folder sorted by name
    images = sorted([
        f for f in os.listdir(input_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    if not images:
        print("No images found in folder.")
        return

    # Read first image to determine size
    first_image = cv2.imread(os.path.join(input_folder, images[0]))
    height, width, _ = first_image.shape

    # Create video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for avi
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Make frame list: forward then backward
    full_sequence = (images + images[::-1])*4

    # Write all images to video
    for img_name in full_sequence:
        print(f"Adding image: {img_name}")
        img_path = os.path.join(input_folder, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Error loading: {img_path}")
            continue
        out.write(frame)

    out.release()
    print(f"Video saved as: {output_video_path}")

def blur_video(input_path: str, output_path: str, blur_ksize=(5,5), blur_type='gaussian'):
    """
    Blurs all video frames and exports a new video.

    Parameters:
    - input_path: Path to the input video.
    - output_path: Path to the output blurred video.
    - blur_ksize: Kernel size (width, height), must be odd numbers.
    - blur_type: 'gaussian', 'average', or 'median'.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Select blur type
        if blur_type == 'gaussian':
            blurred = cv2.GaussianBlur(frame, blur_ksize, 0)
        elif blur_type == 'average':
            blurred = cv2.blur(frame, blur_ksize)
        elif blur_type == 'median':
            # median blur requires odd kernel
            k = blur_ksize[0] if blur_ksize[0] == blur_ksize[1] else min(blur_ksize)
            if k % 2 == 0:
                k += 1
            blurred = cv2.medianBlur(frame, k)
        else:
            raise ValueError(f"Unknown blur_type: {blur_type}")

        out.write(blurred)

    cap.release()
    out.release()
    print(f"Blurred video saved to: {output_path}")

if __name__ == "__main__":
    # Set paths here
    rotation = True
    blur_type = False
    video_path = "Inputs/bad.mp4"
    # visualize_sift_matches_from_video(video_path)
    all_frames = "Inputs/all_frames_b"
    rotated_frames = "Inputs/r_frames_b"
    aligned_frames = "Inputs/a_frames_b"
    panorama_frames = "Inputs/panorama_frames_b"
    final_video = "Inputs/final_video_b.mp4"
    blur_video_f = "Inputs/blurred_video.mp4"

    direction = detect_camera_motion_direction(video_path)
    print("The direction is: ", direction)

    if direction:
        process_and_save_frames(video_path, direction, all_frames)
        print("The frames have been loaded!")
    if rotation:
        align_sequence_rotate_then_vertical(all_frames,aligned_frames)
        # align_images_sequence_rigid_centered(all_frames, rotated_frames)
        # print("The frames have been rotated!")
        # stabilize_vertical_from_images(rotated_frames, aligned_frames)
    else:
        stabilize_vertical_from_images(all_frames, aligned_frames)
    print("The frames have been stabled!")

    if not rotation:
        batch_pushbrooms(aligned_frames, panorama_frames)
    else:
        os.makedirs(panorama_frames, exist_ok=True)
        counter = 10
        # Tree - for center_offset in range(220, 261, 5):
        # view - for center_offset in range(-400, 400, 100):
        for center_offset in range(-400, 400, 100):
            filename = f"panorama_{counter:03d}.png"
            output_path = os.path.join(panorama_frames, filename)
            print(f"\n--- Iteration {counter}: offset={center_offset} ---")
            counter = counter+1
            create_smart_pushbroom(
                aligned_frames,
                output_path,
                min_width=2,
                max_width=8,
                center_offset=center_offset,
                start_index=0,
                stitch_edges = False,
                step_size=2,
                sift_features=500
            )
    if direction == "up" or direction == "down":
        rotate_images_in_folder_inplace(panorama_frames,direction)
    create_video_from_images(panorama_frames,final_video,fps=16)
    if blur_type:
        blur_video(final_video, blur_video_f, blur_ksize=(5, 5), blur_type='gaussian')
