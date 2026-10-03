import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def inspect_image(image_path: str) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    height, width, channels = img.shape
    return {
        "width": width, "height": height, "channels": channels,
        "shape": [height, width, channels],
        "pixel_count": height * width,
        "estimated_bytes": height * width * channels * 1,
        "color_order": "BGR"
    }

def create_pixel_views(image_path: str, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_bgr.shape[:2]
    new_w, new_h = w // 2, h // 2
    downsampled = cv2.resize(img_rgb, (new_w, new_h))

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes[0, 0].imshow(r, cmap='Reds'); axes[0, 0].set_title('Red Channel')
    axes[0, 1].imshow(g, cmap='Greens'); axes[0, 1].set_title('Green Channel')
    axes[0, 2].imshow(b, cmap='Blues'); axes[0, 2].set_title('Blue Channel')
    axes[1, 0].imshow(gray, cmap='gray'); axes[1, 0].set_title('Grayscale')
    axes[1, 1].imshow(downsampled); axes[1, 1].set_title('Downsampled (Half Size)')
    axes[1, 2].axis('off')
    for ax in axes.ravel(): ax.axis('off')
    plt.tight_layout()

    out_path = os.path.join(output_dir, "pixel_views.png")
    plt.savefig(out_path)
    plt.close(fig)
    return {"original_size": [w, h], "downsampled_size": [new_w, new_h], "output_path": out_path}

def create_adjustments(image_path: str, output_dir: str, brightness_delta: int = 40, contrast_factor: float = 1.5, threshold: int = 127) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    img_bgr = cv2.imread(image_path)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bright = np.clip(gray.astype(np.int16) + brightness_delta, 0, 255).astype(np.uint8)
    contrast = np.clip(gray.astype(np.float64) * contrast_factor, 0, 255).astype(np.uint8)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(bright, cmap='gray'); axes[0].set_title(f'Brighter (+{brightness_delta})')
    axes[1].imshow(contrast, cmap='gray'); axes[1].set_title(f'Contrast (x{contrast_factor})')
    axes[2].imshow(thresh, cmap='gray'); axes[2].set_title(f'Threshold ({threshold})')
    for ax in axes: ax.axis('off')
    plt.tight_layout()

    out_path = os.path.join(output_dir, "adjustments.png")
    plt.savefig(out_path)
    plt.close(fig)
    return {"brightness_delta": brightness_delta, "contrast_factor": contrast_factor, "threshold": threshold, "output_path": out_path}

def create_blur_and_edges(image_path: str, output_dir: str, kernel_size: int = 5) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    img_bgr = cv2.imread(image_path)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.blur(gray, (kernel_size, kernel_size))

    edges_orig = cv2.magnitude(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
    edges_blur = cv2.magnitude(cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].imshow(gray, cmap='gray'); axes[0, 0].set_title('Grayscale')
    axes[0, 1].imshow(blurred, cmap='gray'); axes[0, 1].set_title(f'Mean Blur ({kernel_size}x{kernel_size})')
    axes[1, 0].imshow(edges_orig, cmap='gray'); axes[1, 0].set_title('Sobel Edges (Original)')
    axes[1, 1].imshow(edges_blur, cmap='gray'); axes[1, 1].set_title('Sobel Edges (Blurred)')
    for ax in axes.ravel(): ax.axis('off')
    plt.tight_layout()

    out_path = os.path.join(output_dir, "blur_and_edges.png")
    plt.savefig(out_path)
    plt.close(fig)
    return {"kernel_size": kernel_size, "output_path": out_path}

def run_lab(image_path: str, output_dir: str) -> dict:
    return {
        "task1": inspect_image(image_path),
        "task2": create_pixel_views(image_path, output_dir),
        "task3": create_adjustments(image_path, output_dir),
        "task4": create_blur_and_edges(image_path, output_dir)
    }

def main() -> None:
    os.makedirs("outputs", exist_ok=True)
    run_lab("images/original.jpg", "outputs")

if __name__ == "__main__":
    main()