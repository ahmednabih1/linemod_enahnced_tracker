"""
generate_training_images.py
────────────────────────────
Run this ONCE to generate rotated + augmented copies of your training images.
They are saved to the 'rotated_training/' folder next to your originals.

Augmentations applied per rotation:
  • Brightness variation  (dark / neutral / bright)
  • Contrast variation    (low / neutral / high)
  • Gaussian noise        (clean / light / heavy)
  • Gaussian blur         (sharp / slightly soft)

Total images = sources × angles × augmentation_variants

Usage:
    python generate_training_images.py
"""

import cv2
import numpy as np
import os
import json
import glob

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURE THESE
# ─────────────────────────────────────────────────────────────────────────────
SOURCE_IMAGES = [
    "00.png",
    "01.png",
    "02.png",
    "03.png",
    "04.png",
]  # images to rotate from
OUTPUT_FOLDER = "rotated_training"  # folder where augmented images are saved
ANGLE_STEP = 5  # degrees between rotations (5 = good coverage)
ANGLE_RANGE = (-180, 180)  # full rotation coverage
ROI_FILE = "saved_roi.json"  # shared with linemod_detector.py

# Augmentation toggles — set False to skip a group
AUGMENT_BRIGHTNESS = False
AUGMENT_CONTRAST = False
AUGMENT_NOISE = False
AUGMENT_BLUR = False
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
#  AUGMENTATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────


def adjust_brightness(img, factor):
    """
    Multiply pixel values by factor.
    factor < 1.0 → darker,  factor > 1.0 → brighter.
    Uses convertScaleAbs so values stay in [0, 255].
    """
    return cv2.convertScaleAbs(img, alpha=factor, beta=0)


def adjust_contrast(img, factor):
    """
    Scale contrast around the mid-grey point (128).
    factor < 1.0 → washed out,  factor > 1.0 → punchy.
    """
    return cv2.convertScaleAbs(img, alpha=factor, beta=128 * (1 - factor))


def add_gaussian_noise(img, sigma):
    """
    Add zero-mean Gaussian noise with standard deviation sigma.
    Simulates sensor noise and slight texture differences.
    """
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def apply_blur(img, ksize):
    """
    Gaussian blur with kernel size ksize (must be odd).
    Simulates slight focus differences between training and test shots.
    """
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def build_augmentation_variants(img):
    """
    Returns a list of (suffix, augmented_image) tuples for one base image.
    The base image itself (no augmentation) is always included first.
    """
    variants = [("base", img.copy())]

    if AUGMENT_BRIGHTNESS:
        variants.append(("bright_lo", adjust_brightness(img, 0.65)))
        variants.append(("bright_hi", adjust_brightness(img, 1.40)))

    if AUGMENT_CONTRAST:
        variants.append(("contrast_lo", adjust_contrast(img, 0.70)))
        variants.append(("contrast_hi", adjust_contrast(img, 1.40)))

    if AUGMENT_NOISE:
        variants.append(("noise_lt", add_gaussian_noise(img, sigma=8)))
        variants.append(("noise_hvy", add_gaussian_noise(img, sigma=20)))

    if AUGMENT_BLUR:
        variants.append(("blur", apply_blur(img, ksize=3)))

    return variants


# ─────────────────────────────────────────────────────────────────────────────
#  ROI HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def select_roi_once(image_path):
    """Opens the first image so you can draw the bounding box once."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load '{image_path}'")

    print(f"\n>>> A window will open. Draw a box around the object in '{image_path}'.")
    print("    Press SPACE or ENTER to confirm. Press C to cancel.")

    x, y, w, h = cv2.selectROI(
        "Draw ROI once — reused for all images",
        img,
        fromCenter=False,
        showCrosshair=True,
    )
    cv2.destroyAllWindows()

    if w == 0 or h == 0:
        raise ValueError("No ROI selected.")

    roi = (int(x), int(y), int(w), int(h))
    print(f"    ROI selected: x={x}, y={y}, w={w}, h={h}")
    return roi


def save_roi(roi, path=ROI_FILE):
    with open(path, "w") as f:
        json.dump(roi, f)
    print(f"💾 ROI saved to '{path}' — delete this file to redraw the box next time.")


def load_roi(path=ROI_FILE):
    if os.path.exists(path):
        with open(path) as f:
            roi = tuple(json.load(f))
        print(f"📂 Loaded saved ROI from '{path}': {roi}")
        return roi
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  CORE GENERATION
# ─────────────────────────────────────────────────────────────────────────────


def generate_augmented_images(image_path, roi, output_folder, angle_step, angle_range):
    """
    For each angle:
      1. Rotate the source image.
      2. Apply every augmentation variant to the rotated image.
      3. Save each variant as a separate file.

    File naming:
      {base}_{angle}_{augmentation_suffix}.png
      e.g.  00_angle+015_bright_hi.png
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"  ⚠️  Could not load '{image_path}' — skipping.")
        return []

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    base = os.path.splitext(os.path.basename(image_path))[0]

    os.makedirs(output_folder, exist_ok=True)
    saved = []

    angles = range(angle_range[0], angle_range[1] + 1, angle_step)

    for angle in angles:
        # ── rotate ──────────────────────────────────────────────────────────
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            img,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # ── augment ─────────────────────────────────────────────────────────
        for suffix, aug_img in build_augmentation_variants(rotated):
            filename = f"{base}_angle{angle:+04d}_{suffix}.png"
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, aug_img)
            saved.append(output_path)

    n_angles = len(list(angles))
    n_aug = len(build_augmentation_variants(img))  # count on a dummy image
    print(
        f"  ✅ '{image_path}'  →  {n_angles} angles × {n_aug} variants "
        f"= {len(saved)} images  →  '{output_folder}/'"
    )
    return saved


# ─────────────────────────────────────────────────────────────────────────────
#  PREVIEW
# ─────────────────────────────────────────────────────────────────────────────


def preview_sample(output_folder, n=6):
    """
    Shows n evenly-spaced sample images so you can verify augmentations.
    Tries to pick one from each augmentation type for variety.
    """
    files = sorted(glob.glob(os.path.join(output_folder, "*.png")))
    if not files:
        return

    # Try to pick one file per suffix for a representative preview
    suffixes = ["base", "bright_lo", "bright_hi", "contrast_hi", "noise_hvy", "blur"]
    samples = []
    for s in suffixes:
        match = [f for f in files if f"_{s}.png" in f]
        if match:
            samples.append(match[len(match) // 2])  # pick mid-angle example

    # Pad with evenly-spaced files if we need more
    if len(samples) < n:
        step = max(1, len(files) // n)
        samples += files[::step]
    samples = samples[:n]

    for path in samples:
        img = cv2.imread(path)
        label = os.path.basename(path)
        small = cv2.resize(img, (640, 480))
        cv2.putText(
            small,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            2,
        )
        cv2.imshow("Sample (press any key for next)", small)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  LINEMOD — Augmented Training Image Generator")
    print("=" * 55)

    active_aug = [
        name
        for name, flag in [
            ("brightness", AUGMENT_BRIGHTNESS),
            ("contrast", AUGMENT_CONTRAST),
            ("noise", AUGMENT_NOISE),
            ("blur", AUGMENT_BLUR),
        ]
        if flag
    ]
    print(f"\nAugmentations active: {', '.join(active_aug) if active_aug else 'none'}")
    print(
        f"Angle range: {ANGLE_RANGE[0]}° → {ANGLE_RANGE[1]}°, "
        f"step {ANGLE_STEP}°  "
        f"({(ANGLE_RANGE[1] - ANGLE_RANGE[0]) // ANGLE_STEP + 1} angles)"
    )

    # Load existing ROI or ask user to draw one
    roi = load_roi()
    if roi is None:
        roi = select_roi_once(SOURCE_IMAGES[0])
        save_roi(roi)

    # Generate for every source image
    all_generated = []
    print()
    for src in SOURCE_IMAGES:
        paths = generate_augmented_images(
            src, roi, OUTPUT_FOLDER, ANGLE_STEP, ANGLE_RANGE
        )
        all_generated.extend(paths)

    print(f"\n{'─'*55}")
    print(f"Total images generated : {len(all_generated)}")
    print(f"Saved in               : '{OUTPUT_FOLDER}/'")
    print(f"{'─'*55}")

    print("\nShowing sample images — verify augmentations look correct.")
    preview_sample(OUTPUT_FOLDER, n=6)

    print("\nDone! Now run linemod_detector.py to train and match.")
