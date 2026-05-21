import cv2
import numpy as np
import os

LINEMOD_PAD = 320
EDGE_MARGIN = 32  # ✅ keeps mask away from image edges → fixes segfault


def inspect_npz_file(file_path):
    print(f"--- Inspecting: {file_path} ---")
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return None
    try:
        data = np.load(file_path)
        print(f"Found {len(data.files)} arrays in the file:")
        for key in data.files:
            array_data = data[key]
            print(
                f" -> '{key}': Shape {array_data.shape}, Data Type: {array_data.dtype}"
            )
        return data
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None


def setup_linemod_detector():
    print("\n--- Setting up LINEMOD Detector ---")
    detector = cv2.linemod.getDefaultLINE()
    print("LINEMOD Detector successfully initialized!")
    return detector


def pad_image_for_linemod(img, T=LINEMOD_PAD):
    h, w = img.shape[:2]
    pad_h = (T - (h % T)) % T
    pad_w = (T - (w % T)) % T
    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
    assert img.shape[0] % T == 0
    assert img.shape[1] % T == 0
    return img


def check_template_success(result):
    if isinstance(result, tuple):
        return result[0]
    return result


def diagnose_roi_features(roi_img):
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_pixels = np.count_nonzero(edges)
    total_pixels = gray.shape[0] * gray.shape[1]
    edge_ratio = edge_pixels / total_pixels

    print(
        f"    ROI gradient density: {edge_pixels} edge pixels / {total_pixels} total = {edge_ratio:.3f}"
    )

    if edge_ratio < 0.01:
        print("    ⚠️  WARNING: Very few edges detected in ROI!")
        print("       LINEMOD needs strong gradient features (object edges/textures).")
        print(
            "       Try: draw a tighter box around the most detailed part of the object."
        )
    else:
        print(f"    ✅ ROI has sufficient gradient features.")

    cv2.imshow(
        "Edge map of your ROI — should show clear object outline (press any key)", edges
    )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def make_safe_mask(img_shape, x, y, w, h):
    """
    ✅ FIX: Clamps the active mask region to stay EDGE_MARGIN pixels away
    from every image border. OpenCV 4.13 segfaults when the mask is active
    too close to the edge inside addTemplate.
    """
    img_h, img_w = img_shape[:2]

    safe_x1 = max(x, EDGE_MARGIN)
    safe_y1 = max(y, EDGE_MARGIN)
    safe_x2 = min(x + w, img_w - EDGE_MARGIN)
    safe_y2 = min(y + h, img_h - EDGE_MARGIN)

    if safe_x2 <= safe_x1 or safe_y2 <= safe_y1:
        return None  # ROI too close to edge

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    mask[safe_y1:safe_y2, safe_x1:safe_x2] = 255
    return mask


def train_template(detector, image_path, class_id="my_object"):
    print(f"\n--- Training Template from: {image_path} ---")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image '{image_path}'.")
        return False

    print(f"    Training image size: {img.shape[1]}w x {img.shape[0]}h")
    print(
        ">>> Draw a TIGHT box around just the object edges, then press SPACE or ENTER."
    )
    print("    Tip: Don't include too much background — focus on the object itself.")

    x, y, w, h = cv2.selectROI(
        "Select Object for Template", img, fromCenter=False, showCrosshair=True
    )
    cv2.destroyWindow("Select Object for Template")

    if w == 0 or h == 0:
        print("No bounding box selected. Skipping.")
        return False

    print(f"    ROI selected: x={x}, y={y}, w={w}, h={h}")

    roi_img = img[y : y + h, x : x + w]

    print("\n    Checking ROI feature quality...")
    diagnose_roi_features(roi_img)  # ← popup 1: edge map

    img_padded = pad_image_for_linemod(img.copy(), T=LINEMOD_PAD)

    # ✅ FIX: use safe mask instead of raw slice
    full_mask = make_safe_mask(img_padded.shape, x, y, w, h)
    if full_mask is None:
        print(
            "❌ ROI is too close to the image edge. Please select a region further from the border."
        )
        return False

    print(
        f"\n    Full padded image size: {img_padded.shape[1]}w x {img_padded.shape[0]}h"
    )
    print(f"    Mask active region: {w}w x {h}h at ({x},{y})")

    result = detector.addTemplate([img_padded], class_id, full_mask)
    template_id = check_template_success(result)

    print(f"\n    Raw addTemplate result: {result}")
    print(f"    Parsed template ID: {template_id}")

    if template_id != -1:
        print(f"✅ Template trained successfully! Template ID: {template_id}")

        # ← popup 2: preview of trained region
        cv2.imshow(
            "Trained ROI preview — does this look right? (press any key)", roi_img
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return True
    else:
        print("❌ addTemplate returned -1. Template was REJECTED by LINEMOD.")
        print("   1. Not enough gradient features in the selected region")
        print("   2. Try a tighter ROI around the most textured part of the object")
        return False


def find_matches(detector, test_image_path, threshold=80.0):
    print(f"\n--- Searching for matches in: {test_image_path} ---")
    img = cv2.imread(test_image_path)
    if img is None:
        print(f"Error: Could not load image '{test_image_path}'.")
        return

    print(f"    Test image size: {img.shape[1]}w x {img.shape[0]}h")
    img_padded = pad_image_for_linemod(img, T=LINEMOD_PAD)

    thresholds_to_try = [threshold, 70.0, 55.0, 40.0, 25.0]
    matched_at = None

    for t in thresholds_to_try:
        matches, _ = detector.match([img_padded], t)
        print(f"    Threshold {t:5.1f}% → {len(matches)} match(es)")
        if matches:
            matched_at = t
            print(f"    Top matches at {t}%:")
            for i, m in enumerate(matches[:5]):
                print(
                    f"      [{i+1}] similarity={m.similarity:.1f}%, position=({m.x}, {m.y})"
                )
            break

    if matched_at is None:
        print(
            "\n❌ No matches found even at 25%. Check the edge map shown during training."
        )
        return

    best = matches[0]
    display = img_padded.copy()

    templates = detector.getTemplates(best.class_id, best.template_id)
    t_w, t_h = 80, 80
    if templates and templates[0].features:
        xs = [f.x for f in templates[0].features]
        ys = [f.y for f in templates[0].features]
        t_w = max(xs) - min(xs) + 20
        t_h = max(ys) - min(ys) + 20

    for m in matches[:10]:
        cv2.circle(display, (m.x, m.y), 4, (0, 165, 255), -1)

    bx, by = best.x, best.y
    cv2.rectangle(display, (bx, by), (bx + t_w, by + t_h), (0, 255, 0), 2)
    cv2.putText(
        display,
        f"Best: {best.similarity:.1f}%",
        (bx, by - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    cv2.imshow("LINEMOD Result (press any key)", display)  # ← popup 3: match result
    cv2.waitKey(0)
    cv2.destroyAllWindows()


import glob

if __name__ == "__main__":
    detector = cv2.linemod.getDefaultLINE()

    # Select ROI once on the original
    roi = select_roi_once("00.png")

    # Train on ALL generated images automatically
    training_images = sorted(glob.glob("rotated_training/*.png"))
    print(f"Training on {len(training_images)} images...")

    total = 0
    for img_path in training_images:
        total += train_with_rotation(
            detector,
            img_path,
            roi,
            angle_step=0,  # ← 0 means no extra rotation on top,
            angle_range=(0, 0),  #   since the image is already rotated
            class_id="board",
        )

    print(f"\n✅ Total templates: {total}")
    if total > 0:
        find_matches(detector, "04.png")
