# OpenCV LINEMOD Object Detection Pipeline

A robust, template-based 2D/3D object detection pipeline leveraging OpenCV's `linemod` module. This framework includes automated dataset augmentation (rotation, blur, contrast, brightness alterations) and feature density diagnostics to build high-accuracy detection templates.

## 🚀 Features

* **OpenCV 4.x Segfault Fix:** Clamps active mask regions away from image borders via a custom safe-mask implementation (`EDGE_MARGIN`), resolving native crash bugs inside `addTemplate`.
* **Automatic Data Augmentation:** Dynamically generates multi-angle rotation templates with customizable environmental perturbations (Gaussian noise, lighting shifts, and blur).
* **ROI Feature Quality Diagnostic:** Utilizes a Canny edge-density analyzer to evaluate whether a selected Region of Interest (ROI) contains sufficient gradient features before training.
* **Smart Padding:** Guarantees input image compatibility by automatically handling dimensions against LINEMOD's strict alignment constraints (`LINEMOD_PAD = 320`).

---

## 📂 Project Structure

```text
├── linemod_detector.py         # Main detection script (template training & matching)
├── rotating_image_training.py  # Image augmentation & dataset generation engine
├── saved_roi.json              # Cached bounding box coordinates for reusable training
├── .gitignore                  # Prevents tracking of generated cache and large image sets
└── 00.png, 01.png...           # Original source images for training and testing