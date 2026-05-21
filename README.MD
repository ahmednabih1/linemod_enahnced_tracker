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


1. Generate Augmented Templates
Place your source object images (00.png, 01.png, etc.) in the root directory. Run the augmentation script to define your Region of Interest (ROI) and generate variations:

Bash
python rotating_image_training.py
A window will open showing your first image. Draw a tight bounding box around the target object and press ENTER or SPACE.

The script will cache your ROI selections to saved_roi.json and generate an array of training configurations inside a new rotated_training/ folder.

2. Train and Run the Detector
Once training data is generated, initialize the LINEMOD template matching system:

Bash
python linemod_detector.py
The pipeline will parse your augmented images, verify edge feature density profiles, register templates safely, and execute a multi-threshold scanning pass on your test target image (04.png).


---

### Step 3: Push the README to GitHub

Now that the file actually exists on your Mac, tell Git to upload it to your repository by running these three commands in your Terminal:

```bash
# 1. Stage the new README file
git add README.md

# 2. Commit the change
git commit -m "Docs: Add professional README documentation"

# 3. Push it to GitHub
git push origin main