# Water Segmentation & Pretrained U-Net (TensorFlow)

**Files covered**

- `Water_Segmentation.ipynb`
- `pretrained_unet(Tensorflow).ipynb`

**Detected environment / hints from notebooks**

- Notebooks were written for Google Colab (see `google.colab` imports and `/content/drive/MyDrive/...` dataset paths).
- Typical imports seen: `tensorflow` / `tensorflow.keras`, `cv2`, `numpy`, `matplotlib`, `PIL`, `tifffile`, `tifffile`, and utilities for preprocessing and visualization.
- Model-related mentions: `U-Net`, `Unet`, `U-Net (pretrained encoder/backbone)`, `EfficientNet`, `ResNet`, `segmentation_models`.
- Example dataset path present in the notebooks: `/content/drive/MyDrive/Data/data/images` and `/content/drive/MyDrive/Data/data/labels`.

---

## Project summary

This repository contains two complementary notebooks that implement and experiment with semantic segmentation for *water segmentation* problems (binary segmentation of water vs non-water), implemented in TensorFlow / Keras.

- `Water_Segmentation.ipynb` — A full pipeline from data loading (TIFF/PNG/JPG), preprocessing, simple U‑Net (or baseline) training, evaluation and visualization of predictions.

- `pretrained_unet(Tensorflow).ipynb` — A transfer-learning version that replaces or enhances the U‑Net encoder with a pretrained backbone (examples: EfficientNet, ResNet) using either custom code or a segmentation helper library (e.g., `segmentation_models`). This notebook focuses on improved accuracy via pretrained features.

Both notebooks were authored to run in Colab but can be adapted to a local GPU setup.

---

## Table of contents

1. Quick start
2. Recommended environment & dependencies
3. Expected directory layout
4. How to run (Colab & Local)
5. Notebook-by-notebook explanation
6. Typical hyperparameters & callbacks
7. Inference & evaluation
8. Troubleshooting & tips
9. Next steps / improvements
10. License & acknowledgements

---

## 1) Quick start

1. (optional) Open the notebooks in Google Colab. If running in Colab, mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
# adjust the data paths in the notebook to match your folder
```

2. Install packages (see `requirements.txt` example below) and run cells in order.

3. Set the dataset variables (IMAGE\_DIR and MASK\_DIR) near the top of the notebook to point to your images and masks.

4. Train a model (or run the included training/evaluation cells). Save best weights and export results.

---

## 2) Recommended environment & dependencies

**Python**: 3.8 – 3.11 **GPU**: Recommended for training (Colab GPU, local CUDA-enabled GPU)

**Suggested **``** (example)**

```
# Core
python>=3.8
numpy
matplotlib
pillow
opencv-python
scikit-image
tifffile
jupyter

# Machine learning
tensorflow>=2.6  # or a 2.x stable release you have available
keras
segmentation-models  # optional: used if notebooks call it
albumentations  # optional, if augmentations are used
tensorflow-addons  # optional

# Utilities
tqdm
pandas

```

> Notes: `segmentation-models` has specifics depending on the backend (tf.keras vs keras). If you see errors importing it, check the library's documentation. If you use a different pretrained-backbone approach (custom backbone), you may not need that package.

---

## 3) Expected directory layout

```
project-root/
  data/
    images/        # raw images (.tif/.png/.jpg)
    masks/         # segmentation masks (same filename convention)
  notebooks/
    Water_Segmentation.ipynb
    pretrained_unet(Tensorflow).ipynb
  models/          # saved h5 / tf saved_model
  outputs/         # predictions, visualizations
  requirements.txt
  README.md
```

Make sure filenames between `images/` and `masks/` match (or adjust the loader logic in the notebooks).

---

## 4) How to run (detailed)

### Running in Colab (recommended for quick GPU):

1. Upload the notebooks to your Colab instance or open from Drive.
2. Install any missing pip packages at the top of the notebook, e.g.:

```python
!pip install -q -U tensorflow numpy opencv-python tifffile segmentation-models albumentations
```

3. Mount Drive and set dataset paths:

```python
from google.colab import drive
drive.mount('/content/drive')
IMAGE_DIR = '/content/drive/MyDrive/Data/data/images'
MASK_DIR  = '/content/drive/MyDrive/Data/data/labels'
```

4. Run cells sequentially (data loading → preprocessing → model build → train → evaluate → visualize).

### Running locally:

1. Create a virtual environment (venv / conda) and install packages from `requirements.txt`.
2. Copy your data into the `data/images` and `data/masks` folders or change paths in the notebook to point to your dataset.
3. If you use GPU locally, ensure CUDA & cuDNN versions match your TensorFlow build.

---

## 5) Notebook-by-notebook explanation

### `Water_Segmentation.ipynb` (baseline pipeline)

- **Purpose**: Build a working segmentation pipeline and train a baseline U‑Net or similar model for water segmentation.

- **Key steps**:

  1. Load images and masks (TIFF support is included via `tifffile`).
  2. Preprocess: resize, normalize, convert masks to binary classes.
  3. Create tf.data or generator (optional) for batching.
  4. Build model (simple U‑Net architecture using `tf.keras.layers`).
  5. Compile with loss (binary crossentropy / dice loss / combined) and metrics (IoU, accuracy, Dice coefficient).
  6. Train using callbacks (ModelCheckpoint, ReduceLROnPlateau, EarlyStopping).
  7. Visualize training curves and sample predictions.

- **Outputs**: Saved model weights, training plots, example predictions overlaid on original images.

### `pretrained_unet(Tensorflow).ipynb` (transfer learning)

- **Purpose**: Improve segmentation performance by using a pretrained encoder/backbone.

- **Key steps**:

  1. Option A: Use a `segmentation_models`-style API to instantiate a pretrained U‑Net with backbone `EfficientNet`, `ResNet`, etc.
  2. Option B: Manually create an encoder using a pretrained `tf.keras.applications` model and attach a decoder (U‑Net decoder blocks).
  3. Freeze encoder (optional), train decoder, then fine-tune encoder.
  4. Use stronger augmentations and training strategies.
  5. Evaluate with IoU/Dice and save best model.

- **Outputs**: Trained model with pretrained backbone, sample predictions, improved evaluation metrics versus baseline.

---

## 6) Typical hyperparameters & recommended callbacks

- `IMG_SIZE = (256, 256)` (or 512 depending on memory)
- `BATCH_SIZE = 4–16` (GPU memory dependent)
- `EPOCHS = 50–150` (with EarlyStopping)
- **Optimizer**: `Adam` with `lr=1e-3` (then reduce to 1e-4 on fine-tuning)
- **Loss**: `BinaryCrossentropy`, `DiceLoss`, or combined `BCE + Dice` for class imbalance
- **Callbacks**:
  - `ModelCheckpoint(save_best_only=True, monitor='val_iou' or 'val_loss')`
  - `EarlyStopping(patience=10, restore_best_weights=True)`
  - `ReduceLROnPlateau(patience=5, factor=0.5)`

---

## 7) Inference & evaluation

**Basic inference example**

```python
import tensorflow as tf
model = tf.keras.models.load_model('models/best_model.h5', compile=False)
img = ... # load and resize to IMG_SIZE
pred = model.predict(img[None, ...])[0]
mask_pred = (pred[...,0] > 0.5).astype('uint8')  # threshold
```

**Metrics**

- Intersection over Union (IoU)
- Dice coefficient / F1
- Pixel accuracy (useful but sometimes misleading with class imbalance)

Save predictions as PNG/TIFF overlays for qualitative inspection.

---

## 8) Troubleshooting & tips

- **OOM (Out-of-Memory)**: Reduce `BATCH_SIZE` or `IMG_SIZE` or use mixed precision training. Use `tf.data` with `.prefetch()`.
- **Wrong mask shapes**: Ensure masks are single-channel and values are `{0,1}` (not RGB); reshape to `(H,W,1)`.
- **Different filename conventions**: Update the loader mapping logic that pairs `image_filename` with `mask_filename`.
- **Segmentation-models import errors**: Some versions require specific `keras` or `tf.keras` backends. If failing, implement a custom encoder or install a compatible version.

---

## 9) Next steps & improvements

- Try stronger backbones (EfficientNetV2, ResNet50) and progressive resizing.
- Use focal loss or class-balanced loss for heavy class imbalance.
- Implement Test-Time Augmentation (TTA) and ensembling.
- Post-processing: morphological opening/closing to remove small false positives.
- Convert final model to `SavedModel` or TFLite for deployment.

---

## 10) License & acknowledgements

- Add an appropriate license (e.g., MIT) depending on how you want to share the code.
- Acknowledge any public datasets, pretrained backbones and third-party libraries you used.

---

### Final notes

If you want, I can:

- Generate a `requirements.txt` consistent with what's actually imported in your notebooks.
- Create a minimal `run_training.py` script that reproduces the notebook training pipeline (so you can run training from the command line).
- Convert this document into an actual `README.md` file inside the repository and provide it as a downloadable file.

Tell me which of the three next steps you prefer and I’ll prepare it.

