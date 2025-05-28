# 🧱 Wall Damage Classification and Crack Criteria Prediction

This project implements a complete end-to-end pipeline for detecting wall surface damage severity (3 classes) and predicting associated crack characteristics (11 multi-label criteria). It uses PyTorch and ConvNeXt-Tiny architecture, and supports full inference on unlabeled test sets.

> **Note**: Please follow the steps below to reproduce results consistent with our local and Kaggle submissions.

---

## 📁 Directory Structure

```bash
.
├── datasets/
│   ├── damage_classification_forTrain/wall_damage/     # 3-class training images (ImageFolder)
│   ├── crack/train/                                     # Multi-label crack image folders
│   ├── crack/valid/
│   └── test_data/wall/                                  # Unlabeled test images
├── train_labels.csv                                     # Auto-generated from crack/train
├── valid_labels.csv                                     # Auto-generated from crack/valid
├── best_damage_model.pth                                # Saved damage classification model
├── best_crack_model.pth                                 # Saved crack multi-label model
└── wall.csv                                             # Final submission (ID, class, criteria)
```

---

## 🧠 Model Descriptions

### 1. Damage Level Classification (3 Classes)

- Uses `ImageFolder`-based dataset
- Model: `convnext_tiny` with modified classifier output to 3
- Loss: CrossEntropyLoss
- Output:
  - Class A → 18
  - Class B → 19
  - Class C → 20

### 2. Crack Feature Multi-Label Detection (11 Classes)

- CSV-based custom dataset with one-hot encoded labels
- Model: `convnext_tiny`, output layer → 11
- Loss: BCEWithLogitsLoss
- Output: Criteria such as:
  ```
  0: Exposed rebar
  1: No significant damage
  2: Huge Spalling
  3: X and V-shaped cracks
  ...
  10: Small cracks
  ```

---

## 🛠 Preprocessing (CSV Generator)

Run the embedded `make_multilabel_csv()` to generate:
```bash
train_labels.csv
valid_labels.csv
```
from multi-label folder names like: `2_Huge Spalling 4_Continuous Diagonal cracks`

---

## 🚀 Training Workflow

### Damage Classification:
```bash
python train_damage_model.py
# Trains the 3-class ConvNeXt model
# Outputs best_damage_model.pth
```

### Crack Feature Detection:
```bash
python train_crack_model.py
# Trains the 11-label ConvNeXt model using CSV dataset
# Outputs best_crack_model.pth
```

---

## 🔍 Inference Pipeline

1. Load both `best_damage_model.pth` and `best_crack_model.pth`
2. For each image in `datasets/test_data/wall`:
   - Predict damage class → map to class ID: 18, 19, or 20
   - Restrict crack criteria based on class:
     - Class A → [0, 2]
     - Class B → [3, 4, 6, 8]
     - Class C → [1, 5, 7, 9, 10]
   - Run sigmoid on crack model outputs
   - Use threshold 0.5 or fallback to max-probability allowed criterion
3. Save predictions to `wall.csv`:
```csv
ID,class
0001,18,0,2
0002,20,1,5,9
```

---

## 📦 Requirements

```bash
pip install torch torchvision pandas tqdm pillow
```

---


## 📬 Contact

If you have questions or encounter bugs, feel free to open an issue or fork and contribute.
