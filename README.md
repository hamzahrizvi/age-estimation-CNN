# Age Estimation using Deep Learning (Dissertation Project)

## Overview

This project implements a full pipeline for facial age estimation and gender classification using deep learning. It includes:

- Face detection and alignment (MTCNN / alignment tools)
- Transfer learning using VGG16
- Fine-tuning with additional datasets
- Age group classification
- KNN-based exact age estimation (experimental)

The goal is to improve age estimation accuracy by incorporating:
- Medically-informed age groupings
- Gender-based classification
- Multi-dataset training

The project focuses on improving robustness by handling real-world image issues such as blur, lighting, pose variation, and dataset imbalance.
---

## Project Pipeline

```text
Raw Images
    ↓
Face Detection + Alignment + Face Cleaning + Quality Filtering (MTCNN)
    ↓
Image Preprocessing (Resize to 224x224)
    ↓
CSV Creation (Path, Age, Gender, Age Group, Quality Scores)
    ↓
WIKI-IMDB Training (Transfer Learning - VGG16)
    ↓
UTKFace Fine-Tuning
    ↓
FG-Net Testing (Unseen Data)
    ↓
Evaluation (Accuracy, Loss, MSE)
    ↓
KNN Classifier (Exact Age Prediction)
```
## Datasets Used

This project uses publicly available datasets:

WIKI-IMDB
Large dataset (~500K images) used for initial training.
UTKFace
Balanced dataset (~20K images) used for fine-tuning.
FG-Net
Small dataset used for final testing on unseen data.
```text
⚠️ Note:Datasets are not included in this repository due to size and licensing.
```
## Dataset Cleaning
### Face detection using MTCNN
### Rejects:
- No face
- Multiple faces (optional)
- Extreme blur
- Extreme brightness/darkness
- Side-profile faces (pose filtering)
### Computes:
- Blur score
- Brightness score
- Face area ratio
- Detection confidence
- Overall quality score
- Saves rejected image previews for inspection

### Example:
```PowerShell
py -3.8 src\clean_all_faces_from_folder.py --source "path_to_wiki_crop" --output "data\cleaned_faces\wiki" --log "outputs\wiki_cleaning_report.csv" --dataset-name "wiki"

# FG-Net
py -3.8 src\clean_all_faces_from_folder.py --source "path_to_fgnet" --output "data\cleaned_faces\fgnet" --log "outputs\fgnet_cleaning_report.csv" --dataset-name "fgnet"

# IMDB
py -3.8 src\clean_all_faces_from_folder.py --source "path_to_imdb_crop" --output "data\cleaned_faces\imdb" --log "outputs\imdb_cleaning_report.csv" --dataset-name "imdb"

# UTKFace
py -3.8 src\clean_all_faces_from_folder.py --source "data\UTKFace" --output "data\cleaned_faces\utk" --log "outputs\utk_cleaning_report.csv" --dataset-name "utk"
```

## CSV Creation
### Interactive filename parsing using regex
### Supports multiple datasets (UTKFace, FG-Net, custom)
### Outputs:
- age, gender, age_group
- final_label (18 classes)
- quality_score
- face metrics

### Example (UTKFace):
```PowerShell
py -3.8 src\create_dataset_csv_interactive.py ^
  --source "data\cleaned_faces\utk" ^
  --output "data\utk_final.csv" ^
  --dataset-name "utkface" ^
  --pattern "^(?P<age>\d+)_(?P<gender>\d+)_(?P<race>\d+)_.*$"
```
### Example (FG-Net):
```PowerShell
py -3.8 src\create_dataset_csv_interactive.py ^
  --source "data\cleaned_faces\fgnet" ^
  --output "data\fgnet_final.csv" ^
  --dataset-name "fgnet" ^
  --pattern "^.*A(?P<age>\d+).*$"
```


## Model Architecture

### Base model:

VGG16 (ImageNet pretrained)

### Custom layers:

Global Average Pooling
Dense layer(s)
Dropout (0.5)
Output layer (Softmax with 18 classes)

Output classes:

9 age groups × 2 genders = 18 classes

### Age Groups

The model uses medically-inspired age groupings:

### Class	Age Range
```text 
| Class | Age Range |
|-------|----------|
| 0     | 0–2      |
| 1     | 3–5      |
| 2     | 6–13     |
| 3     | 14–18    |
| 4     | 19–24    |
| 5     | 25–33    |
| 6     | 34–48    |
| 7     | 49–64    |
| 8     | 65+      |
```
## Installation

### Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/age-estimation-dissertation.git
cd age-estimation-dissertation
pip install -r requirements.txt
```
### Create environment and install dependencies:
```bash
pip install -r requirements.txt
```
### How to Run
#### 1. Preprocessing
```bash
python src/preprocessing.py
```
#### 2. Train on WIKI-IMDB
```bash
python src/train.py
```
#### 3. Fine-tune on UTKFace
```bash
python src/finetune.py
```
#### 4. Evaluate on FG-Net
```bash
python src/evaluate.py
```
### Results
```text
Typical results observed:

- WIKI-IMDB validation accuracy: ~55%
- UTKFace fine-tuning accuracy: ~42%
- FG-Net test accuracy: ~40%
```
```text
⚠️ Note: Performance varies depending on dataset quality and preprocessing.
```

## Evaluation Metrics
```text
- Accuracy
- Validation Accuracy
- Loss (Categorical Crossentropy)
- Mean Squared Error (MSE)
- Confusion Matrix (recommended)
- Mean Absolute Error (recommended improvement)
```
## KNN Classifier (Experimental)

### After CNN prediction:
```text
- Features are extracted from an intermediate layer
- KNN is applied to estimate exact age
```
### Result:
```text
- Low accuracy (~12–13%)
- Needs further improvement
```
## Limitations
```text
- Dataset quality issues (lighting, resolution, noise)
- Class imbalance in age groups
- Overfitting in large dense layers
- Weak performance on children and elderly faces
- KNN classifier underperforms
```
## Future Improvements
```text
- Use better datasets (e.g., MORPH II)
- Replace Dense layers with lighter architecture
- Add ethnicity and emotion features
- Improve augmentation techniques
- Use regression-based age prediction (MAE)
- Try modern architectures (EfficientNet, ResNet)
- Improve fine-tuning strategy
- Replace KNN with regression models (SVR, RandomForest)
```
## Project Structure
```text
├── src/
│   ├── clean_all_faces_from_folder.py
│   ├── create_dataset_csv_interactive.py
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── finetune.py
│   ├── evaluate.py
│   ├── knn_classifier.py
│   └── utils.py
├── notebooks/
│   ├── Age_Estimation_Main.ipynb
│   └── Model h5 (Wiki_IMDB transfer learning).ipynb
├── models/
│   └── .gitkeep
├── outputs/
├── config.yaml
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```
## Installation

```Bash
git clone https://github.com/YOUR_USERNAME/age-estimation-dissertation.git
cd age-estimation-dissertation
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

Datasets used in this project are not included and remain subject to their original licenses.

Author

Syed Muhammad Hamzah Rizvi
MSc Robotics and Automation
University of Salford
