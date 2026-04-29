# age-estimation-CNN

## Overview

This project implements a deep learning pipeline for **facial age estimation and gender classification** using Convolutional Neural Networks (CNNs). The approach combines:

- Face detection and alignment (MTCNN / alignment tools)
- Transfer learning using VGG16
- Fine-tuning with additional datasets
- Age group classification
- KNN-based exact age estimation (experimental)

The goal is to improve age estimation accuracy by incorporating:
- Medically-informed age groupings
- Gender-based classification
- Multi-dataset training

---

## Project Pipeline

```text
Raw Images
    ↓
Face Detection + Alignment (MTCNN)
    ↓
Image Preprocessing (Resize to 224x224)
    ↓
CSV Creation (Path, Age, Gender, Age Group)
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
## Model Architecture

Base model:

VGG16 (ImageNet pretrained)

Custom layers:

Global Average Pooling
Dense layer(s)
Dropout (0.5)
Output layer (Softmax with 18 classes)

Output classes:

9 age groups × 2 genders = 18 classes

Age Groups

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
Installation

## Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/age-estimation-dissertation.git
cd age-estimation-dissertation
```
## Create environment and install dependencies:
```bash
pip install -r requirements.txt
```
## How to Run
### 1. Preprocessing
```bash
python src/preprocessing.py
```
### 2. Train on WIKI-IMDB
```bash
python src/train.py
```
### 3. Fine-tune on UTKFace
```bash
python src/finetune.py
```
### 4. Evaluate on FG-Net
```bash
python src/evaluate.py
```
## Results
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
├── notebooks/
├── src/
├── models/
├── outputs/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
``` 
## License

This project is licensed under the MIT License. See the LICENSE file for details.

Datasets used in this project are not included and remain subject to their original licenses.

Author

Syed Muhammad Hamzah Rizvi
MSc Robotics and Automation
University of Salford
