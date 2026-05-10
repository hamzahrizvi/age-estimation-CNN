# Facial Age Estimation Pipeline

This repository contains an end-to-end facial age estimation project built with Python, TensorFlow/Keras, and transfer learning.

The project started as a basic CNN experiment and has been reworked into a cleaner machine learning pipeline with dataset preparation, face cleaning, model training, fine-tuning, and evaluation scripts.

The main goal is to predict facial age groups from face images. Earlier experiments also used a combined age-group and gender label, but the next development stage will focus more directly on age-group classification to improve accuracy and make the project more practical.

---

## Project Overview

The pipeline supports multiple public face-age datasets:

- WIKI
- IMDB
- UTKFace
- FG-Net

The current workflow is:

```text
raw images
→ face detection and cleaning
→ CSV label generation
→ transfer learning training
→ fine-tuning
→ evaluation
```

The repository is structured so that preprocessing, training, fine-tuning, and evaluation can be run separately.

---

## Repository Structure

```text
age-estimation-git/
├── src/
│   ├── clean_all_faces_from_folder.py
│   ├── create_dataset_csv_interactive.py
│   ├── create_wiki_from_cleaning_report.py
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── finetune.py
│   ├── evaluate.py
│   ├── knn.py
│   └── utils.py
│
├── notebooks/
│   └── original / experimental notebooks
│
├── data/
│   └── CSV files only
│
├── outputs/
│   └── logs, reports, confusion matrices
│
├── config.yaml
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

Large datasets and trained model files are not included in the repository.

---

## Current Model Task

The current model uses 18 classes:

```text
9 age groups × 2 gender classes
```

The label is calculated as:

```python
final_label = (age_group * 2) + gender
```

This means each class represents both age group and gender.

For example:

```text
age_group = 5
gender = 1
final_label = 11
```

This approach is useful for experimenting with multi-attribute facial classification, but it also makes the task harder. A future version of this project will use age-group-only classification as the main target.

---

## Age Groups

| Group | Age Range |
|---|---|
| 0 | 0–2 |
| 1 | 3–5 |
| 2 | 6–13 |
| 3 | 14–18 |
| 4 | 19–24 |
| 5 | 25–33 |
| 6 | 34–48 |
| 7 | 49–64 |
| 8 | 65+ |

---

## Datasets

### WIKI

Used as the main base training dataset.

### IMDB

Prepared as an additional large-scale training dataset. IMDB contains a large number of images, so cleaning is handled in chunks.

### UTKFace

Used for fine-tuning.

UTKFace images are already aligned face crops, so they should not be re-cropped.

### FG-Net

Used as an external cross-dataset evaluation set.

FG-Net does not provide gender labels in the current CSV setup, so it is evaluated using age-group predictions only.

---

## Face Cleaning Pipeline

The main cleaning script is:

```text
src/clean_all_faces_from_folder.py
```

It performs:

- face detection
- face cropping
- blur filtering
- brightness filtering
- side-pose filtering
- multiple-face rejection
- face area filtering
- quality scoring
- rejected-image preview saving
- CSV cleaning report generation
- resumable processing
- chunk-based processing

The goal is to reduce noisy training data before model training.

---

## Cleaning Features

### Blur Detection

Images with low Laplacian variance are rejected as blurry.

### Brightness Filtering

Images that are too dark or too bright are removed.

### Pose Filtering

Extreme side-profile faces are filtered using facial landmark offsets.

### Multiple Face Rejection

Images with more than one detected face can be rejected to avoid incorrect labels.

### Quality Score

Accepted images receive a quality score based on detection confidence, blur, and brightness.

---

## Training Setup

The current training setup uses transfer learning.

The main baseline model is VGG16 with a smaller custom classification head:

```text
VGG16 backbone
→ GlobalAveragePooling2D
→ BatchNormalization
→ Dense(512)
→ Dropout(0.5)
→ Dense(18)
```

This replaced an older architecture that used very large dense layers, which caused overfitting.

---

## Current Training Results

### VGG16 Base Training on WIKI

The VGG16 model was trained on the cleaned WIKI dataset.

Final observed WIKI training run:

```text
training accuracy: approximately 46%
validation accuracy: approximately 39%
```

The model plateaued after a number of epochs, which suggests that VGG16 is a reasonable baseline but not the final model to rely on.

### Fine-Tuning on UTKFace

The WIKI-trained VGG16 model was fine-tuned on UTKFace.

Final observed result:

```text
UTK validation accuracy: 32.75%
```

This showed that fine-tuning helped the model adapt to a different dataset.

### FG-Net Cross-Dataset Evaluation

FG-Net was used as an external stress test.

Observed result:

```text
age-group accuracy: 7.39%
near age-group accuracy: 24.55%
```

This result is low, but useful. It shows that cross-dataset generalisation is difficult, especially when training data, face quality, dataset format, and age distribution differ.

FG-Net is currently treated as a domain-shift test rather than the main success metric.

---

## Important Notes on Results

The current results should be understood as a baseline, not the final target.

The main lessons from the first full version are:

- the preprocessing pipeline works
- training and fine-tuning scripts are functional
- VGG16 can learn useful patterns but is limited
- cross-dataset evaluation is much harder than same-dataset validation
- the 18-class age-plus-gender setup makes the problem significantly harder
- age-group-only classification is the next realistic path toward stronger accuracy

The next target is to build a cleaner age-only model and aim for around 70% validation accuracy on a properly cleaned and balanced dataset.

---

## Environment

The project was tested using:

```text
Python 3.8
TensorFlow 2.10
CUDA-enabled GPU environment
NVIDIA RTX 2070 laptop GPU
```

TensorFlow 2.10 was used because newer TensorFlow versions do not support native Windows GPU training in the same way.

---

## Installation

Create and activate a Conda environment:

```powershell
conda create -n tfgpu python=3.8
conda activate tfgpu
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Check GPU availability:

```python
import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))
```

---

## Example Commands

### Train VGG16 on WIKI

```powershell
python src\train.py --csv data\csv\wiki_final.csv --model vgg16 --output models\vgg16_wiki_base.keras --epochs 100 --batch-size 32 --learning-rate 0.0001
```

### Fine-Tune on UTKFace

```powershell
python src\finetune.py --csv data\csv\utk_final.csv --model-in models\vgg16_wiki_base.keras --model-out models\vgg16_wiki_utk_finetuned.keras --epochs 15 --batch-size 16 --learning-rate 0.000001
```

### Evaluate on FG-Net

```powershell
python src\evaluate.py --csv data\csv\fgnet_final.csv --model models\vgg16_wiki_utk_finetuned.keras --batch-size 16 --image-root "path\to\fgnet\images"
```

---

## Current Limitations

This project is still under active development.

Known limitations:

- VGG16 is used mainly as a baseline
- the 18-class combined age/gender task is difficult
- FG-Net does not contain usable gender labels in the current setup
- dataset shift between WIKI, UTKFace, and FG-Net is significant
- class imbalance affects performance
- accuracy is not yet high enough for production use
- age-only classification is likely to perform better than age-plus-gender classification

---

## Next Development Goals

The next stage of the project will focus on improving accuracy and making the repository stronger as a machine learning portfolio project.

Planned improvements:

- train an age-group-only model with 9 output classes
- use EfficientNetB0 as the main model
- compare VGG16 and EfficientNetB0
- add ResNet50 or ConvNeXt experiments
- improve class balancing
- add better evaluation metrics
- add confusion matrix visualisations
- add inference script for a single image
- add a simple demo interface
- improve README documentation with result tables and example outputs

---

## Planned Age-Only Direction

The next version will move from this:

```text
18 classes = age group + gender
```

to this:

```text
9 classes = age group only
```

This should make the task more realistic and improve accuracy.

The planned model output will be:

```python
Dense(9, activation="softmax")
```

instead of:

```python
Dense(18, activation="softmax")
```

The main target is:

```text
70%+ validation accuracy on cleaned age-group classification
```

This is a realistic goal for a portfolio-ready version of the project.

---

## Why This Project Matters

This project demonstrates several practical machine learning engineering skills:

- working with messy real-world datasets
- building preprocessing pipelines
- cleaning image data
- handling dataset imbalance
- training CNN transfer learning models
- fine-tuning pretrained networks
- evaluating domain shift
- writing modular Python scripts
- using GPU training locally
- documenting results honestly

The project is not presented as a finished production model. It is a practical machine learning pipeline that is being improved step by step.

---

## License

This project is licensed under the terms included in the repository license file.
