# Facial Age Estimation with Hierarchical EfficientNet

This project is an end-to-end facial age estimation pipeline built with Python, TensorFlow/Keras, and transfer learning.

The project started as a simple age classification experiment and was gradually rebuilt into a more complete machine learning pipeline: dataset cleaning, metadata processing, model training, fine-tuning, evaluation, and experiment tracking.

The final version uses a hierarchical EfficientNetB0 model that predicts:

- gender
- coarse age group
- fine age group

The best result came from training on WIKI + IMDB and then fine-tuning on UTKFace.

---

## Why I Built This

The goal was not just to train a model, but to build a realistic ML workflow around a messy real-world computer vision problem.

Facial age estimation is difficult because:

- age labels are noisy
- facial ageing is gradual, not categorical
- age groups have fuzzy boundaries
- public datasets are heavily imbalanced
- WIKI/IMDB contain many adult celebrity faces
- underage and borderline age groups are underrepresented
- models often collapse toward dominant adult classes

A big part of this project was testing these limitations rather than hiding them.

---

## Datasets

The project uses several public facial age datasets:

| Dataset | Use |
|---|---|
| WIKI | Large-scale base training |
| IMDB | Large-scale base training |
| UTKFace | Fine-tuning on cleaner aligned face crops |
| FG-NET | External stress test / domain-shift evaluation |

Large datasets and trained models are not included in this repository.

---

## Final Age Classes

The final version uses 7 fine age groups:

| Class | Age Range |
|---:|---|
| 0 | 0–13 |
| 1 | 14–17 |
| 2 | 18–24 |
| 3 | 25–33 |
| 4 | 34–48 |
| 5 | 49–64 |
| 6 | 65+ |

Coarse age groups:

| Class | Age Range |
|---:|---|
| 0 | 0–13 |
| 1 | 14–48 |
| 2 | 49+ |

Gender classes:

| Class | Label |
|---:|---|
| 0 | male |
| 1 | female |

---

## Model Architecture

The best model uses EfficientNetB0 as a shared feature extractor with three prediction heads.

```text
Input face image
        ↓
EfficientNetB0 backbone
        ↓
Shared dense representation
        ↓
 ┌───────────────┬───────────────────┬────────────────┐
 │ Gender output │ Coarse age output │ Fine age output │
 │ 2 classes     │ 3 classes         │ 7 classes       │
 └───────────────┴───────────────────┴────────────────┘
```

This is a multi-task model. Instead of forcing the network to predict one combined class, it learns related tasks together.

---

## Training Pipeline

Each image is processed using a TensorFlow `tf.data` pipeline:

```text
image path from CSV
→ read from disk
→ decode JPEG/PNG
→ resize to 224×224
→ preprocess for EfficientNet
→ apply training augmentation
→ batch images
→ prefetch for GPU efficiency
```

Training augmentations:

- horizontal flip
- brightness adjustment
- contrast adjustment

Validation data is not augmented.

---

## Training Strategy

The model was trained in stages.

### Phase 1: Base Training

The model was trained on the combined WIKI + IMDB dataset.

EfficientNet was initially frozen so the classifier heads could learn without damaging the pretrained ImageNet features.

### Phase 2: EfficientNet Fine-Tuning

After the classifier heads had stabilised, the top EfficientNet layers were unfrozen and trained with a very low learning rate.

Only the upper EfficientNet layers were unfrozen, while BatchNorm layers were kept frozen.

### Phase 3: UTKFace Fine-Tuning

The best WIKI + IMDB model was fine-tuned on UTKFace.

This significantly improved performance because UTKFace contains cleaner, more consistent aligned face crops.

---

## Best Results

### Best Model: WIKI + IMDB → UTK Fine-Tuned EfficientNet

| Metric | Result |
|---|---:|
| Fine age exact accuracy | 57.86% |
| Fine age near-group accuracy (+/-1) | 92.58% |
| Coarse age accuracy | 88.38% |
| Gender accuracy | 87.46% |

The near-group metric is important because age is ordinal. Predicting a neighbouring age group is much less severe than predicting a completely different life stage.

For example:

```text
True: 25–33
Predicted: 34–48
```

is a much better error than:

```text
True: 25–33
Predicted: 0–13
```

---

## Detailed UTK Fine-Tuned Results

### Fine Age Classification

| Age Group | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0–13 | 0.85 | 0.90 | 0.87 | 3,494 |
| 14–17 | 0.58 | 0.04 | 0.07 | 739 |
| 18–24 | 0.44 | 0.16 | 0.24 | 2,670 |
| 25–33 | 0.54 | 0.69 | 0.60 | 6,915 |
| 34–48 | 0.46 | 0.52 | 0.49 | 4,752 |
| 49–64 | 0.55 | 0.50 | 0.53 | 3,177 |
| 65+ | 0.68 | 0.68 | 0.68 | 1,937 |

Overall fine-age accuracy:

```text
57.86%
```

Near-group accuracy:

```text
92.58%
```

### Coarse Age Classification

| Age Group | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0–13 | 0.91 | 0.85 | 0.88 | 3,494 |
| 14–48 | 0.89 | 0.94 | 0.91 | 15,076 |
| 49+ | 0.85 | 0.76 | 0.80 | 5,114 |

Overall coarse-age accuracy:

```text
88.38%
```

### Gender Classification

| Gender | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| male | 0.91 | 0.82 | 0.86 | 11,298 |
| female | 0.85 | 0.92 | 0.88 | 12,386 |

Overall gender accuracy:

```text
87.46%
```

---

## Experiment History

Several experiments were attempted before reaching the final version.

### 1. VGG16 Baseline

The original model used VGG16 and an 18-class target:

```text
9 age groups × 2 gender classes
```

This was useful as a baseline but had limitations:

- less flexible target structure
- weaker generalisation
- worse domain-shift behaviour

### 2. Improved VGG16

The classifier head was improved by replacing large dense layers with:

- GlobalAveragePooling
- BatchNormalization
- Dropout
- smaller dense layers

This made the model cleaner and less overfit.

### 3. Hierarchical EfficientNet

The project then moved to EfficientNetB0 with three outputs:

- gender
- coarse age
- fine age

This became the strongest architecture.

### 4. Weighted Loss Experiment

Weighted loss was tested to improve minority age-group recall.

Result:

```text
best validation fine-age accuracy: ~28.30%
```

This performed worse than the standard training setup.

Likely reason:

```text
the minority classes were too small and noisy, so aggressive weighting destabilised training.
```

### 5. Balanced Downsampling Experiment

Balanced downsampling was tested by limiting the number of samples per class.

This also reduced performance because too much useful adult data was removed.

### 6. Soft Ordinal Label Experiment

Soft labels were tested to reduce harsh penalties between neighbouring age groups.

Result:

| Metric | Result |
|---|---:|
| Fine exact accuracy | 44.99% |
| Fine near-group accuracy | 87.31% |
| Coarse age accuracy | 84.33% |
| Gender accuracy | 82.07% |

This did not outperform the hard-label model.

### 7. UTK Fine-Tuning

Fine-tuning on UTKFace produced the strongest final result.

This suggests the model benefits heavily from cleaner, more consistent face crops.

---

## Key Lessons

The main lesson from this project is that the limiting factor was not just architecture.

The biggest bottlenecks were:

- dataset imbalance
- underrepresented teenage and borderline age groups
- noisy age labels
- domain shift between datasets
- ambiguous visual age boundaries

The model performed well on broad age groups and near-group prediction, but struggled with exact classification in visually similar age bands such as:

```text
14–17 vs 18–24
18–24 vs 25–33
```

This is expected in facial age estimation and is why near-group accuracy was tracked.

---

## Running the Project

### Create Environment

```powershell
conda create -n tfgpu python=3.8
conda activate tfgpu
pip install -r requirements.txt
```

### Train Base Model

```powershell
python src\train.py --csv data\csv\wiki_imdb_hierarchical.csv --output models\efficientnet_hierarchical_7class_weights.h5 --epochs 25 --batch-size 16 --learning-rate 0.0001
```

### Fine-Tune on WIKI + IMDB

```powershell
python src\train.py --csv data\csv\wiki_imdb_hierarchical.csv --output models\efficientnet_hierarchical_7class_finetuned_weights.h5 --resume-from models\efficientnet_hierarchical_7class_weights.h5 --initial-epoch 25 --epochs 35 --batch-size 16 --learning-rate 0.000003 --fine-tune
```

### Create UTK Hierarchical CSV

```powershell
python src\create_utk_hierarchical_csv.py --input-csv data\csv\utk_final.csv --output-csv data\csv\utk_hierarchical.csv --image-root data\cleaned_faces\utk
```

### Fine-Tune on UTKFace

```powershell
python src\train.py --csv data\csv\utk_hierarchical.csv --output models\efficientnet_hierarchical_7class_utk_finetuned_weights.h5 --resume-from models\efficientnet_hierarchical_7class_finetuned_weights.h5 --initial-epoch 35 --epochs 45 --batch-size 16 --learning-rate 0.000001 --fine-tune
```

### Evaluate

```powershell
python src\evaluate.py --csv data\csv\utk_hierarchical.csv --weights models\efficientnet_hierarchical_7class_utk_finetuned_weights.h5 --batch-size 16 --output-dir outputs\evaluation_utk_7class_finetuned
```

---

## Repository Structure

```text
src/
├── train.py
├── evaluate.py
├── create_combined_age_gender_csv.py
├── create_imdb_from_mat_and_report.py
├── create_utk_hierarchical_csv.py

docs/
└── results/

data/
models/
outputs/
```

The `data/`, `models/`, and `outputs/` folders are ignored by Git because they contain datasets, trained weights, and generated files.

Selected result images and summary files can be stored in:

```text
docs/results/
```

---

## Current Status

This version is strong enough to present as a portfolio project.

It demonstrates:

- data cleaning
- dataset merging
- metadata conversion
- transfer learning
- multi-task learning
- fine-tuning
- GPU training
- experiment tracking
- evaluation beyond accuracy
- honest analysis of failed experiments

The strongest result is the UTK fine-tuned hierarchical EfficientNet model.

---

## Future Work

Possible next improvements:

- collect more data for 14–17 and 18–24 age groups
- add landmark-based face alignment
- add confidence-aware predictions
- create a Streamlit demo
- add model card documentation
- test ConvNeXt or EfficientNetV2
- export a lightweight inference model
- build an age-restriction classifier focused on threshold decisions

---

## Final Note

This project is not presented as a production-ready age verification system.

It is a practical ML engineering project showing how a model improves through preprocessing, architecture changes, transfer learning, fine-tuning, and evidence-based experiment tracking.