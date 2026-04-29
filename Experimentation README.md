# Experimental Models and Explorations

## Overview

This branch contains **additional experiments** conducted during the dissertation project on facial age estimation. These experiments explore alternative architectures, fine-tuning strategies, and hybrid approaches beyond the main VGG16 pipeline.

⚠️ These are **not final models** and are included for **research transparency and comparison purposes only**.

---

## Experiments Included

### 1. VGG16 (WIKI-IMDB Baseline)
- Transfer learning using VGG16 pretrained on ImageNet
- Trained on WIKI-IMDB dataset
- Achieved ~55–57% validation accuracy

✅ **Best-performing model across all experiments**

---

### 2. VGG19 / ResNet Testing
- Alternative deep architectures tested
- Similar pipeline to VGG16
- Lower performance (~35–40% validation accuracy)

📌 **Conclusion:** More complex models did not outperform VGG16

---

### 3. FG-Net Fine-Tuning
- Fine-tuned pretrained model using FG-Net dataset
- Very small dataset (~700 training images)

❌ Results:
- Low validation accuracy (~10–15%)
- High instability and overfitting

📌 **Conclusion:** FG-Net is too small for effective fine-tuning

---

### 4. KNN on CNN Features
- Extracted features from second-last CNN layer
- Applied K-Nearest Neighbors (KNN)

❌ Results:
- Accuracy ~12–13%

📌 **Conclusion:** CNN embeddings alone are insufficient for exact age prediction using KNN

---

## Key Takeaways

- VGG16 provided the most stable and reliable performance
- Larger architectures (VGG19, ResNet) did not improve results
- Small datasets (FG-Net) lead to poor generalisation
- Hybrid ML approaches (CNN + KNN) require better feature engineering

---

## Purpose of This Branch

- Document failed and exploratory experiments
- Provide comparison against final model
- Support dissertation analysis and discussion
- Enable future improvements and reproducibility

---

## Notes

- Datasets are not included due to size and licensing
- Results may vary depending on preprocessing and training configuration
- These experiments are **not optimised for production use**

---

## Suggested Improvements

- Use regression-based models (MAE instead of classification)
- Try modern architectures (EfficientNet, MobileNet)
- Improve dataset balance and augmentation
- Apply dimensionality reduction (PCA) before KNN
- Use larger and more diverse datasets

---
