# Source Code Layout

- `dataset.py` - age grouping, combined 18-class label creation, UTKFace CSV helpers.
- `preprocessing.py` - image resizing/preprocessing with failure logging.
- `model.py` - improved VGG16 model definition and fine-tuning helpers.
- `train.py` - WIKI-IMDB transfer-learning training script.
- `finetune.py` - UTKFace fine-tuning script.
- `evaluate.py` - FG-Net/test-set evaluation script.
- `knn.py` - experimental KNN classifier using intermediate CNN features.
- `utils.py` - reproducibility and file helpers.
