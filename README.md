# Moldvision Toolkit

## Project Structure
``` 
.
├── __init__.py               # Imports main models for easy access
├── default_model.py          # Default VGG-based PyTorch Lightning classifier
├── database.py               # Dataset and data module for image loading & splits
├── lgbm_feature_extract.py   # Utilities for feature extraction for traditional ML
├── lightgbm_model.py         # LightGBM functions: training, prediction, utilities
├── moldvision.py             # Twin-branch VGG16 Lightning model for image pairs
├── sixchan_model.py          # Lightning model for 6-channel image inputs
├── LICENSE                   # Project license
└── .gitignore                # Standard Python ignore rules
```
## Main Components
- **Deep Learning Models**
    - (moldvision.py): Twin-branch VGG16-based classifier for paired images. `MoldVision`
    - (default_model.py): Standard VGG-based classifier. `MoldVGG_Default`
    - (sixchan_model.py): Model supporting 6-channel image input. `MoldVGG_6Chan`

- **Traditional Machine Learning**
    - : Feature loading, k-fold data partitioning, LightGBM training and prediction helpers. `lightgbm_model.py`

- **Feature Engineering**
    - : Functions for extracting color, texture, and shape features from images and masks. `lgbm_feature_extract.py`

- **Data Handling**
    - :
        - : Custom dataset class for image and annotation access. `MoldDataset`
        - : PyTorch Lightning DataModule for easy train/val/test/predict splitting and transformation. `MoldDataModule`

`database.py`

## Installation
1. Install Python 3.12+.
2. Install required packages:
``` bash
    pip install torch torchvision pytorch-lightning lightgbm numpy pandas opencv-python scikit-learn pillow
```
## Usage
### Deep Learning Example
``` python
from moldvision import MoldVision

model = MoldVision(opt_lr=0.01, lr_pat=5, batch_size=32, num_epochs=20)
# Then use with a PyTorch Lightning Trainer
```
### LightGBM Pipeline Example
``` python
from lightgbm_model import fit_and_predict_lgbm, load_features

# Load features, split for k-folds, train and predict
```
### Feature Extraction Example
``` python
from lgbm_feature_extract import extract_and_save_features

# Extract features for classic ML models
```
### Dataset Usage
``` python
from database import MoldDataModule

dm = MoldDataModule(...)
dm.prepare_data()
```
## License
This project is licensed under the terms described in the [LICENSE](./LICENSE) file.
