# MoldVision Toolkit - WIP

A unified Python package and CLI suite for the MoldVision Paper, offering:

- **Deep-learning models**: Three VGG-based architectures in PyTorch Lightning  
  - **Twin**: shared VGG16 backbone with two independent heads  
  - **6-Channel**: VGG16 adapted to 6-channel inputs  
  - **Default**: plain VGG16 with a custom head  
- **Data utilities**: a `LightningDataModule` with stratified 5-fold cross-validation  
- **LGBM pipeline**: end-to-end feature extraction (color histograms, LBP, contour descriptors) and LightGBM 5-fold CV  
- **CLI scripts & examples**: easy entry points for both deep-learning and LGBM workflows  

---

## Installation

From your project root (or after cloning):

```bash
pip install -e .

### License