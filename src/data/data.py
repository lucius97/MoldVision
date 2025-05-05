import os
from typing import List, Dict, Tuple, Optional
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
import pytorch_lightning as pl

def encode_column(df: pd.DataFrame, column: str) -> Tuple[List, Dict]:
    """Generic encoder for categorical columns."""
    uniques = df[column].unique()
    return list(uniques), {val: idx for idx, val in enumerate(uniques)}

def validate_group_split(df: pd.DataFrame, group_col: str, train_idx, val_idx, class_col: Optional[str] = None):
    """
    Validates the group and class distribution in train and test splits.

    Parameters:
        df (DataFrame): The dataset DataFrame.
        group_col (str): Column name for groups.
        train_idx (list): Indices for training set.
        val_idx (list): Indices for Validation set.
        class_col (str, optional): Column name for class. Default is None.
    """
    try:
        groups = df[group_col]
        common = set(groups.iloc[train_idx]) & set(groups.iloc[val_idx])
        ratio = len(common) / len(groups.unique())
        print(f"Common groups ({len(common)}): {common}\nRatio: {ratio:.3f}")
        if class_col:
            tc = pd.Series(df[class_col].iloc[train_idx]).value_counts(normalize=True)
            vc = pd.Series(df[class_col].iloc[val_idx]).value_counts(normalize=True)
            print("Train class dist:\n", tc)
            print("Val class dist:\n", vc)
        return {"common_groups_ratio": ratio}

    except KeyError as e:
        print(f"KeyError: {e} - Check your column names.")
    except IndexError as e:
        print(f"IndexError: {e} - Check your indices.")

class MoldDataset(Dataset):
  """Unified dataset for 'twin', '6Chan', or single-channel modes."""

  def __init__(
          self,
          root: str,
          main_csv: str,
          fold_csv: str,
          mode: str = 'default',
          transform: Optional[transforms.Compose] = None
  ):

      self.root = root
      self.df_main = pd.read_csv(main_csv)
      self.df_fold = pd.read_csv(fold_csv)
      self.mode = mode
      self.transform = transform

      self.classes, self.class_to_idx = encode_column(self.df_main, 'class')
      # preload image paths and indices
      self.items = [
          (os.path.join(self.root, row['class'], row['file_name']), row['file_name'])
          for _, row in self.df_fold.iterrows()
      ]
      self.cache = {}

  def __len__(self):
      return len(self.items)

  def __getitem__(self, idx: int):
      if idx in self.cache:
          return self.cache[idx]

      front_path, fname = self.items[idx]
      img_front = Image.open(front_path)

      # back image lookup
      main_idx = self.df_main[self.df_main['file_name'] == fname].index[0]
      back_row = self.df_main.iloc[main_idx + 3]
      back_path = os.path.join(self.root, back_row['class'], back_row['file_name'])
      img_back = Image.open(back_path)

      if self.transform:
          img_front = self.transform(img_front)
          img_back = self.transform(img_back)

      class_idx = self.class_to_idx[self.df_fold.loc[idx, 'class']]
      if self.mode == 'twin':
          result = ((img_front, img_back), class_idx, fname)
      elif self.mode == '6Chan':
          tensor_front = img_front if isinstance(img_front, torch.Tensor) else transforms.ToTensor()(img_front)
          tensor_back = img_back if isinstance(img_back, torch.Tensor) else transforms.ToTensor()(img_back)
          result = (torch.cat((tensor_front, tensor_back), dim=0), class_idx)
      else:
          result = (img_front, class_idx)

      self.cache[idx] = result
      return result

  def get_classes(self):
      return self.df_fold['class'].values

  def get_groups(self):
      return self.df_fold['platenr'].values

  def class_from_idx(self, idx):
      inv = {v: k for k, v in self.class_to_idx.items()}
      return inv[idx]

class MoldDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        fold_idx: int = 0,
        num_folds: int = 5,
        num_workers: int = 0,
        pin_memory: bool = True,
        transforms: Optional[transforms.Compose] = None,
        model_type: str = 'default'
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['transforms'])
        self.transforms = transforms
        self.data_train = self.data_val = self.data_test = self.data_predict = None

    def prepare_data(self, side='Both', split_day=None):
        folds_dir = os.path.join(self.hparams.data_dir, 'folds')
        if not os.path.exists(os.path.join(folds_dir, 'test_fold_0.csv')):
            os.makedirs(folds_dir, exist_ok=True)
            data = pd.read_csv(os.path.join(self.hparams.data_dir, 'dataset.csv'))
            # optional filtering omitted for brevity
            kf = StratifiedGroupKFold(n_splits=self.hparams.num_folds)
            groups = data['platenr']; classes = data['class']
            for fold, (rest, test) in enumerate(kf.split(data, classes, groups)):
                data.iloc[test].to_csv(f'{folds_dir}/test_fold_{fold}.csv', index=False)
                tr, va = next(kf.split(data.iloc[rest], data['class'].iloc[rest], groups.iloc[rest]))
                data.iloc[rest].iloc[tr].to_csv(f'{folds_dir}/train_fold_{fold}.csv', index=False)
                data.iloc[rest].iloc[va].to_csv(f'{folds_dir}/val_fold_{fold}.csv', index=False)
                validate_group_split(data.iloc[rest], 'platenr', tr, va, 'class')

    def setup(self, stage: Optional[str] = None):
        base = self.hparams.data_dir
        paths = lambda kind: os.path.join(base, 'folds', f"{kind}_fold_{self.hparams.fold_idx}.csv")
        if stage in (None, 'fit'):
            self.data_train = MoldDataset(base, os.path.join(base, 'dataset.csv'), paths('train'), self.hparams.model_type, self.transforms)
            self.data_val   = MoldDataset(base, os.path.join(base, 'dataset.csv'), paths('val'),   self.hparams.model_type, self.transforms)
        if stage in (None, 'test'):
            self.data_test  = MoldDataset(base, os.path.join(base, 'dataset.csv'), paths('test'),  self.hparams.model_type, self.transforms)
        if stage in (None, 'predict'):
            self.data_predict = MoldDataset(base, os.path.join(base, 'dataset.csv'), paths('test'), self.hparams.model_type, self.transforms)

    def _dataloader(self, ds: Dataset, shuffle=False):
        return DataLoader(ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory, shuffle=shuffle)

    def train_dataloader(self): return self._dataloader(self.data_train, shuffle=True)
    def val_dataloader(self):   return self._dataloader(self.data_val)
    def test_dataloader(self):  return self._dataloader(self.data_test)
    def predict_dataloader(self): return self._dataloader(self.data_predict)
