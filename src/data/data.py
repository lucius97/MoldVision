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
        fold_csv: str,    # This is (usually) a CSV with only the data of this fold!
        mode: str = 'default',
        transform: Optional[transforms.Compose] = None,
        group_col: str = 'ID',
        pos_col: str = 'rotation_index',
        main_csv: str = None,  # Direct path to the main database csv
    ):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.group_col = group_col
        self.pos_col = pos_col

        # Load the fold's data (train/val/test for this fold)
        self.df = pd.read_csv(fold_csv)

        # Optionally load the full database csv for lookups (top/bottom matching, etc)
        # If not provided, just use self.df. If provided, use for back-matching.
        self.df_main = pd.read_csv(main_csv) if main_csv is not None else self.df

        # Encode classes (use only those present in this fold's csv)
        self.classes, self.class_to_idx = encode_column(self.df, 'class')

        # Precompute unique groups, if desired
        self.groups = self.df[self.group_col].drop_duplicates().tolist()

        # Preload file information: only top images (assuming front), and their filenames
        self.items = [
            (os.path.join(self.root, row['filename']), row['filename'])
            for _, row in self.df.iterrows() if row['top/bottom'] == 'top'
        ]
        self.cache = {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        if idx in self.cache:
            return self.cache[idx]

        top_path, fname = self.items[idx]
        img_front = Image.open(top_path)

        # Find bottom (back) image using filename mapping
        # e.g. replace "top" with "bottom" in filename, or join on other fields from CSV
        row = self.df[self.df['filename'] == fname].iloc[0]
        bottom_row = self.df_main[
            (self.df_main['ID'] == row['ID']) &
            (self.df_main['day'] == row['day']) &
            (self.df_main['rotation_index'] == row['rotation_index']) &
            (self.df_main['top/bottom'] == 'bottom')
        ]
        if not bottom_row.empty:
            back_fname = bottom_row.iloc[0]['filename']
            back_path = os.path.join(self.root, back_fname)
            img_back = Image.open(back_path)
        else:
            img_back = img_front  # fallback: could also raise an error

        if self.transform:
            img_front = self.transform(img_front)
            img_back = self.transform(img_back)

        class_idx = self.class_to_idx[row['class']]
        if self.mode == 'twin':
            result = ((img_front, img_back), class_idx, fname)
        elif self.mode == '6Chan':
            tensor_front = img_front if isinstance(img_front, torch.Tensor) else transforms.ToTensor()(img_front)
            tensor_back = img_back if isinstance(img_back, torch.Tensor) else transforms.ToTensor()(img_back)
            result = (torch.cat((tensor_front, tensor_back), dim=0), class_idx, fname)
        else:
            result = (img_front, class_idx, fname)

        self.cache[idx] = result
        return result

    def get_classes(self):
        return self.df['class'].values

    def get_groups(self):
        return self.df[self.group_col].values

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

    def prepare_data(
        self, 
        stratify_col=None, 
        group_col=None, 
        filter_fn=None
    ):
        """
        Create cross-validation folds if not already present.
        fold CSVs are written to: <data_dir>/folds/train/val/test_fold_{i}.csv

        Args:
            stratify_col: name of column to stratify on (default: tries 'class' if present, else None)
            group_col: name of group column (default: tries 'ID' if present, else None)
            filter_fn: optional function to filter the dataframe before splitting
        """
        folds_dir = os.path.join(self.hparams.data_dir, 'folds')
        exist_test_fold = os.path.exists(os.path.join(folds_dir, 'test_fold_0.csv'))

        if exist_test_fold:
            print("Folds already exist, skipping fold creation.")
            return

        os.makedirs(folds_dir, exist_ok=True)
        data_path = os.path.join(self.hparams.data_dir, 'dataset.csv')
        data = pd.read_csv(data_path)

        # Optionally filter data before making folds
        if filter_fn is not None:
            data = filter_fn(data)

        # Guess default stratify/group columns if not specified; else, use provided
        possible_strat_cols = ['class', 'label']
        possible_group_cols = ['ID', 'group']

        if stratify_col is None:
            stratify_col = next((col for col in possible_strat_cols if col in data.columns), None)
        if group_col is None:
            group_col = next((col for col in possible_group_cols if col in data.columns), None)

        stratify_vals = data[stratify_col] if stratify_col else None
        group_vals = data[group_col] if group_col else None

        # Choose the right splitter
        if stratify_vals is not None and group_vals is not None:
            splitter = StratifiedGroupKFold(n_splits=self.hparams.num_folds)
            split_args = (data, stratify_vals, group_vals)
        elif stratify_vals is not None:
            from sklearn.model_selection import StratifiedKFold
            splitter = StratifiedKFold(n_splits=self.hparams.num_folds)
            split_args = (data, stratify_vals)
        elif group_vals is not None:
            from sklearn.model_selection import GroupKFold
            splitter = GroupKFold(n_splits=self.hparams.num_folds)
            split_args = (data, group_vals)
        else:
            from sklearn.model_selection import KFold
            splitter = KFold(n_splits=self.hparams.num_folds)
            split_args = (data,)

        for fold, (trainval_idx, test_idx) in enumerate(splitter.split(*split_args)):
            rest_df = data.iloc[trainval_idx]
            test_df = data.iloc[test_idx]
            test_df.to_csv(f'{folds_dir}/test_fold_{fold}.csv', index=False)

            # Further split train+val using same logic
            if stratify_vals is not None and group_vals is not None:
                sub_splitter = StratifiedGroupKFold(n_splits=self.hparams.num_folds)
                sub_args = (rest_df, rest_df[stratify_col], rest_df[group_col])
            elif stratify_vals is not None:
                from sklearn.model_selection import StratifiedKFold
                sub_splitter = StratifiedKFold(n_splits=self.hparams.num_folds)
                sub_args = (rest_df, rest_df[stratify_col])
            elif group_vals is not None:
                from sklearn.model_selection import GroupKFold
                sub_splitter = GroupKFold(n_splits=self.hparams.num_folds)
                sub_args = (rest_df, rest_df[group_col])
            else:
                from sklearn.model_selection import KFold
                sub_splitter = KFold(n_splits=self.hparams.num_folds)
                sub_args = (rest_df,)

            train_idx, val_idx = next(sub_splitter.split(*sub_args))
            rest_df.iloc[train_idx].to_csv(f'{folds_dir}/train_fold_{fold}.csv', index=False)
            rest_df.iloc[val_idx].to_csv(f'{folds_dir}/val_fold_{fold}.csv', index=False)

        print(f"Fold CSVs written to {folds_dir}")

    def setup(self, stage: Optional[str] = None, main_csv: Optional[str] = None):
        """
        Sets up dataset splits (train/val/test/predict) for each stage.

        Args:
            stage: The stage for which to set up data. One of None, 'fit', 'test', 'predict'.
            main_csv: Path to the main/reference CSV file (if required by the dataset).
                      If None, a default is used.

        This method loads each split from its respective fold CSV,
        and optionally allows overriding the main CSV to make the
        data setup agnostic to a specific file name or structure.
        """
        base = self.hparams.data_dir
        get_fold = lambda kind: os.path.join(base, 'folds', f"{kind}_fold_{self.hparams.fold_idx}.csv")
        main_db_csv = main_csv if main_csv is not None else os.path.join(base, "main_database.csv")

        if stage in (None, 'fit'):
            self.data_train = MoldDataset(
                base, get_fold('train'), self.hparams.model_type, self.transforms, main_csv=main_db_csv
            )
            self.data_val = MoldDataset(
                base, get_fold('val'), self.hparams.model_type, self.transforms, main_csv=main_db_csv
            )
        if stage in (None, 'test'):
            self.data_test = MoldDataset(
                base, get_fold('test'), self.hparams.model_type, self.transforms, main_csv=main_db_csv
            )
        if stage in (None, 'predict'):
            self.data_predict = MoldDataset(
                base, get_fold('test'), self.hparams.model_type, self.transforms, main_csv=main_db_csv
            )

    def _dataloader(self, ds: Dataset, shuffle=False):
        return DataLoader(ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory, shuffle=shuffle)

    def train_dataloader(self): return self._dataloader(self.data_train, shuffle=True)
    def val_dataloader(self):   return self._dataloader(self.data_val)
    def test_dataloader(self):  return self._dataloader(self.data_test)
    def predict_dataloader(self): return self._dataloader(self.data_predict)