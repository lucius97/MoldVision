# test_lightgbm_model.py
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.models import lightgbm_model as lgbm_mod

def test_load_features(tmp_path):
    vals = {
        'color': np.ones((2, 3)),
        'lbp': np.ones((2, 2)),
        'shape': np.ones((2, 1)),
        'y': np.array([0, 1]),
    }
    np.savez(tmp_path / "test.npz", **vals)
    color, lbp, shape, y = lgbm_mod.load_features(str(tmp_path / "test.npz"))
    assert color.shape == (2, 3)
    assert lbp.shape == (2, 2)
    assert shape.shape == (2, 1)
    assert y.tolist() == [0, 1]

def test_get_fold_files(tmp_path):
    (tmp_path / "train_fold_0.csv").write_text("dummy")
    (tmp_path / "val_fold_0.csv").write_text("dummy")
    (tmp_path / "unrelated_file.txt").write_text("dummy")

    train_files, val_files = lgbm_mod.get_fold_files(str(tmp_path))
    assert len(train_files) == 1
    assert train_files[0].endswith('train_fold_0.csv')
    assert len(val_files) == 1
    assert val_files[0].endswith('val_fold_0.csv')

def test_standardize_features():
    color = np.array([[1, 2], [3, 4]])
    lbp = np.array([[5, 6], [7, 8]])
    shape = np.array([[9], [10]])
    X_std, sc1, sc2, sc3 = lgbm_mod.standardize_features(color, lbp, shape)
    assert X_std.shape == (2, 5)
    # Each feature should be mean 0, std 1 after transform
    np.testing.assert_allclose(X_std.mean(axis=0), np.zeros(5))

def test_fit_and_predict_lgbm():
    # Patch lgb.train to not actually train
    X = np.random.rand(6, 3)
    y = np.array([0, 1, 2, 0, 1, 2])
    X_val = np.random.rand(3, 3)
    with patch("lightgbm.train") as mock_train:
        fake_model = MagicMock()
        fake_model.predict.return_value = np.eye(3, 5)[:, :5]  # shape (3, 5)
        mock_train.return_value = fake_model
        preds, logits, model = lgbm_mod.fit_and_predict_lgbm(
            X, y, X_val, num_class=5)
    assert logits.shape == (3, 5)
    assert preds.shape == (3,)
    assert (preds == np.argmax(logits, axis=1)).all()