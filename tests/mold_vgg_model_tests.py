# tests/test_vgg_models.py

import torch
import pytest

from src.models import MoldVGG_Default, MoldVGG_6Chan, MoldVision


# Sample config dictionary
sample_config = {
    'LEARNING_RATE': 0.001,
    'BATCH_SIZE': 2,
    'LR_PATIENCE': 1,
    'NUM_EPOCHS': 1,
    'DROPOUT_RATE': 0.3,
    'CLASS_NAMES': ['a', 'b'],
    'WEIGHT_DECAY': 0.00001
}

# ---- MoldVGG_Default Tests ----

def test_moldvggdefault_init():
    model = MoldVGG_Default(sample_config)
    assert hasattr(model, 'loss_fn')
    assert hasattr(model, 'base')

def test_moldvggdefault_forward_shape():
    model = MoldVGG_Default(sample_config)
    x = torch.randn(2, 3, 224, 224)
    y = model.forward(x)
    assert y.shape[0] == 2

@pytest.mark.parametrize("method_name", ["training_step", "validation_step", "test_step", "predict_step"])
def test_moldvggdefault_steps(method_name):
    model = MoldVGG_Default(sample_config)
    x = torch.randn(2, 3, 224, 224)
    y = torch.randint(0, 2, (2,))
    method = getattr(model, method_name)
    if method_name == "predict_step":
        batch = (x, y, ["name1", "name2"])
    else:
        batch = (x, y)
    output = method(batch, 0)  # Supply batch_idx=0
    if method_name == "predict_step":
        assert output['logits'].shape[0] == 2
        assert output['labels'].shape[0] == 2
        assert output['name'] == ["name1", "name2"]
    elif method_name == "test_step":
        assert output['logits'].shape[0] == 2
        assert output['labels'].shape[0] == 2
    elif method_name == "validation_step":
        assert output is None
    else:
        assert output is not None

# ---- MoldVGG_6Chan Tests ----

def test_moldvgg6chan_init():
    model = MoldVGG_6Chan(sample_config)
    assert hasattr(model, 'loss_fn')
    assert hasattr(model, 'head')
    assert hasattr(model, 'classifier')

def test_moldvgg6chan_forward_shape():
    model = MoldVGG_6Chan(sample_config)
    x = torch.randn(2, 6, 224, 224)
    y = model.forward(x)
    assert y.shape[0] == 2

@pytest.mark.parametrize("method_name", ["training_step", "validation_step", "test_step", "predict_step"])
def test_moldvgg6chan_steps(method_name):
    model = MoldVGG_6Chan(sample_config)
    x = torch.randn(2, 6, 224, 224)
    y = torch.randint(0, 2, (2,))
    if method_name == "predict_step":
        batch = (x, y, ["name1", "name2"])
    else:
        batch = (x, y)
    method = getattr(model, method_name)
    output = method(batch, 0)  # Supply batch_idx=0
    if method_name == "predict_step":
        assert output['logits'].shape[0] == 2
        assert output['labels'].shape[0] == 2
        assert output['name'] == ["name1", "name2"]
    elif method_name == "test_step":
        assert output['logits'].shape[0] == 2
        assert output['labels'].shape[0] == 2
    elif method_name == "validation_step":
        assert output is None  # by design, val step returns None
    else:
        assert output is not None


# ---- MoldVision Tests ----

def test_moldvision_init():
    model = MoldVision(sample_config)
    assert hasattr(model, 'loss_fn')
    assert hasattr(model, 'branch_top')
    assert hasattr(model, 'branch_bottom')
    assert hasattr(model, 'classifier')


def test_moldvision_forward_shape():
    model = MoldVision(sample_config)
    x_t = torch.randn(2, 3, 224, 224)
    x_b = torch.randn(2, 3, 224, 224)
    y = model.forward(x_t, x_b)
    assert y.shape[0] == 2

@pytest.mark.parametrize("method_name", ["training_step", "validation_step", "test_step", "predict_step"])
def test_moldvision_steps(method_name):
    model = MoldVision(sample_config)
    x_t = torch.randn(2, 3, 224, 224)
    x_b = torch.randn(2, 3, 224, 224)
    y = torch.randint(0, 2, (2,))
    if method_name == "predict_step":
        batch = (x_t, x_b, y, ["name1", "name2"])
    else:
        batch = (x_t, x_b, y)
    method = getattr(model, method_name)
    output = method(batch, 0)  # Supply batch_idx=0
    if method_name == "predict_step":
        assert output['logits'].shape[0] == 2
        assert output['labels'].shape[0] == 2
        assert output['name'] == ["name1", "name2"]
    elif method_name == "test_step":
        assert output['logits'].shape[0] == 2
        assert output['labels'].shape[0] == 2
    elif method_name == "validation_step":
        assert output is None  # by design, val step returns None
    else:
        assert output is not None