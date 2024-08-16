import pytest
import torch
import torch.nn as nn

from src.train.utils.models import ConvNet, PretrainedResNet, get_model


class TestConvNet:
    @staticmethod
    def test_call():
        x = torch.rand(32, 3, 32, 32)
        model = ConvNet()
        yhat = model(x)
        assert list(yhat.shape) == [32, 2]
        assert isinstance(yhat, torch.Tensor)


class TestPretrainedResNet:
    @staticmethod
    def test_call():
        x = torch.rand(32, 3, 32, 32)
        model = PretrainedResNet()
        yhat = model(x)
        assert list(yhat.shape) == [32, 2]
        assert isinstance(yhat, torch.Tensor)


def test_get_model():
    model = get_model("conv_net")
    assert isinstance(model, nn.Module)
    model = get_model("pretrained_res_net")
    assert isinstance(model, nn.Module)
    with pytest.raises(ValueError):
        get_model("invalid_model")
