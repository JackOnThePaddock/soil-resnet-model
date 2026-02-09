"""Tests for NationalSoilNet model architecture."""

import torch
import pytest

from src.models.resnet import NationalSoilNet, ResidualBlock, DEFAULT_TARGETS


class TestResidualBlock:
    def test_output_shape(self):
        block = ResidualBlock(dim=128, dropout=0.2)
        x = torch.randn(4, 128)
        out = block(x)
        assert out.shape == (4, 128)

    def test_skip_connection(self):
        """Output should differ from input (transformation applied)."""
        block = ResidualBlock(dim=64, dropout=0.0)
        block.eval()
        x = torch.randn(2, 64)
        out = block(x)
        assert not torch.allclose(x, out)


class TestNationalSoilNet:
    def test_default_targets(self):
        model = NationalSoilNet()
        assert model.target_names == DEFAULT_TARGETS
        assert model.num_targets == 7

    def test_forward_dict_output(self):
        model = NationalSoilNet(input_dim=64, hidden_dim=128)
        model.eval()
        x = torch.randn(4, 64)
        outputs = model(x)
        assert isinstance(outputs, dict)
        assert set(outputs.keys()) == set(DEFAULT_TARGETS)
        for target, pred in outputs.items():
            assert pred.shape == (4,), f"{target} shape mismatch"

    def test_forward_stacked_output(self):
        model = NationalSoilNet(input_dim=64, hidden_dim=128)
        model.eval()
        x = torch.randn(8, 64)
        stacked = model.forward_stacked(x)
        assert stacked.shape == (8, 7)

    def test_custom_targets(self):
        targets = ["ph", "cec"]
        model = NationalSoilNet(target_names=targets)
        model.eval()
        x = torch.randn(2, 64)
        out = model(x)
        assert set(out.keys()) == {"ph", "cec"}
        stacked = model.forward_stacked(x)
        assert stacked.shape == (2, 2)

    def test_custom_dimensions(self):
        model = NationalSoilNet(input_dim=32, hidden_dim=64, num_res_blocks=3, dropout=0.5)
        model.eval()
        x = torch.randn(3, 32)
        stacked = model.forward_stacked(x)
        assert stacked.shape == (3, 7)

    def test_single_sample(self):
        """Model should handle batch size of 1."""
        model = NationalSoilNet()
        model.eval()
        x = torch.randn(1, 64)
        stacked = model.forward_stacked(x)
        assert stacked.shape == (1, 7)

    def test_gradient_flow(self):
        """Gradients should flow through the model."""
        model = NationalSoilNet()
        x = torch.randn(4, 64, requires_grad=True)
        stacked = model.forward_stacked(x)
        loss = stacked.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (4, 64)
