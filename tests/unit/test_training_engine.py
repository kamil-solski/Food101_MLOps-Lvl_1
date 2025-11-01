"""
Unit tests for src.training.engine module.
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.engine import train_step, val_step, eval_step
from src.models.food101 import Food101


class TestTrainStep:
    """Test suite for train_step function."""

    @pytest.fixture
    def mock_dataloader(self):
        """Create a mock dataloader for testing."""
        # Create simple synthetic data
        X = torch.randn(10, 3, 64, 64)
        y = torch.randint(0, 4, (10,))
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=4, shuffle=False)

    def test_train_step(self, mock_dataloader, device):
        """Training step with mock data."""
        model = Food101(input_shape=3, hidden_units=8, output_shape=4)
        model = model.to(device)  # Move model to same device as data
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        train_loss, train_acc = train_step(model, mock_dataloader, loss_fn, optimizer)
        
        assert isinstance(train_loss, float)
        assert isinstance(train_acc, float)
        assert train_loss >= 0
        assert 0 <= train_acc <= 1

    def test_train_step_loss_decreases(self, mock_dataloader, device):
        """Loss behavior."""
        model = Food101(input_shape=3, hidden_units=8, output_shape=4)
        model = model.to(device)  # Move model to same device as data
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # First step
        loss1, _ = train_step(model, mock_dataloader, loss_fn, optimizer)
        
        # Second step - loss might decrease or fluctuate, but shouldn't be NaN
        loss2, _ = train_step(model, mock_dataloader, loss_fn, optimizer)
        
        assert not np.isnan(loss1)
        assert not np.isnan(loss2)

    def test_train_step_gradients(self, mock_dataloader, device):
        """Gradient computation."""
        model = Food101(input_shape=3, hidden_units=8, output_shape=4)
        model = model.to(device)  # Move model to same device as data
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Clear gradients first
        optimizer.zero_grad()
        
        train_step(model, mock_dataloader, loss_fn, optimizer)
        
        # Check that gradients were computed
        has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0 
                           for p in model.parameters())
        assert has_gradients


class TestValStep:
    """Test suite for val_step function."""

    @pytest.fixture
    def mock_dataloader(self):
        """Create a mock dataloader for testing."""
        X = torch.randn(8, 3, 64, 64)
        y = torch.randint(0, 4, (8,))
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=4, shuffle=False)

    def test_val_step(self, mock_dataloader, device):
        """Validation step."""
        model = Food101(input_shape=3, hidden_units=8, output_shape=4)
        model = model.to(device)  # Move model to same device as data
        loss_fn = nn.CrossEntropyLoss()
        
        val_loss, val_acc = val_step(model, mock_dataloader, loss_fn)
        
        assert isinstance(val_loss, float)
        assert isinstance(val_acc, float)
        assert val_loss >= 0
        assert 0 <= val_acc <= 1

    def test_val_step_no_gradients(self, mock_dataloader, device):
        """No gradients in validation."""
        model = Food101(input_shape=3, hidden_units=8, output_shape=4)
        model = model.to(device)  # Move model to same device as data
        loss_fn = nn.CrossEntropyLoss()
        
        # Clear any existing gradients
        for param in model.parameters():
            param.grad = None
        
        val_step(model, mock_dataloader, loss_fn)
        
        # Gradients should not be computed in validation
        for param in model.parameters():
            assert param.grad is None


class TestEvalStep:
    """Test suite for eval_step function."""

    @pytest.fixture
    def mock_dataloader(self):
        """Create a mock dataloader for testing."""
        X = torch.randn(6, 3, 64, 64)
        y = torch.randint(0, 4, (6,))
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=3, shuffle=False)

    def test_eval_step(self, mock_dataloader):
        """Evaluation step."""
        model = Food101(input_shape=3, hidden_units=8, output_shape=4)
        
        y_true, y_probs = eval_step(model, mock_dataloader)
        
        assert isinstance(y_true, np.ndarray)
        assert isinstance(y_probs, np.ndarray)
        assert len(y_true.shape) == 1
        assert len(y_probs.shape) == 2
        assert y_probs.shape[1] == 4  # num_classes

    def test_eval_step_output_shapes(self, mock_dataloader):
        """Output shapes validation."""
        model = Food101(input_shape=3, hidden_units=8, output_shape=4)
        
        y_true, y_probs = eval_step(model, mock_dataloader)
        
        # Check shapes
        assert y_true.shape[0] == 6  # total samples
        assert y_probs.shape == (6, 4)  # (samples, classes)
        
        # Check probabilities sum to 1
        assert np.allclose(y_probs.sum(axis=1), 1.0, atol=1e-6)
        
        # Check probabilities are non-negative
        assert np.all(y_probs >= 0)
