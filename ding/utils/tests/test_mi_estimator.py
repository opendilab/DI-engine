import pytest
import numpy as np
import torch
from ding.utils.mi_estimator import ContrastiveLoss
from torch.utils.data import TensorDataset, DataLoader


@pytest.mark.unittest
@pytest.mark.parametrize('noise', [0.1, 1.0, 3.0])
@pytest.mark.parametrize('dims', [[16], [1, 16, 16]])
def test_infonce_loss(noise, dims):
    batch_size = 128
    N_batch = 10
    x_dim = [batch_size * N_batch] + dims

    encode_shape = 16
    x = np.random.normal(0, 1, size=x_dim)
    y = x ** 2 + noise * np.random.normal(0, 1, size=x_dim)

    estimator = ContrastiveLoss(x.shape[1:], y.shape[1:], encode_shape=encode_shape)
    dataset = TensorDataset(torch.Tensor(x), torch.Tensor(y))
    dataloader = DataLoader(dataset, batch_size=batch_size)
    optimizer = torch.optim.Adam(estimator.parameters(), lr=3e-4)

    for epoch in range(3):
        train_loss = 0.
        test_loss = 0.
        for inputs in dataloader:
            x, y = inputs
            optimizer.zero_grad()
            loss = estimator.forward(x, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        with torch.no_grad():
            for inputs in dataloader:
                x, y = inputs
                outputs = estimator.forward(x, y)
                test_loss += outputs.item()
