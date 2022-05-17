import pytest
import numpy as np
import torch
from ding.utils.mi_estimator import ContrastiveLoss
from torch.utils.data import TensorDataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter


@pytest.mark.benchmark
@pytest.mark.parametrize('noise', [0.1, 1.0, 3.0])
def test_infonce_loss(noise):
    batch_size = 64
    N_batch = 100
    x_dim = [batch_size * N_batch, 16]

    embed_dim = 16
    x = np.random.normal(0, 1, size=x_dim)
    y = x ** 2 + noise * np.random.normal(0, 1, size=x_dim)

    estimator = ContrastiveLoss(x.shape[1:], y.shape[1:], embed_dim=embed_dim)
    dataset = TensorDataset(torch.Tensor(x), torch.Tensor(y))
    dataloader = DataLoader(dataset, batch_size=batch_size)
    optimizer = torch.optim.Adam(estimator.parameters(), lr=3e-4)
    # writer = SummaryWriter()
    for epoch in range(10):
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
                test_loss += outputs

        # writer.add_scalar('Loss/train', train_loss / N_batch, epoch)
        # writer.add_scalar('Loss/train', test_loss / N_batch, epoch)
        # print('epoch {}: train loss {:.4f}, test_loss {:.4f}'.format(epoch, \
        # train_loss / N_batch, test_loss / N_batch))
    # writer.close()
