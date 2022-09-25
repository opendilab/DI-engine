import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from torchvision import transforms
from dizoo.multi_mnist import MultiMNIST
from ding.model.template.lenet import MultiLeNetR, MultiLeNetO
from ding.torch_utils.optimizer_helper import PCGrad
import logging


# ------------------ CHANGE THE CONFIGURATION -------------
PATH = './dataset'
LR = 0.0005
BATCH_SIZE = 256
NUM_EPOCHS = 100
TASKS = ['R', 'L']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ---------------------------------------------------------

def create_logger(name):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


accuracy = lambda logits, gt: ((logits.argmax(dim=-1) == gt).float()).mean()
to_dev = lambda inp, dev: [x.to(dev) for x in inp]
logger = create_logger('Main')

global_transformer = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307, ), (0.3081, ))])

train_dst = MultiMNIST(PATH,
                       train=True,
                       download=True,
                       transform=global_transformer,
                       multi=True)
train_loader = torch.utils.data.DataLoader(train_dst,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           num_workers=4)

val_dst = MultiMNIST(PATH,
                     train=False,
                     download=True,
                     transform=global_transformer,
                     multi=True)
val_loader = torch.utils.data.DataLoader(val_dst,
                                         batch_size=100,
                                         shuffle=True,
                                         num_workers=1)
nets = {
    'rep': MultiLeNetR().to(DEVICE),
    'L': MultiLeNetO().to(DEVICE),
    'R': MultiLeNetO().to(DEVICE)
}
param = [p for v in nets.values() for p in list(v.parameters())]
optimizer = torch.optim.Adam(param, lr=LR)
optimizer = PCGrad(optimizer)

for ep in range(NUM_EPOCHS):
    for net in nets.values():
        net.train()
    for batch in train_loader:
        mask = None
        optimizer.zero_grad()
        img, label_l, label_r = to_dev(batch, DEVICE)
        rep, mask = nets['rep'](img, mask)
        out_l, mask_l = nets['L'](rep, None)
        out_r, mask_r = nets['R'](rep, None)

        losses = [F.nll_loss(out_l, label_l), F.nll_loss(out_r, label_r)]
        optimizer.pc_backward(losses)
        # sum(losses).backward()
        optimizer.step()

    losses, acc = [], []
    for net in nets.values():
        net.eval()
    for batch in val_loader:
        img, label_l, label_r = to_dev(batch, DEVICE)
        mask = None
        rep, mask = nets['rep'](img, mask)
        out_l, mask_l = nets['L'](rep, None)
        out_r, mask_r = nets['R'](rep, None)

        losses.append([
            F.nll_loss(out_l, label_l).item(),
            F.nll_loss(out_r, label_r).item()
        ])
        acc.append(
            [accuracy(out_l, label_l).item(),
             accuracy(out_r, label_r).item()])
    losses, acc = np.array(losses), np.array(acc)
    logger.info('epoches {}/{}: loss (left, right) = {:5.4f}, {:5.4f}'.format(
        ep, NUM_EPOCHS, losses[:,0].mean(), losses[:,1].mean()))
    logger.info(
        'epoches {}/{}: accuracy (left, right) = {:5.3f}, {:5.3f}'.format(
            ep, NUM_EPOCHS, acc[:,0].mean(), acc[:,1].mean()))