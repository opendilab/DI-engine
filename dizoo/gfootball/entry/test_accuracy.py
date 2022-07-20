import torch
from ding.torch_utils import to_list
import logging
import math
from ding.utils.data import NaiveRLDataset
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)


def test_accuracy_in_dataset(data_path, batch_size, policy):
    """
    Overview:
        Evaluate total accuracy and accuracy of each action in dataset from
        ``datapath`` using the ``policy`` for gfootball env.
    """
    dataset = NaiveRLDataset(data_path)
    dataloader = DataLoader(dataset, batch_size)

    total_accuracy_in_dataset = []
    action_accuracy_in_dataset = {k: [] for k in range(19)}
    for _, minibatch in enumerate(dataloader):
        res = policy._forward_eval(minibatch['obs'])
        pred_action = torch.argmax(res['logit'], dim=1)
        total_accuracy = torch.sum(pred_action == minibatch['action'].squeeze(-1)).item() / minibatch['action'].shape[0]
        total_accuracy_in_dataset.append(total_accuracy)

        for action_int in to_list(torch.unique(minibatch['action'])):
            action_index = (pred_action == action_int).nonzero(as_tuple=True)[0]
            action_accuracy = (pred_action[action_index] == pred_action.view(-1)[action_index]
                               ).float().mean()
            if math.isnan(action_accuracy):
                action_accuracy = 0.0
            action_accuracy_in_dataset[action_int].append(action_accuracy)
            logging.info(f'the accuracy of action {action_int} in current train mini-batch is: {action_accuracy}')

    # accuracy statistics for debugging in discrete action space env, e.g. for gfootball
    logging.info(f'total accuracy in dataset: {torch.tensor(total_accuracy_in_dataset).mean()}')
    logging.info(
        f'accuracy of each action in dataset: { {k: torch.tensor(action_accuracy_in_dataset[k]).mean() for k in range(19)} }')
