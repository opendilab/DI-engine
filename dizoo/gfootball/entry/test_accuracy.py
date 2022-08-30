import torch
import logging
import math
from ding.torch_utils import to_list
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
        policy_output = policy._forward_eval(minibatch['obs'])
        pred_action = policy_output['action']
        total_accuracy = (pred_action == minibatch['action'].view(-1)).float().mean()
        total_accuracy_in_dataset.append(total_accuracy)

        for action_unique in to_list(torch.unique(minibatch['action'])):
            # find the index where action is `action_unique` in `pred_action`
            action_index = (pred_action == action_unique).nonzero(as_tuple=True)[0]
            action_accuracy = (pred_action[action_index] == minibatch['action'].view(-1)[action_index]).float().mean()
            if math.isnan(action_accuracy):
                action_accuracy = 0.0
            action_accuracy_in_dataset[action_unique].append(action_accuracy)
            # logging.info(f'the accuracy of action {action_unique} in current train mini-batch is: {action_accuracy}')

    logging.info(f'total accuracy in dataset is: {torch.tensor(total_accuracy_in_dataset).mean().item()}')
    logging.info(
        f'accuracy of each action in dataset is (nan means the action does not appear in the dataset): '
        f'{ {k: torch.tensor(action_accuracy_in_dataset[k]).mean().item() for k in range(19)} }'
    )
