import unittest
import numpy as np
import torch
import treetensor.torch as ttorch

from ding.policy.common_utils import default_preprocess_learn

shape_test = [
    [2],
    [1],
]

dtype_test = [
    "int32",
    "int64",
    "float32",
    "float64",
]

data_type_test = [
    "numpy",
    "torch",
    "treetensor",
]


def get_action(shape, dtype, class_type):
    if class_type == "numpy":
        return np.random.randn(*shape).astype(dtype)
    else:
        if dtype == "int32":
            dtype = torch.int32
        elif dtype == "int64":
            dtype = torch.int64
        elif dtype == "float16":
            dtype = torch.float16
        elif dtype == "float32":
            dtype = torch.float32
        elif dtype == "float64":
            dtype = torch.float64

        if class_type == "torch":
            return torch.randn(*shape).type(dtype)
        elif class_type == "treetensor":
            return ttorch.randn(*shape).type(dtype)


class TestCommonUtils(unittest.TestCase):

    def test_default_preprocess_learn_action(self):

        for shape in shape_test:
            for dtype in dtype_test:
                for data_type in data_type_test:

                    data = [
                        {
                            'obs': np.random.randn(4, 84, 84),
                            'action': get_action(shape, dtype, data_type),
                            'reward': 1.0,
                            'next_obs': np.random.randn(4, 84, 84),
                            'done': False,
                            'weight': 1.0,
                            'value': 1.0,
                            'adv': 1.0,
                        } for _ in range(10)
                    ]
                    use_priority_IS_weight = False
                    use_priority = False
                    use_nstep = False
                    ignore_done = False
                    data = default_preprocess_learn(data, use_priority_IS_weight, use_priority, use_nstep, ignore_done)

                    self.assertEqual(data['obs'].shape, torch.Size([10, 4, 84, 84]))
                    if dtype in ["int32", "int64"] and shape[0] == 1:
                        self.assertEqual(data['action'].shape, torch.Size([10]))
                    else:
                        self.assertEqual(data['action'].shape, torch.Size([10, *shape]))
                    self.assertEqual(data['reward'].shape, torch.Size([10]))
                    self.assertEqual(data['next_obs'].shape, torch.Size([10, 4, 84, 84]))
                    self.assertEqual(data['done'].shape, torch.Size([10]))
                    self.assertEqual(data['weight'].shape, torch.Size([10]))
                    self.assertEqual(data['value'].shape, torch.Size([10]))
                    self.assertEqual(data['adv'].shape, torch.Size([10]))

    def test_default_preprocess_learn_reward_done_adv_1d(self):

        data = [
            {
                'obs': np.random.randn(4, 84, 84),
                'action': np.random.randn(2),
                'reward': np.array([1.0]),
                'next_obs': np.random.randn(4, 84, 84),
                'done': False,
                'value': np.array([1.0]),
                'adv': np.array([1.0]),
            } for _ in range(10)
        ]
        use_priority_IS_weight = False
        use_priority = False
        use_nstep = False
        ignore_done = False
        data = default_preprocess_learn(data, use_priority_IS_weight, use_priority, use_nstep, ignore_done)

        self.assertEqual(data['reward'].shape, torch.Size([10]))
        self.assertEqual(data['done'].shape, torch.Size([10]))
        self.assertEqual(data['weight'], None)
        self.assertEqual(data['value'].shape, torch.Size([10]))
        self.assertEqual(data['adv'].shape, torch.Size([10]))

    def test_default_preprocess_learn_ignore_done(self):
        data = [
            {
                'obs': np.random.randn(4, 84, 84),
                'action': np.random.randn(2),
                'reward': np.array([1.0]),
                'next_obs': np.random.randn(4, 84, 84),
                'done': True,
                'value': np.array([1.0]),
                'adv': np.array([1.0]),
            } for _ in range(10)
        ]
        use_priority_IS_weight = False
        use_priority = False
        use_nstep = False
        ignore_done = True
        data = default_preprocess_learn(data, use_priority_IS_weight, use_priority, use_nstep, ignore_done)

        self.assertEqual(data['done'].dtype, torch.float32)
        self.assertEqual(torch.sum(data['done']), 0)

    def test_default_preprocess_learn_use_priority_IS_weight(self):
        data = [
            {
                'obs': np.random.randn(4, 84, 84),
                'action': np.random.randn(2),
                'reward': 1.0,
                'next_obs': np.random.randn(4, 84, 84),
                'done': False,
                'priority_IS': 1.0,
                'value': 1.0,
                'adv': 1.0,
            } for _ in range(10)
        ]
        use_priority_IS_weight = True
        use_priority = True
        use_nstep = False
        ignore_done = False
        data = default_preprocess_learn(data, use_priority_IS_weight, use_priority, use_nstep, ignore_done)

        self.assertEqual(data['weight'].shape, torch.Size([10]))
        self.assertEqual(torch.sum(data['weight']), torch.tensor(10.0))

    def test_default_preprocess_learn_nstep(self):
        data = [
            {
                'obs': np.random.randn(4, 84, 84),
                'action': np.random.randn(2),
                'reward': np.array([1.0, 2.0, 0.0]),
                'next_obs': np.random.randn(4, 84, 84),
                'done': False,
                'value': 1.0,
                'adv': 1.0,
            } for _ in range(10)
        ]
        use_priority_IS_weight = False
        use_priority = False
        use_nstep = True
        ignore_done = False
        data = default_preprocess_learn(data, use_priority_IS_weight, use_priority, use_nstep, ignore_done)

        self.assertEqual(data['reward'].shape, torch.Size([3, 10]))
        self.assertEqual(data['reward'][0][0], torch.tensor(1.0))
        self.assertEqual(data['reward'][1][0], torch.tensor(2.0))
        self.assertEqual(data['reward'][2][0], torch.tensor(0.0))


if __name__ == '__main__':
    A = TestCommonUtils()
    A.test_default_preprocess_learn_nstep()
