'''
Copyright 2020 Sensetime X-lab. All Rights Reserved
Main Function:
    1. Define the basic class(interface) of reinforcement learning algorithm
    2. Define the general tool(helper) functions
'''


class BaseRLAlgorithm(object):
    '''
    Overview: Basic class(interface) for all the reinforcement learning algorithm
    Interface: __init__, __call__, __repr__
    '''

    def __init__(self, cfg):
        '''
        Overview: initialize hyper-parameters
        Arguments:
            - cfg (:obj:`dict`) config dict contains hyper-parameters setting
        '''
        raise NotImplementedError

    def __call__(self, inputs):
        '''
        Overview: implement the coorresponding RL algorithm for one step iteration
        Arguments:
            - inputs (:obj:`dict`) input dict contains all the necessary parts (default key: device, dtype)
        Returns:
            - ret (:obj:`dict`) output dict contains all the loss items and the necessary metrics
        '''
        raise NotImplementedError

    def __repr__(self):
        '''
        Overview: algorithm introduction and related hyper-parameters setting
        Returns:
            - s (:obj:`str`) information string
        '''
        raise NotImplementedError
