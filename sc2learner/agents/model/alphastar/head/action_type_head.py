'''
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. Implementation for action_type_head, including basic processes.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from sc2learner.nn_utils import build_activation, ResFCBlock, fc_block, one_hot
from sc2learner.rl_utils import CategoricalPdPytorch


class ActionTypeHead(nn.Module):
    '''
        Overview: The action type head uses lstm_output and scalar_context to get 
                  action_type_logits, action_type and its autoregressive_embedding.
        Interface: __init__, forward
    '''
    def __init__(self, cfg):
        '''
            Overview: initialize architect.
            Arguments:
                - cfg (:obj:`dict`): head architecture definition
        '''
        super(ActionTypeHead, self).__init__()
        self.act = build_activation(cfg.activation)  # use relu as default
        self.project = fc_block(cfg.input_dim, cfg.res_dim, activation=self.act, norm_type=cfg.norm_type)
        blocks = [ResFCBlock(cfg.res_dim, cfg.res_dim, self.act, cfg.norm_type) for _ in range(cfg.res_num)]
        self.res = nn.Sequential(*blocks)
        self.action_fc = fc_block(cfg.res_dim, cfg.action_num, activation=None, norm_type=None)

        self.action_map_fc = fc_block(cfg.action_num, cfg.action_map_dim, activation=self.act, norm_type=None)
        self.pd = CategoricalPdPytorch
        self.glu1 = build_activation('glu')(cfg.action_map_dim, cfg.gate_dim, cfg.context_dim)
        self.glu2 = build_activation('glu')(cfg.input_dim, cfg.gate_dim, cfg.context_dim)
        self.action_num = cfg.action_num

    def forward(self, lstm_output, scalar_context, temperature=1.0, action=None):
        '''
            Overview: This head embeds lstm_output into a 1D tensor of size 256, passes it through 16 ResBlocks 
                      with layer normalization each of size 256, and applies a ReLU. The output is converted to
                      a tensor with one logit for each possible action type through a GLU gated by scalar_context. 
                      action_type is sampled from these logits using a multinomial with temperature 0.8. Note that 
                      during supervised learning, action_type will be the ground truth human action type, and 
                      temperature is 1.0 (and similarly for all other arguments).
                      autoregressive_embedding is then generated by first applying a ReLU and linear layer of 
                      size 256 to the one-hot version of action_type, and projecting it to a 1D tensor of size 1024
                      through a GLU gated by scalar_context. That projection is added to another projection of 
                      lstm_output into a 1D tensor of size 1024 gated by scalar_context to yield autoregressive_embedding.
            Arguments:
                - lstm_output (:obj:`tensor`): The output of the LSTM
                - scalar_context (:obj:`tensor`): A 1D tensor of certain scalar features, include available_actions, 
                                                  cumulative_statistics, beginning_build_order
                - temperature (:obj:`float`): 
                - action (:obj:`str`): 
            Returns:
                - (:obj`tensor`): action_type_logits corresponding to the probabilities of taking each action
                - (:obj`tensor`): action_type sampled from the action_type_logits
                - (:obj`tensor`): autoregressive_embedding that combines information from lstm_output
                                  and all previous sampled arguments.
        '''
        x = self.project(lstm_output)  # embeds lstm_output into a 1D tensor of size of res_dim, use 256 as default
        x = self.res(x)  # passes x through 16 ResBlocks with layer normalization and ReLU
        x = self.action_fc(x)  # fc for action type without normalization
        if action is None:
            handle = self.pd(x.div(temperature))
            # action_type is sampled from these logits using a multinomial with temperature 0.8. 
            # Note that during supervised learning, action_type will be the ground truth human action type, 
            # and temperature is 1.0 (and similarly for all other arguments).
            action = handle.sample()

        # to get autoregressive_embedding
        action_one_hot = one_hot(action, self.action_num)  # one-hot version of action_type
        embedding1 = self.action_map_fc(action_one_hot)  
        embedding1 = self.glu1(embedding1, scalar_context)
        embedding2 = self.glu2(lstm_output, scalar_context)
        embedding = embedding1 + embedding2

        return x, action, embedding


def test_action_type_head():
    class CFG:
        def __init__(self):
            self.input_dim = 384
            self.res_dim = 256
            self.res_num = 16
            self.action_num = 314
            self.action_map_dim = 256
            self.gate_dim = 1024
            self.context_dim = 120
            self.activation = 'relu'
            self.norm_type = 'LN'

    model = ActionTypeHead(CFG()).cuda()
    lstm_output = torch.randn(4, 384).cuda()
    scalar_context = torch.randn(4, 120).cuda()
    logits, action, embedding = model(lstm_output, scalar_context)
    print(model)
    print(logits.shape)
    print(action)
    print(embedding.shape)


if __name__ == "__main__":
    test_action_type_head()
