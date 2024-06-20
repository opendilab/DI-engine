"""
This is the implementation of elastic decision transformer 

Reference: https://github.com/kristery/Elastic-DT/blob/master/decision_transformer/model.py
"""
import math
from typing import Union, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ding.utils import SequenceType

class MaskedCausalAttention(nn.Module):
    def __init__(
        self,
        h_dim: int,
        max_T: int,
        n_heads: int,
        drop_p: float,
        mgdt: bool = False,
        dt_mask: bool = False,
        att_mask: Optional[torch.Tensor] = None,
        num_inputs: int = 4,
        real_rtg: bool = False # currently not used to change the attention mask since it will make sampling more complicated
    ) -> None:
        """
        Overview:
            The implementation of masked causal attention in decision transformer.
            
        Arguments:
            - h_dim (:obj:`int`): The dimension of the hidden layers, such as 128.
            - max_T (:obj:`int`): The max context length of the attention, such as 6.
            - n_heads (:obj:`int`): The number of heads in calculating attention, such as 8.
            - drop_p (:obj:`float`): The drop rate of the drop-out layer, such as 0.1.
            - mgdt (:obj:`bool`): If use multi-game decision transformer.
            - dt_mask (:obj:`bool`): If use decision transformer mask.
            - att_mask (:obj:`Optional[torch.Tensor]`): Define attention mask manually of default.
            - num_inputs (:obj:`int`): The number of inputs when mgdt mode is used.
            - real_rth (:obj:`bool`): 
        """
        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T
        self.num_inputs=num_inputs
        self.real_rtg=real_rtg

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        if att_mask is not None:
            mask = att_mask
        else:
            ones = torch.ones((max_T, max_T))
            mask = torch.tril(ones).view(1, 1, max_T, max_T)
            if (mgdt and not dt_mask):
                # need to mask the return except for the first return entry
                # this is the default practice used by their notebook
                # for every inference, we first estimate the return value for the first return
                # then we estimate the action for at timestamp t
                # it is actually not mentioned in the paper. (ref: ret_sample_fn, single_return_token)
                # mask other ret entries (s, R, a, s, R, a)
                period = num_inputs
                ret_order = 2
                ret_masked_rows = torch.arange(period + ret_order-1, max_T, period).long()
                # print(ret_masked_rows)
                # print(max_T, ret_masked_rows, mask.shape)
                mask[:, :, :, ret_masked_rows] = 0

        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer("mask", mask)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            MaskedCausalAttention forward computation graph, input a sequence tensor \
            and return a tensor with the same shape.
        
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor.
        
        Returns:
            - out (:obj:`torch.Tensor`): Output tensor, the shape is the same as the input.
        
        Examples:
            >>> inputs = torch.randn(2, 4, 64)
            >>> model = MaskedCausalAttention(64, 5, 4, 0.1)
            >>> outputs = model(inputs)
            >>> assert outputs.shape == torch.Size([2, 4, 64])
        """
        B, T, C = x.shape  # batch size, seq length, h_dim * n_heads

        N, D = self.n_heads, C // self.n_heads,
        # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1, 2)
        k = self.k_net(x).view(B, T, N, D).transpose(1, 2)
        v = self.v_net(x).view(B, T, N, D).transpose(1, 2)

        # weights (B, N, T, T)
        weights = q @ k.transpose(2, 3) / math.sqrt(D)
        # causal mask applied to weights
        #print(f"shape of weights: {weights.shape}, shape of mask: {self.mask.shape}, T: {T}")
        weights = weights.masked_fill(
            self.mask[..., :T, :T] == 0, float("-inf")
        )
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B, T, N * D)

        out = self.proj_drop(self.proj_net(attention))
        return out
    
class Block(nn.Module):
    def __init__(self,
        h_dim: int,
        max_T: int,
        n_heads: int,
        drop_p: float,
        mgdt: bool=False,
        dt_mask: bool=False,
        att_mask: Optional[torch.Tensor]=None,
        num_inputs: int=4,
        real_rtg: bool=False
    ) -> None:
        """
        Overview:
            The decision transformer block based on MaskedCasualAttention.
        
        Arguments:
            - h_dim (:obj:`int`): The dimension of the hidden layers, such as 128.
            - max_T (:obj:`int`): The max context length of the attention, such as 6.
            - n_heads (:obj:`int`): The number of heads in calculating attention, such as 8.
            - drop_p (:obj:`float`): The drop rate of the drop-out layer, such as 0.1.
            - mgdt (:obj:`bool`): If use multi-game decision transformer.
            - dt_mask (:obj:`bool`): If use decision transformer mask.
            - att_mask (:obj:`Optional[torch.Tensor]`): Define attention mask manually of default.
            - num_inputs (:obj:`int`): The number of inputs when mgdt mode is used.
            - real_rth (:obj:`bool`): 
        """
        super().__init__()
        self.num_inputs = num_inputs
        self.attention = MaskedCausalAttention(
            h_dim,
            max_T,
            n_heads,
            drop_p,
            mgdt=mgdt,
            dt_mask=dt_mask,
            att_mask=att_mask,
            num_inputs=num_inputs,
            real_rtg=real_rtg
        )
        self.mlp = nn.Sequential(
            nn.Linear(h_dim, 4 * h_dim),
            nn.GELU(),
            nn.Linear(4 * h_dim, h_dim),
            nn.Dropout(drop_p),
        )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Forward computation graph of the decision transformer block, input a sequence tensor 
            and return a tensor with the same shape.
        
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor.
        
        Returns:
            - output (:obj:`torch.Tensor`): Output tensor, the shape is the same as the input.
        
        Examples:
            >>> inputs = torch.randn(2, 4, 64)
            >>> model = Block(64, 5, 4, 0.1)
            >>> outputs = model(inputs)
            >>> outputs.shape == torch.Size([2, 4, 64])
        """
        # Attention -> LayerNorm -> MLP -> LayerNorm
        # print(f"shape of x: {x.shape}, shape of attention: {self.attention(x).shape}")
        x = x + self.attention(x)  # residual
        x = self.ln1(x)
        x = x + self.mlp(x)  # residual
        x = self.ln2(x)
        return x
    

class DecisionTransformer(nn.Module):
    """
    Overview:
        The implementation of decision transformer.
    Interfaces:
        ``__init__``, ``forward``, ``configure_optimizers``
    """
    
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        n_blocks: int,
        h_dim: int,
        context_len: int,
        n_heads: int,
        drop_p: float,
        env_name: str,
        max_timestep: int = 4096,
        num_bin: int = 120,
        dt_mask: bool = False,
        rtg_scale: int =1000,
    ) -> None:
        """
        Overview:
            Initialize the DecisionTransformer Model according to input arguments.
        
        Arguments:
            - state_dim (:obj:`int`): Dimension of state, such as 17.
            - act_dim (:obj:`int`): The dimension of actions, such as 6.
            - n_blocks (:obj:`int`): The number of transformer blocks in the decision transformer, such as 3.
            - h_dim (:obj:`int`): The dimension of the hidden layers, such as 128.
            - context_len (:obj:`int`): The max context length of the attention, such as 6.
            - n_heads (:obj:`int`): The number of heads in calculating attention, such as 8.
            - drop_p (:obj:`float`): The drop rate of the drop-out layer, such as 0.1.
            - env_name (:obj:`str`): The name of environment.
            - max_timestep (:obj:`int`): The max length of the total sequence, defaults to be 4096.
            - num_bin (:obj:`int`): Number of return output bins, such as 60.
            - dt_mask (:obj:`bool`): Whether use mask in the blocks of Decision Transformer.
            - rtg_scale (:obj:`int`): The scale factor for normalizing the return-to-go values during training.
        """
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim
        self.num_bin = num_bin
        # for return scaling
        self.env_name = env_name
        self.rtg_scale = rtg_scale

        ### transformer blocks
        input_seq_len = 4 * context_len
        blocks = [Block(
                h_dim,
                input_seq_len,
                n_heads,
                drop_p,
                mgdt=True,
                dt_mask=dt_mask,
            ) for _ in range(n_blocks)
        ]
        self.transformer = nn.Sequential(*blocks)

        ### projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = torch.nn.Linear(1, h_dim)
        self.embed_state = torch.nn.Linear(state_dim, h_dim)
        self.embed_reward = torch.nn.Linear(1, h_dim)

        # continuous actions
        self.embed_action = torch.nn.Linear(act_dim, h_dim)
        use_action_tanh = True  # True for continuous actions
        
        
        ### prediction heads
        self.predict_rtg = torch.nn.Linear(h_dim, int(num_bin))
        self.predict_state = torch.nn.Linear(h_dim, state_dim)
        self.predict_action = nn.Sequential(
            *(
                [nn.Linear(h_dim, act_dim)]
                + ([nn.Tanh()] if use_action_tanh else [])
            )
        )
        self.predict_reward = torch.nn.Linear(h_dim, 1)

    def forward(
            self, 
            timesteps: torch.Tensor, 
            states: torch.Tensor, 
            actions: torch.Tensor, 
            returns_to_go: torch.Tensor, 
            rewards: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        B, T, _ = states.shape

        returns_to_go = returns_to_go.float()
        # returns_to_go = (
        #     encode_return(
        #         self.env_name, returns_to_go, num_bin=self.num_bin, rtg_scale=self.rtg_scale
        #     )
        #     - self.num_bin / 2
        # ) / (self.num_bin / 2)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_rtg(returns_to_go) + time_embeddings
        rewards_embeddings = self.embed_reward(rewards) + time_embeddings
        

        # stack rtg, states and actions and reshape sequence as
        # (r_0, s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2 ...)
        h = (
            torch.stack(
                (
                    state_embeddings,
                    returns_embeddings,
                    action_embeddings,
                    rewards_embeddings,
                ),
                dim=1,
            )
            .permute(0, 2, 1, 3)
            .reshape(B, 4 * T, self.h_dim)
        )

        h = self.embed_ln(h)

        # transformer and prediction
        h = self.transformer(h)

        h = h.reshape(B, T, 4, self.h_dim).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_rtg(h[:, 0])  # predict next rtg given s
        state_preds = self.predict_state(
            h[:, 3]
        )  # predict next state given s, R, a, r
        action_preds = self.predict_action(
            h[:, 1]
        )  # predict action given s, R
        reward_preds = self.predict_reward(
            h[:, 2]
        )  # predict reward given s, R, a

        return state_preds, action_preds, return_preds, reward_preds


# a version that does not use reward at all
class ElasticDecisionTransformer(DecisionTransformer):
    """
    Overview:
        The implementation of elsatic decision transformer.
    Interfaces:
        ``__init__``, ``forward``
    """
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        n_blocks: int,
        h_dim: int,
        context_len: int,
        n_heads: int,
        drop_p: float,
        env_name: str,
        max_timestep: int = 4096,
        num_bin: int = 120,
        dt_mask: bool = False,
        rtg_scale: int = 1000,
        num_inputs: int = 3,
        real_rtg: bool = False,
        is_continuous: bool = True, # True for continuous action
    ) -> None:
        """
        Overview:
            Initialize the Elastic Decision Transformer Model. The definition of Elastic Decision Transformer \
            is defined based on Decision Transformer.
        
        Arguments:
            - state_dim (:obj:`int`): Dimension of state, such as 17.
            - act_dim (:obj:`int`): The dimension of actions, such as 6.
            - n_blocks (:obj:`int`): The number of transformer blocks in the decision transformer, such as 3.
            - h_dim (:obj:`int`): The dimension of the hidden layers, such as 128.
            - context_len (:obj:`int`): The max context length of the attention, such as 6.
            - n_heads (:obj:`int`): The number of heads in calculating attention, such as 8.
            - drop_p (:obj:`float`): The drop rate of the drop-out layer, such as 0.1.
            - max_timestep (:obj:`int`): The max length of the total sequence, defaults to be 4096.
            - num_bin (:obj:`int`): Number of return output bins, such as 60.
            - dt_mask (:obj:`bool`): Whether use mask in the blocks of Decision Transformer.
            - rtg_scale (:obj:`int`): The scale factor for normalizing the return-to-go values during training.
            - num_inputs (:obj:`int`): The input arguments of EDT. 3 for state, return, action while 4 for state, return, action, reward. 
            - real_rtg (:obj:`bool`): Realized return-to-go, which represents the actual cumulative return from the current state to the end of the episode.
            - is_continuous (:obj:`bool`): True for continuous action, while False for discrete action.
        """
        super().__init__(state_dim, act_dim, n_blocks, h_dim, context_len, n_heads, drop_p, \
            env_name, max_timestep=max_timestep, num_bin=num_bin, dt_mask=dt_mask, rtg_scale=rtg_scale,
        )
        # return, state, action
        self.num_inputs = num_inputs
        self.is_continuous = is_continuous
        input_seq_len = num_inputs * context_len
        blocks = [
            Block(
                h_dim,
                input_seq_len,
                n_heads,
                drop_p,
                mgdt=True,
                dt_mask=dt_mask,
                num_inputs=num_inputs,
                real_rtg=real_rtg,
            )
            for _ in range(n_blocks)
        ]
        self.transformer = nn.Sequential(*blocks)

        ### projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = torch.nn.Linear(1, h_dim)
        self.embed_state = torch.nn.Linear(state_dim, h_dim)
        self.embed_reward = torch.nn.Linear(1, h_dim)
        # # discrete actions
        if not self.is_continuous:
            self.embed_action = torch.nn.Embedding(18, h_dim)
        else:
            self.embed_action = torch.nn.Linear(act_dim, h_dim)

        ### prediction heads
        self.predict_rtg = torch.nn.Linear(h_dim, int(num_bin))
        self.predict_rtg2 = torch.nn.Linear(h_dim, 1)
        self.predict_state = torch.nn.Linear(h_dim + act_dim, state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(h_dim, act_dim)] + ([nn.Tanh()] if is_continuous else []))
        )
        self.predict_reward = torch.nn.Linear(h_dim, 1)

    def forward(
            self, 
            timesteps: torch.Tensor, 
            states: torch.Tensor, 
            actions: torch.Tensor, 
            returns_to_go: torch.Tensor, 
            *args, 
            **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Overview:
            Forward computation graph of the decision transformer, input a sequence tensor \
            and return a tensor with the same shape. Suppose B is batch size and T is context length.
        
        Arguments:
            - timesteps (:obj:`torch.Tensor`): The timestep for input sequence with shape (B, T).
            - states (:obj:`torch.Tensor`): The sequence of states with shape (B, T, S) where S is state size.
            - actions (:obj:`torch.Tensor`): The sequence of actions with shape (B, T, A) where A is action size.
            - returns_to_go (:obj:`torch.Tensor`): The sequence of return-to-go with shape (B, T, 1).
            - rewards (:obj:`Optional[torch.Tensor]`): The sequence of rewards obtained at each timestep with shape (B, T, 1). \
            If provided and `num_inputs` is 4, it will be used in the computation.
        
        Returns: 
            - output (:obj:`Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]`): Output contains 5 tensors, \
            they are correspondingly `state_preds`, `action_preds`, `return_preds`, `return_preds2`, `reward_preds`.
        
        Examples:
            >>> B, T, S, A, H, N = 5, 23, 17, 7, 64, 121
            >>> # B: batch_size  
            >>> # T: length  
            >>> # S: state_dim  
            >>> # A: action_dim  
            >>> # H: hidden_din  
            >>> # N: num_bin
            >>> model = ElasticDecisionTransformer(
            ...         state_dim=S,
            ...         act_dim=A,
            ...         h_dim=H,
            ...         context_len=T,
            ...         num_bin=N,
            ...         n_blocks=5,
            ...         n_heads=8, # H must be divisible by n_heads
            ...         drop_p=0.1,
            ...         env_name="example_env",
            ... )
            >>> timesteps = torch.randint(0, 4096, (B, T))
            >>> states = torch.randn(B, T, S)
            >>> actions = torch.randn(B, T, A)
            >>> returns_to_go = torch.randn(B, T, 1)
            >>> rewards = torch.randn(B, T, 1)
            >>> state_preds, action_preds, return_preds, return_preds2, reward_preds = model(
            ...    timesteps, states, actions, returns_to_go
            ... )
            >>> assert state_preds.shape == torch.Size([B, T, S])
            >>> assert action_preds.shape == torch.Size([B, T, A])
            >>> assert return_preds.shape == torch.Size([B, T, N])
            >>> assert return_preds2.shape == torch.Size([B, T, 1])
            >>> assert reward_preds.shape == torch.Size([B, T, 1])

        """
        B, T, _ = states.shape
        returns_to_go = returns_to_go.float()
        # returns_to_go = (
        #     encode_return(
        #         self.env_name, returns_to_go, num_bin=self.num_bin, rtg_scale=self.rtg_scale
        #     )
        #     - self.num_bin / 2
        # ) / (self.num_bin / 2)
        rewards = kwargs.get("rewards", None)
        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_rtg(returns_to_go) + time_embeddings
        if rewards is not None and self.num_inputs == 4:
            rewards_embeddings = self.embed_reward(rewards) + time_embeddings
        assert self.num_inputs == 3 or 4

        # stack rtg, states and actions and reshape sequence as
        # (r_0, s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2 ...)
        if self.num_inputs == 3:
            h = (
                torch.stack((state_embeddings, returns_embeddings, action_embeddings), dim=1,)
                .permute(0, 2, 1, 3).reshape(B, self.num_inputs * T, self.h_dim)
            )
        elif self.num_inputs == 4:
            h = (
                torch.stack((state_embeddings, returns_embeddings, action_embeddings, rewards_embeddings), dim=1,)
                .permute(0, 2, 1, 3).reshape(B, self.num_inputs * T, self.h_dim)
            )
        h = self.embed_ln(h)

        # transformer and prediction
        h = self.transformer(h)
        h = h.reshape(B, T, self.num_inputs, self.h_dim).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_rtg(h[:, 0])  # predict next rtg given s
        return_preds2 = self.predict_rtg2(h[:, 0])  # predict next rtg with implicit loss
        action_preds = self.predict_action(h[:, 1])  # predict action given s, R
        state_preds = self.predict_state(torch.cat((h[:, 1], action_preds), 2))
        reward_preds = self.predict_reward(h[:, 2])  # predict reward given s, R, a

        return (
            state_preds,
            action_preds,
            return_preds,
            return_preds2,
            reward_preds,
        )
        
    