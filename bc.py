from ding.entry.application_entry import collect_episodic_demo_data
from dizoo.smac.config.smac_MMM2_masac_config import main_config, create_config
from ding.model.template.maqac import MAQAC
from ding.policy.common_utils import default_preprocess_learn
from random import shuffle
import torch
import pickle
import torch.nn as nn
import os
from ding.torch_utils import Adam, to_device
from tensorboardX import SummaryWriter
import copy

tb_name = 'bc_tb'
tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(tb_name), 'serial'))
model = MAQAC(agent_obs_shape=204,global_obs_shape=431,action_shape=18,twin_critic=True,actor_head_hidden_size=256,critic_head_hidden_size=256,)
optimizer_q = Adam(
    model.critic.parameters(),
    lr=5e-4,
)
#model.reset()
model.train()
data_path='./MMM2_mc_expert.pkl'
data=pickle.load(open(data_path,'rb'))

epochs = 10
batch_size = 320
max_data = len(data)
j=0
for epoch in range(epochs):
    #data=shuffle(data)
    for i in range(int(max_data/batch_size)+1):
        j=j+1
        train_data = copy.deepcopy(data[i*320:min((i+1)*320,max_data)])
        train_data = default_preprocess_learn(train_data)
        target_q_value = model.forward(train_data, 'compute_critic')['q_value']
        q0=target_q_value[0]
        q1=target_q_value[1]
        act = train_data['action']
        reward = train_data['reward']
        batch_range = torch.arange(act.shape[0])
        actor_range = torch.arange(act.shape[1])
        batch_actor_range = torch.arange(act.shape[0] * act.shape[1])
        temp_q0 = q0.reshape(act.shape[0] * act.shape[1], -1)
        temp_act = act.reshape(act.shape[0] * act.shape[1])
        q_s_a0 = temp_q0[batch_actor_range, temp_act]
        q_s_a0 = q_s_a0.reshape(act.shape[0], act.shape[1])
        temp_q1 = q1.reshape(act.shape[0] * act.shape[1], -1)
        temp_act = act.reshape(act.shape[0] * act.shape[1])
        q_s_a1 = temp_q1[batch_actor_range, temp_act]
        q_s_a1 = q_s_a1.reshape(act.shape[0], act.shape[1])
        reward = reward.unsqueeze(1)
        reward = reward.expand(act.shape[0],act.shape[1])
        criterion = nn.MSELoss(reduction='none')
        loss0 = criterion(q_s_a0,reward).mean()
        loss1 = criterion(q_s_a1,reward).mean()
        loss = (loss0+loss1)/2

        optimizer_q.zero_grad()
        loss.backward()
        optimizer_q.step()

        tb_logger.add_scalar('q_loss/loss', loss, j)