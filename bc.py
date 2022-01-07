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
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import copy

tb_name = 'ma_bc_tb_256_test_use_sub_madata_eval2'
tb_logger = SummaryWriter(os.path.join('/mnt/lustre/weiyuhong/exp_bc_maac/{}/log/'.format(tb_name), 'serial'))
model = MAQAC(agent_obs_shape=204,global_obs_shape=431,action_shape=18,twin_critic=True,actor_head_hidden_size=256,critic_head_hidden_size=256,)
optimizer_q = Adam(
    model.critic.parameters(),
    lr=5e-4,
)
#scheduler = lr_scheduler.MultiStepLR(optimizer_q,milestones=[1000,2000],gamma = 0.2)

#model.reset()
model.train()
data_path='./MMM2_mc_masac_subbest_expert.pkl'
data_path2='./MMM2_mc_expert.pkl'
data=pickle.load(open(data_path,'rb'))
data2=pickle.load(open(data_path2,'rb'))
epochs = 10
batch_size = 320
max_data = len(data)
j=0



for epoch in range(epochs):
    shuffle(data)
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
        #scheduler.step()
        shuffle(data2)
        eval_dataset = copy.deepcopy(data2[0:400])
        eval_data = default_preprocess_learn(eval_dataset)

        eval_target_q_value = model.forward(eval_data, 'compute_critic')['q_value']
        eval_q0=eval_target_q_value[0]
        eval_q1=eval_target_q_value[1]
        eval_act = eval_data['action']
        eval_reward = eval_data['reward']
        eval_batch_range = torch.arange(eval_act.shape[0])
        eval_actor_range = torch.arange(eval_act.shape[1])
        eval_batch_actor_range = torch.arange(eval_act.shape[0] * eval_act.shape[1])
        eval_temp_q0 = eval_q0.reshape(eval_act.shape[0] * eval_act.shape[1], -1)
        eval_temp_act = eval_act.reshape(eval_act.shape[0] * eval_act.shape[1])
        eval_q_s_a0 = eval_temp_q0[eval_batch_actor_range, eval_temp_act]
        eval_q_s_a0 = eval_q_s_a0.reshape(eval_act.shape[0], eval_act.shape[1])
        eval_temp_q1 = eval_q1.reshape(eval_act.shape[0] * eval_act.shape[1], -1)
        eval_temp_act = eval_act.reshape(eval_act.shape[0] * eval_act.shape[1])
        eval_q_s_a1 = eval_temp_q1[eval_batch_actor_range, eval_temp_act]
        eval_q_s_a1 = eval_q_s_a1.reshape(eval_act.shape[0], eval_act.shape[1])
        eval_reward = eval_reward.unsqueeze(1)
        eval_reward = eval_reward.expand(eval_act.shape[0],eval_act.shape[1])


        loss_eval_1 = criterion(eval_q_s_a0,eval_reward).mean()
        loss_eval_2 = criterion(eval_q_s_a1,eval_reward).mean()
        loss_eval = (loss_eval_1+loss_eval_2)/2
        tb_logger.add_scalar('q_loss/loss', loss, j)
        tb_logger.add_scalar('q_loss/loss_eval', loss_eval, j)
        #tb_logger.add_scalar('q_loss/lr', optimizer_q.state_dict()['param_groups'][0]['lr'], j)