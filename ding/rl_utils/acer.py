from collections import namedtuple
import torch
import torch.nn.functional as F
EPS=1e-30
def acer_policy_error(q_values,q_retraces,v_pred,target_pi,actions,ratio,c_clip_ratio=10.0):
    """
        Overview:
            Get ACER policy loss 
        Arguments:
            - q_values (:obj:`torch.Tensor`): Q values
            - q_retraces (:obj:`torch.Tensor`): Q values (be calculated by retrace method)
            - v_pred (:obj:`torch.Tensor`): V values
            - target_pi (:obj:`torch.Tensor`): The new policy's probability
            - actions (:obj:`torch.Tensor`): The actions in replay buffer
            - ratio (:obj:`torch.Tensor`): ratio of new polcy with behavior policy
            - c_clip_ratio (:obj:`float`): clip value for ratio
        Returns:
            - actor_loss (:obj:`torch.Tensor`): policy loss from q_retrace
            - bc_loss (:obj:`torch.Tensor`): correct policy loss
        """
    actions=actions.unsqueeze(-1)
    with torch.no_grad():
        advantage_retraces = q_retraces-v_pred #shape T,B,1
        advantage_native = q_values-v_pred #shape T,B,env_action_shape
    actor_loss = ratio.gather(-1,actions).clamp(max=c_clip_ratio)*advantage_retraces*(target_pi.gather(-1,actions)+EPS).log() #shape T,B,1
    bc_loss = (1.0-c_clip_ratio/(ratio+EPS)).clamp(min=0.0)*target_pi.detach()*advantage_native*(target_pi+EPS).log() #shape T,B,env_action_shape
    bc_loss=bc_loss.sum(-1).unsqueeze(-1)
    return actor_loss,bc_loss
    

def acer_value_error(q_values,q_retraces,actions):
    """
        Overview:
            Get ACER critic loss 
        Arguments:
            - q_values (:obj:`torch.Tensor`): Q values
            - q_retraces (:obj:`torch.Tensor`): Q values (be calculated by retrace method)
            - actions (:obj:`torch.Tensor`): The actions in replay buffer
            - ratio (:obj:`torch.Tensor`): ratio of new polcy with behavior policy
        Returns:
            - critic_loss (:obj:`torch.Tensor`): critic loss 
        """
    actions=actions.unsqueeze(-1)
    critic_loss=0.5*(q_retraces-q_values.gather(-1,actions)).pow(2)
    return critic_loss

def acer_trust_region_update(actor_gradients,target_pi,avg_pi,trust_region_value):
    """
        Overview:
            calcuate gradient with trust region constrain
        Arguments:
            - actor_gradients (:obj:`list(torch.Tensor)`): gradients value's for different part
            - target_pi (:obj:`torch.Tensor`): The new policy's probability 
            - avg_pi (:obj:`torch.Tensor`): The average policy's probability
            - trust_region_value (:obj:`float`): the range of trust region
        Returns:
            - critic_loss (:obj:`torch.Tensor`): critic loss 
        """
    with torch.no_grad():
        KL_gradients = [(avg_pi/(target_pi+EPS))]
    update_gradients = []
    for actor_gradient,KL_gradient in zip(actor_gradients,KL_gradients):
        scale = actor_gradient.mul(KL_gradient).sum(-1).unsqueeze(-1)-trust_region_value
        scale = torch.div(scale,KL_gradient.mul(KL_gradient).sum(-1).unsqueeze(-1)).clamp(min=0.0)
        update_gradients.append(actor_gradient-scale*KL_gradient)
    return update_gradients

