from ding.utils import POLICY_REGISTRY
from ding.rl_utils import get_epsilon_greedy_fn
from .base_policy import CommandModePolicy

from .dqn import DQNPolicy, DQNSTDIMPolicy
from .c51 import C51Policy
from .qrdqn import QRDQNPolicy
from .iqn import IQNPolicy
from .fqf import FQFPolicy
from .rainbow import RainbowDQNPolicy
from .r2d2 import R2D2Policy
from .r2d2_gtrxl import R2D2GTrXLPolicy
from .r2d2_collect_traj import R2D2CollectTrajPolicy
from .sqn import SQNPolicy
from .ppo import PPOPolicy, PPOOffPolicy, PPOPGPolicy, PPOSTDIMPolicy
from .offppo_collect_traj import OffPPOCollectTrajPolicy
from .ppg import PPGPolicy, PPGOffPolicy
from .a2c import A2CPolicy
from .impala import IMPALAPolicy
from .ngu import NGUPolicy
from .ddpg import DDPGPolicy
from .td3 import TD3Policy
from .td3_vae import TD3VAEPolicy
from .td3_bc import TD3BCPolicy
from .sac import SACPolicy, SACDiscretePolicy
from .mbpolicy.mbsac import MBSACPolicy, STEVESACPolicy
from .qmix import QMIXPolicy
from .wqmix import WQMIXPolicy
from .collaq import CollaQPolicy
from .coma import COMAPolicy
from .atoc import ATOCPolicy
from .acer import ACERPolicy
from .qtran import QTRANPolicy
from .sql import SQLPolicy

from .dqfd import DQFDPolicy
from .r2d3 import R2D3Policy

from .d4pg import D4PGPolicy
from .cql import CQLPolicy, CQLDiscretePolicy
from .decision_transformer import DTPolicy
from .pdqn import PDQNPolicy
from .sac import SQILSACPolicy


class EpsCommandModePolicy(CommandModePolicy):

    def _init_command(self) -> None:
        r"""
        Overview:
            Command mode init method. Called by ``self.__init__``.
            Set the eps_greedy rule according to the config for command
        """
        eps_cfg = self._cfg.other.eps
        self.epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

    def _get_setting_collect(self, command_info: dict) -> dict:
        r"""
        Overview:
            Collect mode setting information including eps
        Arguments:
            - command_info (:obj:`dict`): Dict type, including at least ['learner_train_iter', 'collector_envstep']
        Returns:
           - collect_setting (:obj:`dict`): Including eps in collect mode.
        """
        # Decay according to `learner_train_iter`
        # step = command_info['learner_train_iter']
        # Decay according to `envstep`
        step = command_info['envstep']
        return {'eps': self.epsilon_greedy(step)}

    def _get_setting_learn(self, command_info: dict) -> dict:
        return {}

    def _get_setting_eval(self, command_info: dict) -> dict:
        return {}


class DummyCommandModePolicy(CommandModePolicy):

    def _init_command(self) -> None:
        pass

    def _get_setting_collect(self, command_info: dict) -> dict:
        return {}

    def _get_setting_learn(self, command_info: dict) -> dict:
        return {}

    def _get_setting_eval(self, command_info: dict) -> dict:
        return {}


@POLICY_REGISTRY.register('dqn_command')
class DQNCommandModePolicy(DQNPolicy, EpsCommandModePolicy):
    pass


@POLICY_REGISTRY.register('dqn_stdim_command')
class DQNSTDIMCommandModePolicy(DQNSTDIMPolicy, EpsCommandModePolicy):
    pass


@POLICY_REGISTRY.register('dqfd_command')
class DQFDCommandModePolicy(DQFDPolicy, EpsCommandModePolicy):
    pass


@POLICY_REGISTRY.register('c51_command')
class C51CommandModePolicy(C51Policy, EpsCommandModePolicy):
    pass


@POLICY_REGISTRY.register('qrdqn_command')
class QRDQNCommandModePolicy(QRDQNPolicy, EpsCommandModePolicy):
    pass


@POLICY_REGISTRY.register('iqn_command')
class IQNCommandModePolicy(IQNPolicy, EpsCommandModePolicy):
    pass


@POLICY_REGISTRY.register('fqf_command')
class FQFCommandModePolicy(FQFPolicy, EpsCommandModePolicy):
    pass


@POLICY_REGISTRY.register('rainbow_command')
class RainbowDQNCommandModePolicy(RainbowDQNPolicy, EpsCommandModePolicy):
    pass


@POLICY_REGISTRY.register('r2d2_command')
class R2D2CommandModePolicy(R2D2Policy, EpsCommandModePolicy):
    pass


@POLICY_REGISTRY.register('r2d2_gtrxl_command')
class R2D2GTrXLCommandModePolicy(R2D2GTrXLPolicy, EpsCommandModePolicy):
    pass


@POLICY_REGISTRY.register('r2d2_collect_traj_command')
class R2D2CollectTrajCommandModePolicy(R2D2CollectTrajPolicy, DummyCommandModePolicy):
    pass


@POLICY_REGISTRY.register('r2d3_command')
class R2D3CommandModePolicy(R2D3Policy, EpsCommandModePolicy):
    pass


@POLICY_REGISTRY.register('sqn_command')
class SQNCommandModePolicy(SQNPolicy, DummyCommandModePolicy):
    pass


@POLICY_REGISTRY.register('sql_command')
class SQLCommandModePolicy(SQLPolicy, EpsCommandModePolicy):
    pass


@POLICY_REGISTRY.register('ppo_command')
class PPOCommandModePolicy(PPOPolicy, DummyCommandModePolicy):
    pass


@POLICY_REGISTRY.register('ppo_stdim_command')
class PPOSTDIMCommandModePolicy(PPOSTDIMPolicy, DummyCommandModePolicy):
    pass


@POLICY_REGISTRY.register('ppo_pg_command')
class PPOPGCommandModePolicy(PPOPGPolicy, DummyCommandModePolicy):
    pass


@POLICY_REGISTRY.register('ppo_offpolicy_command')
class PPOOffCommandModePolicy(PPOOffPolicy, DummyCommandModePolicy):
    pass


@POLICY_REGISTRY.register('offppo_collect_traj_command')
class PPOOffCollectTrajCommandModePolicy(OffPPOCollectTrajPolicy, DummyCommandModePolicy):
    pass


@POLICY_REGISTRY.register('a2c_command')
class A2CCommandModePolicy(A2CPolicy, DummyCommandModePolicy):
    pass


@POLICY_REGISTRY.register('impala_command')
class IMPALACommandModePolicy(IMPALAPolicy, DummyCommandModePolicy):
    pass


@POLICY_REGISTRY.register('ppg_offpolicy_command')
class PPGOffCommandModePolicy(PPGOffPolicy, DummyCommandModePolicy):
    pass


@POLICY_REGISTRY.register('ppg_command')
class PPGCommandModePolicy(PPGPolicy, DummyCommandModePolicy):
    pass


@POLICY_REGISTRY.register('ddpg_command')
class DDPGCommandModePolicy(DDPGPolicy, CommandModePolicy):

    def _init_command(self) -> None:
        r"""
        Overview:
            Command mode init method. Called by ``self.__init__``.
            If hybrid action space, set the eps_greedy rule according to the config for command,
            otherwise, just a empty method
        """
        if self._cfg.action_space == 'hybrid':
            eps_cfg = self._cfg.other.eps
            self.epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

    def _get_setting_collect(self, command_info: dict) -> dict:
        r"""
        Overview:
            Collect mode setting information including eps when hybrid action space
        Arguments:
            - command_info (:obj:`dict`): Dict type, including at least ['learner_step', 'envstep']
        Returns:
           - collect_setting (:obj:`dict`): Including eps in collect mode.
        """
        if self._cfg.action_space == 'hybrid':
            # Decay according to `learner_step`
            # step = command_info['learner_step']
            # Decay according to `envstep`
            step = command_info['envstep']
            return {'eps': self.epsilon_greedy(step)}
        else:
            return {}

    def _get_setting_learn(self, command_info: dict) -> dict:
        return {}

    def _get_setting_eval(self, command_info: dict) -> dict:
        return {}


@POLICY_REGISTRY.register('td3_command')
class TD3CommandModePolicy(TD3Policy, DummyCommandModePolicy):
    pass


@POLICY_REGISTRY.register('td3_vae_command')
class TD3VAECommandModePolicy(TD3VAEPolicy, DummyCommandModePolicy):
    pass


@POLICY_REGISTRY.register('td3_bc_command')
class TD3BCCommandModePolicy(TD3BCPolicy, DummyCommandModePolicy):
    pass


@POLICY_REGISTRY.register('sac_command')
class SACCommandModePolicy(SACPolicy, DummyCommandModePolicy):
    pass


@POLICY_REGISTRY.register('mbsac_command')
class MBSACCommandModePolicy(MBSACPolicy, DummyCommandModePolicy):
    pass


@POLICY_REGISTRY.register('stevesac_command')
class STEVESACCommandModePolicy(STEVESACPolicy, DummyCommandModePolicy):
    pass


@POLICY_REGISTRY.register('cql_command')
class CQLCommandModePolicy(CQLPolicy, DummyCommandModePolicy):
    pass


@POLICY_REGISTRY.register('cql_discrete_command')
class CQLDiscreteCommandModePolicy(CQLDiscretePolicy, EpsCommandModePolicy):
    pass


@POLICY_REGISTRY.register('dt_command')
class DTCommandModePolicy(DTPolicy, DummyCommandModePolicy):
    pass


@POLICY_REGISTRY.register('qmix_command')
class QMIXCommandModePolicy(QMIXPolicy, EpsCommandModePolicy):
    pass


@POLICY_REGISTRY.register('wqmix_command')
class WQMIXCommandModePolicy(WQMIXPolicy, EpsCommandModePolicy):
    pass


@POLICY_REGISTRY.register('collaq_command')
class CollaQCommandModePolicy(CollaQPolicy, EpsCommandModePolicy):
    pass


@POLICY_REGISTRY.register('coma_command')
class COMACommandModePolicy(COMAPolicy, EpsCommandModePolicy):
    pass


@POLICY_REGISTRY.register('atoc_command')
class ATOCCommandModePolicy(ATOCPolicy, DummyCommandModePolicy):
    pass


@POLICY_REGISTRY.register('acer_command')
class ACERCommandModePolisy(ACERPolicy, DummyCommandModePolicy):
    pass


@POLICY_REGISTRY.register('qtran_command')
class QTRANCommandModePolicy(QTRANPolicy, EpsCommandModePolicy):
    pass


@POLICY_REGISTRY.register('ngu_command')
class NGUCommandModePolicy(NGUPolicy, EpsCommandModePolicy):
    pass


@POLICY_REGISTRY.register('d4pg_command')
class D4PGCommandModePolicy(D4PGPolicy, DummyCommandModePolicy):
    pass


@POLICY_REGISTRY.register('pdqn_command')
class PDQNCommandModePolicy(PDQNPolicy, EpsCommandModePolicy):
    pass


@POLICY_REGISTRY.register('sac_discrete_command')
class SACDiscreteCommandModePolicy(SACDiscretePolicy, EpsCommandModePolicy):
    pass


@POLICY_REGISTRY.register('sqil_sac_command')
class SQILSACCommandModePolicy(SQILSACPolicy, DummyCommandModePolicy):
    pass
