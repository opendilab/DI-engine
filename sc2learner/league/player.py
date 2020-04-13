import numpy as np
from collections import namedtuple


class Player:
    """
    Overview: basic player class, player is the basic member of a league
    Property: race, payoff, checkpoint_path, player_id
    """
    _name = "BasePlayer"

    def __init__(self, race, init_payoff, checkpoint_path, player_id):
        self._race = race
        self._payoff = init_payoff
        self._checkpoint_path = checkpoint_path
        self._player_id = player_id

    @property
    def race(self):
        return self._race

    @property
    def payoff(self):
        """
        Overview: return the current player payoff
        """
        return self._payoff

    @property
    def checkpoint_path(self):
        return self._checkpoint_path

    @property
    def player_id(self):
        return self._player_id


class ActivePlayer(Player):
    """
    Overview: active player class, active player can be updated
    Interface: is_trained_enough, snapshot, mutate, get_match, update_agent_step
    """
    _name = "ActivePlayer"
    BRANCH = namedtuple("BRANCH", ['name', 'prob'])

    def __init__(self, *args, branch_probs):
        """
        Overview: initialize player metadata
        Arguments:
            - branch_probs: (:obj:`list`): a list contains the probabilities of selecting the different opponent branch
        """
        super(ActivePlayer, self).__init__(*args)
        assert isinstance(branch_probs, dict)
        self._branch_probs = [self.BRANCH(k, v) for k, v in branch_probs]
        self._total_agent_steps = 0

    def is_trained_enough(self):
        """
        Overview: return whether this player is trained enough for further operation
        Returns:
            - ret (:obj:`bool`): whether this player is trained enough
        """
        raise NotImplementedError

    def snapshot(self):
        """
        Overview: generate a snapshot of the current player
        Returns:
            - snapshot_player (:obj:`HistoricalPlayer`): snapshot player
        Note:
            this method only generates a player object without saving the checkpoint, which should be completed by the interaction
            between coordinator and learner
        """
        path = self.checkpoint_path.split('.pth')[0] + '_{}'.format(self._total_agent_steps) + '.pth'
        return HistoricalPlayer(
            self.race, self.payoff, path, self.player_id, self.player_id + '-{}'.format(self._total_agent_steps)
        )

    def mutate(self, info):
        """
        Overview: mutate the current player
        Arguments:
            - info (:obj:`dict`): related information for the mutation
        Returns:
            - mutation_result (:obj:`str or None`): if the player does the mutation operation then returns the
                cooresponding model path, otherwise, returns None
        """
        raise NotImplementedError

    def get_match(self):
        """
        Overview: get an opponent to do a match
        Returns:
            - opponent (:obj:`Player`): match opponent
        """
        p = np.random.uniform()
        L = len(self._branch_probs)
        cum_p = [0.] + [sum([j.prob for j in self._branch_probs[:i + 1]]) for i in range(L)]
        idx = [cum_p[i] <= p and p < cum_p[i + 1] for i in range(L)].index(True)
        branch_name = self._name2branch(self._branch_probs[idx].name)
        return getattr(self, branch_name)()

    def _name2branch(self, s):
        return '_' + s + '_branch'

    def update_agent_step(self, step):
        """
        Overview: update agent step
        Arguments:
            - step (:obj:`int`): current agent step
        """
        self._total_agent_steps = step


class HistoricalPlayer(Player):
    """
    Overview: historical player with fixed checkpoint
    Property: parent_id
    """
    _name = "HistoricalPlayer"

    def __init__(self, *args, parent_id):
        super(HistoricalPlayer, self).__init__(*args)
        self._parent_id = parent_id

    @property
    def parent_id(self):
        return self._parent_id


class PFSPPlayer(ActivePlayer):
    _name = "PFSPPlayer"

    def _pfsp_branch(self):
        pass

    def _sp_branch(self):
        pass

    def _verification_branch(self):
        pass


class MainPlayer(PFSPPlayer):
    _name = "MainPlayer"

    def __init__(self, *args):
        super(MainPlayer, self).__init__(*args)

    # override
    def is_trained_enough(self):
        pass

    # override
    def mutate(self):
        """
        Overview: MainPlayer does no mutation
        """
        return None


class MainExploiter(PFSPPlayer):
    _name = "MainExploiter"


class LeagueExploiter(PFSPPlayer):
    _name = "LeagueExploiter"
