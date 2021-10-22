from ding.policy import Policy
from ding.model import model_wrap


class fake_policy(Policy):

    def _init_learn(self):
        pass

    def _forward_learn(self, data):
        pass

    def _init_eval(self):
        self._eval_model = model_wrap(self._model, 'base')

    def _forward_eval(self, data):
        self._eval_model.eval()
        output = self._eval_model.forward(data)
        return output

    def _monitor_vars_learn(self):
        return ['forward_time', 'backward_time', 'sync_time']

    def _init_collect(self):
        pass

    def _forward_collect(self, data):
        pass

    def _process_transition(self):
        pass

    def _get_train_sample(self):
        pass
