import gym

from ding.utils import ENV_REGISTRY

from .utils import *
from ding.envs import BaseEnv
import openai


@ENV_REGISTRY.register('tabmwp')
class TabMWP(BaseEnv):
    def __init__(self, cfg):
        # args contains: cand_number, train_number, engine, temperature,
        # max_tokens, top_p, frequency_penalty, presence_penalty, api_key
        # option_inds, prompt_format
        self._args = cfg
        self._init_flag = False
        self.problems, self.cand_pids, self.train_pids = None, None, None
        self.problem_id = None
        self.cand_examples = []
        openai.api_key = cfg.api_key
        self.observation_space = None
        self.action_space = None
        self.reward_space = gym.spaces.Box(
                low=-1, high=1, shape=(1, ), dtype=np.float32
            )

    def seed(self, seed, dynamic_seed=False):
        self._args.seed = seed

    def reset(self):
        self.problems, self.cand_pids, self.train_pids = load_data(self._args)
        self.cand_examples = []
        for pid in self.cand_pids:
            example = create_example_from_pid(pid, self.problems, self._args, test=True)
            self.cand_examples.append(example)

        self._init_flag = True
        self.problem_id = 0
        train_sample = create_example_from_pid(self.train_pids[self.problem_id], self.problems, self._args, test=True)
        obs = {'train_sample': train_sample, 'candidate_samples': self.cand_examples}
        return obs

    def close(self):
        self._init_flag = False

    def step(self, action):
        cids = action
        shot_pids = [self.cand_pids[cid] for cid in cids]
        # print(f"shot_pids: {shot_pids}")

        # generate the prompt input
        prompt = build_prompt(self.problems, shot_pids, self.train_pids[self.problem_id], self._args)

        # get the output from GPT-3
        output = get_gpt3_output(prompt, self._args)

        # extract the prediction from the output
        prediction = extract_prediction(output, self.problems[self.problem_id]['choices'], self._args.option_inds)

        # normalize the number in the text
        prediction_norm = normalize_answer(prediction, self.problems[self.problem_id]['unit'])

        if prediction_norm.lower() == normalize_answer(self.problems[self.problem_id]['answer'],
                                                       self.problems[self.problem_id]['unit']).lower():
            _reward = 1
        else:
            _reward = -1

        self.problem_id += 1
        if self.problem_id == self._args.train_number:
            done = True
        else:
            done = False
        info = {}

        train_sample = create_example_from_pid(self.train_pids[self.problem_id], self.problems, self._args, test=True)
        obs = {'train_sample': train_sample, 'candidate_samples': self.cand_examples}

        return obs, _reward, done, info

    def __repr__(self) -> str:
        return "DI-engine tabmwp Env"

