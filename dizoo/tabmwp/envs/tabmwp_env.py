import gym

from ding.utils import ENV_REGISTRY

from .utils import *
from ding.envs import BaseEnv, BaseEnvTimestep
import openai


@ENV_REGISTRY.register('tabmwp')
class TabMWP(BaseEnv):
    model = None
    tokenizer = None

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
            low=-1, high=1, shape=(1,), dtype=np.float32
        )
        self.correct_num = 0
        assert self._args.engine in ['text-davinci-002', 'glm-10B']
        if self._args.engine == 'glm-10B' and TabMWP.model is None:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            TabMWP.tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-10b-chinese", trust_remote_code=True)
            model = AutoModelForSeq2SeqLM.from_pretrained("THUDM/glm-10b-chinese", trust_remote_code=True)
            TabMWP.model = model.half().cuda()

    def get_output(self, inp):
        inputs = TabMWP.tokenizer(inp + " [sMASK]", return_tensors="pt")
        inputs = TabMWP.tokenizer.build_inputs_for_generation(inputs, max_gen_length=512)
        inputs = {key: value.cuda() for key, value in inputs.items()}
        outputs = TabMWP.model.generate(**inputs, max_length=512, eos_token_id=TabMWP.tokenizer.eop_token_id,
                                        pad_token_id=TabMWP.tokenizer.eos_token_id)
        outputs = TabMWP.tokenizer.decode(outputs[0].tolist())

        t0 = outputs.find('<|startofpiece|>') + 16
        t1 = outputs.find('<|endofpiece|>')

        return outputs[t0:t1]

    def seed(self, seed, dynamic_seed=False):
        self._args.seed = seed

    def reset(self):
        self.problems, self.cand_pids, self.train_pids = load_data(self._args)
        self.cand_examples = []
        self.correct_num = 0
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
        cids = []
        act = action.item()
        idx = 0
        while act > 0:
            if act % 2 != 0:
                cids.append(idx)
            act = act // 2
            idx += 1
        assert len(cids) == self._cfg.shot_number
        shot_pids = [self.cand_pids[cid] for cid in cids]
        # print(f"shot_pids: {shot_pids}")

        pid = self.train_pids[self.problem_id]

        # generate the prompt input
        prompt = build_prompt(self.problems, shot_pids, pid, self._args)

        # get the output from LM
        if self._args.engine == 'text-davinci-002':
            output = get_gpt3_output(prompt, self._args)
        else:
            output = self.get_output(prompt)

        # extract the prediction from the output
        prediction = extract_prediction(output, self.problems[pid]['choices'], self._args.option_inds)

        # normalize the number in the text
        prediction_norm = normalize_answer(prediction, self.problems[pid]['unit'])

        if prediction_norm.lower() == normalize_answer(self.problems[pid]['answer'],
                                                       self.problems[pid]['unit']).lower():
            _reward = 1
            self.correct_num += 1
        else:
            _reward = -1

        self.problem_id += 1
        if self.problem_id == self._args.train_number:
            done = True
            info = {'eval_episode_return': self.correct_num / self._args.train_number}
        else:
            done = False
            info = {}

        train_sample = create_example_from_pid(pid, self.problems, self._args, test=True)
        obs = {'train_sample': train_sample, 'candidate_samples': self.cand_examples}

        return BaseEnvTimestep(obs, _reward, done, info)

    def __repr__(self) -> str:
        return "DI-engine tabmwp Env"
