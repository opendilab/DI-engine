import gym

from ding.utils import ENV_REGISTRY
from ding.envs import BaseEnv, BaseEnvTimestep
from dizoo.tabmwp.envs.utils import *


@ENV_REGISTRY.register('tabmwp')
class TabMWP(BaseEnv):
    model = None
    tokenizer = None

    def __init__(self, cfg):
        self._args = cfg
        self.enable_replay = cfg.enable_replay
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
        assert self._args.engine in ['text-davinci-002', 'glm-10B', 'rwkv-7B', 'internlm-7B']
        if self._args.engine == 'glm-10B' and TabMWP.model is None:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            TabMWP.tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-10b", trust_remote_code=True)
            model = AutoModelForSeq2SeqLM.from_pretrained("THUDM/glm-10b", trust_remote_code=True)
            TabMWP.model = model.half().cuda()
        elif self._args.engine == 'rwkv-7B' and TabMWP.model is None:
            from transformers import AutoTokenizer, RwkvForCausalLM
            TabMWP.tokenizer = AutoTokenizer.from_pretrained("sgugger/rwkv-7b-pile", trust_remote_code=True)
            model = RwkvForCausalLM.from_pretrained("sgugger/rwkv-7b-pile")
            TabMWP.model = model.half().cuda()
        elif self._args.engine == 'internlm-7B' and TabMWP.model is None:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            TabMWP.tokenizer = AutoTokenizer.from_pretrained("internlm/internlm-7b", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained("internlm/internlm-7b", trust_remote_code=True).cuda()
            TabMWP.model = model.eval()

    @lru_cache(maxsize=10000)
    def get_output(self, inp: str) -> str:
        inputs = TabMWP.tokenizer(inp + " [MASK].", return_tensors="pt")
        inputs = TabMWP.tokenizer.build_inputs_for_generation(inputs, max_gen_length=512)
        inputs = {key: value.cuda() for key, value in inputs.items()}
        outputs = TabMWP.model.generate(**inputs, max_length=512, eos_token_id=TabMWP.tokenizer.eop_token_id,
                                        pad_token_id=TabMWP.tokenizer.eos_token_id)
        outputs = TabMWP.tokenizer.decode(outputs[0].tolist())

        t0 = outputs.find('<|startofpiece|>') + 16
        t1 = outputs.find('<|endofpiece|>')

        return outputs[t0:t1]

    def seed(self, seed: int, dynamic_seed: bool = False) -> None:
        self._args.seed = seed

    def reset(self) -> dict:
        self.problems, self.cand_pids, self.train_pids = load_data(self._args)
        if self.enable_replay:
            with open('sampled_pids.txt') as f:
                tmp = f.read().split('\n')
                a, b = tmp[0], tmp[1]
            self.cand_pids, self.train_pids = eval(a), eval(b)
            self.cand_pids = self.cand_pids[:self._args.cand_number]
            self.train_pids = self.train_pids[:self._args.train_number]
            self.results_memory = []
            with open('model_in_and_out.txt') as f:
                tmp = f.read().split('\n')
            for tt in tmp:
                if len(tt.strip()) == 0:
                    continue
                self.results_memory.append(eval(tt))

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

    def search_answer(self, pid, pids):
        for item in self.results_memory:
            if item['pid'] != pid:
                continue
            if item['shot_pids'] == pids:
                return item['output']

        raise ValueError('item does not exists.')

    def parse_all_answers(self):
        n = len(self.cand_pids)
        self.problem_id = 0

        with open('sampled_pid.txt', 'w') as f:
            f.write(str(self.cand_pids) + '\n')
            f.write(str(self.train_pids) + '\n')

        with open('model_in_out.txt', 'w') as f:
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    while self.problem_id < self._args.train_number:
                        shot_pids = [self.cand_pids[i], self.cand_pids[j]]
                        pid = self.train_pids[self.problem_id]

                        # generate the prompt input
                        prompt = build_prompt(self.problems, shot_pids, pid, self._args)

                        # get the output from LM
                        # assert self._args.engine == 'text-davinci-002'
                        output = get_gpt3_output(prompt, self._args)
                        self.problem_id += 1

                        output_txt = {'shot_pids': shot_pids, 'pid': pid, 'prompt': prompt, 'output': output}
                        f.write(str(output_txt) + '\n')

    def close(self) -> None:
        self._init_flag = False

    def step(self, action: np.array) -> BaseEnvTimestep:
        cids = []
        act = action.item()
        # Convert action of one scalar in to indexes.
        idx = 0
        while act > 0:
            if act % 2 != 0:
                cids.append(idx)
            act = act // 2
            idx += 1

        shot_pids = [self.cand_pids[cid] for cid in cids]
        pid = self.train_pids[self.problem_id]

        # generate the prompt input
        prompt = build_prompt(self.problems, shot_pids, pid, self._args)

        # get the output from LM
        if self.enable_replay:
            output = self.search_answer(pid, shot_pids)
        elif self._args.engine == 'text-davinci-002':
            output = get_gpt3_output(prompt, self._args)
        elif self._args.engine == 'rwkv-7B':
            output = calc_rwkv(self.model, self.tokenizer, prompt)
        elif self._args.engine == 'internlm-7B':
            output = calc_internlm(self.model, self.tokenizer, prompt, self._args)
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


if __name__ == '__main__':
    from easydict import EasyDict
    env_cfg = EasyDict(dict(
        cand_number=16,
        train_number=20,
        engine='text-davinci-002',
        temperature=0.,
        max_tokens=512,
        top_p=1.,
        frequency_penalty=0.,
        presence_penalty=0.,
        option_inds=["A", "B", "C", "D", "E", "F"],
        api_key='xxx',
        prompt_format='TQ-A',
        enable_replay=True,
        seed=0,
    ))
    env = TabMWP(env_cfg)
    env.seed(0)
    env.reset()
    env.search_answer('22976', ['32889', '8044'])

