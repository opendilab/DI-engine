import os
from functools import lru_cache

import gym
import openai
import numpy as np

from ding.utils import ENV_REGISTRY
from ding.envs import BaseEnv, BaseEnvTimestep
from dizoo.tabmwp.envs.utils import create_example_from_pid, build_prompt, get_gpt3_output, calc_rwkv, calc_internlm,\
    extract_prediction, normalize_answer, load_data


@ENV_REGISTRY.register('tabmwp')
class TabMWP(BaseEnv):
    model = None
    tokenizer = None

    def __init__(self, cfg):
        self.cfg = cfg
        self.enable_replay = cfg.enable_replay
        self._init_flag = False
        self.problems, self.cand_pids, self.train_pids = None, None, None
        self.problem_id = 0
        self.cand_examples = []
        openai.api_key = cfg.api_key
        self.observation_space = gym.spaces.Dict()
        self.action_space = gym.spaces.Discrete(self.cfg.cand_number * (self.cfg.cand_number - 1))
        self.reward_space = gym.spaces.Box(low=-1, high=1, shape=(1, ), dtype=np.float32)
        self.correct_num = 0

        # Initialize language model if needed.
        assert self.cfg.engine in ['text-davinci-002', 'glm-10B', 'rwkv-7B', 'internlm-7B']

        try:
            if self.cfg.engine == 'glm-10B' and TabMWP.model is None:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                TabMWP.tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-10b", trust_remote_code=True)
                model = AutoModelForSeq2SeqLM.from_pretrained("THUDM/glm-10b", trust_remote_code=True)
                TabMWP.model = model.half()
            elif self.cfg.engine == 'rwkv-7B' and TabMWP.model is None:
                from transformers import AutoTokenizer, RwkvForCausalLM
                TabMWP.tokenizer = AutoTokenizer.from_pretrained("sgugger/rwkv-7b-pile", trust_remote_code=True)
                model = RwkvForCausalLM.from_pretrained("sgugger/rwkv-7b-pile")
                TabMWP.model = model.half()
            elif self.cfg.engine == 'internlm-7B' and TabMWP.model is None:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                TabMWP.tokenizer = AutoTokenizer.from_pretrained("internlm/internlm-7b", trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained("internlm/internlm-7b", trust_remote_code=True)
                TabMWP.model = model.eval()
        except ImportError:
            import sys
            from ditk import logging
            logging.warning("not found transformer, please install it using: pip install transformers")
            sys.exit(1)

    @lru_cache(maxsize=10000)
    def get_output(self, inp: str) -> str:
        inputs = TabMWP.tokenizer(inp + " [MASK].", return_tensors="pt")
        inputs = TabMWP.tokenizer.build_inputs_for_generation(inputs, max_gen_length=512)
        inputs = {key: value.cuda() for key, value in inputs.items()}
        outputs = TabMWP.model.generate(
            **inputs,
            max_length=512,
            eos_token_id=TabMWP.tokenizer.eop_token_id,
            pad_token_id=TabMWP.tokenizer.eos_token_id
        )
        outputs = TabMWP.tokenizer.decode(outputs[0].tolist())

        t0 = outputs.find('<|startofpiece|>') + 16
        t1 = outputs.find('<|endofpiece|>')

        return outputs[t0:t1]

    def seed(self, seed: int, dynamic_seed: bool = False) -> None:
        self.cfg.seed = seed

    def reset(self) -> dict:
        self.problems, self.cand_pids, self.train_pids = load_data(self.cfg)
        if TabMWP.model is not None:
            TabMWP.model = TabMWP.model.cuda()
        if self.enable_replay:
            self.cand_pids = [
                '32889', '8044', '16892', '5408', '4051', '37355', '17962', '25807', '30602', '5514', '19270', '23713',
                '17209', '33379', '34987', '11177'
            ]
            if self.cfg.seed == 0:  # train
                self.train_pids = [
                    '14229', '3409', '29980', '799', '5086', '21778', '36441', '34146', '69', '33433', '26979', '18135',
                    '13347', '17679', '38426', '3454', '10432', '31011', '12162', '13063', '7812', '29661', '24482',
                    '4970', '4405', '17405', '27781', '26724', '5993', '16442', '30148', '15895', '6855', '29903',
                    '18107', '29504', '11106', '32964', '29891', '32104', '15712', '24287', '4997', '32581', '21020',
                    '17247', '31455', '13245', '15850', '10011', '10313', '10158', '1817', '33479', '35842', '14198',
                    '26039', '3791', '4909', '37056', '7144', '8185', '2131', '4398', '38199', '29520', '37329',
                    '21388', '28659', '15044', '28510', '12903', '11794', '37095', '32229', '22918', '31680', '15024',
                    '24607', '26930'
                ]
                model_io_path = 'dizoo/tabmwp/data/model_in_out_train.txt'
                if not os.path.exists(model_io_path):
                    os.system(
                        f'wget https://opendilab.net/download/DI-zoo/tabmwp/model_in_out_train.txt -O ' +
                        model_io_path + ' --no-check-certificate'
                    )
            else:
                self.train_pids = [
                    '21037', '22976', '2224', '14145', '27962', '26553', '22110', '16541', '26044', '19492', '31882',
                    '11991', '27594', '7637', '15394', '7666', '5177', '33761', '13703', '29105'
                ]
                model_io_path = 'dizoo/tabmwp/data/model_in_out_eval.txt'
                os.system(
                    f'wget https://opendilab.net/download/DI-zoo/tabmwp/model_in_out_eval.txt -O ' + model_io_path +
                    ' --no-check-certificate'
                )

            self.cfg.cand_number = len(self.cand_pids)
            self.cfg.train_number = len(self.train_pids)

            self.results_memory = []
            with open(model_io_path, encoding="ISO-8859-1") as f:
                tmp = f.read().split('\n')
            for tt in tmp:
                if len(tt.strip()) == 0:
                    continue
                self.results_memory.append(eval(tt))

        self.cand_examples = []
        self.correct_num = 0
        for pid in self.cand_pids:
            example = create_example_from_pid(pid, self.problems, self.cfg, test=True)
            self.cand_examples.append(example)

        self._init_flag = True
        self.problem_id = 0
        train_sample = create_example_from_pid(self.train_pids[self.problem_id], self.problems, self.cfg, test=True)
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
        self.cand_pids = [
            '32889', '8044', '16892', '5408', '4051', '37355', '17962', '25807', '30602', '5514', '19270', '23713',
            '17209', '33379', '34987', '11177', '30218', '26066', '24169', '28492'
        ]
        self.train_pids = [
            '14229', '3409', '29980', '799', '5086', '21778', '36441', '34146', '69', '33433', '26979', '18135',
            '13347', '17679', '38426', '3454', '10432', '31011', '12162', '13063', '7812', '29661', '24482', '4970',
            '4405', '17405', '27781', '26724', '5993', '16442', '30148', '15895', '6855', '29903', '18107', '29504',
            '11106', '32964', '29891', '32104', '15712', '24287', '4997', '32581', '21020', '17247', '31455', '13245',
            '15850', '10011', '10313', '10158', '1817', '33479', '35842', '14198', '26039', '3791', '4909', '37056',
            '7144', '8185', '2131', '4398', '38199', '29520', '37329', '21388', '28659', '15044', '28510', '12903',
            '11794', '37095', '32229', '22918', '31680', '15024', '24607', '26930'
        ]
        self.problem_id = 0
        self.cfg.train_number = len(self.train_pids)
        n = len(self.cand_pids)

        with open('sampled_pid.txt', 'w') as f:
            f.write(str(self.cand_pids) + '\n')
            f.write(str(self.train_pids) + '\n')

        with open('model_in_out.txt', 'w') as f:
            while self.problem_id < self.cfg.train_number:
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            continue
                        shot_pids = [self.cand_pids[i], self.cand_pids[j]]
                        pid = self.train_pids[self.problem_id]

                        # generate the prompt input
                        prompt = build_prompt(self.problems, shot_pids, pid, self.cfg)

                        # get the output from LM
                        # assert self._args.engine == 'text-davinci-002'
                        output = get_gpt3_output(prompt, self.cfg)

                        output_txt = {'shot_pids': shot_pids, 'pid': pid, 'prompt': prompt, 'output': output}
                        f.write(str(output_txt) + '\n')
                        print(self.problem_id, i, j)

                self.problem_id += 1

    def close(self) -> None:
        self._init_flag = False

    def step(self, action: np.array) -> BaseEnvTimestep:
        shot_pids = [self.cand_pids[cid] for cid in action]
        pid = self.train_pids[self.problem_id]

        # generate the prompt input
        prompt = build_prompt(self.problems, shot_pids, pid, self.cfg)

        # get the output from LM
        if self.enable_replay:
            output = self.search_answer(pid, shot_pids)
        elif self.cfg.engine == 'text-davinci-002':
            output = get_gpt3_output(prompt, self.cfg)
        elif self.cfg.engine == 'rwkv-7B':
            output = calc_rwkv(self.model, self.tokenizer, prompt)
        elif self.cfg.engine == 'internlm-7B':
            output = calc_internlm(self.model, self.tokenizer, prompt, self.cfg)
        else:
            output = self.get_output(prompt)

        # extract the prediction from the output
        prediction = extract_prediction(output, self.problems[pid]['choices'], self.cfg.option_inds)

        # normalize the number in the text
        prediction_norm = normalize_answer(prediction, self.problems[pid]['unit'])

        if prediction_norm.lower() == normalize_answer(self.problems[pid]['answer'],
                                                       self.problems[pid]['unit']).lower():
            reward = 1
            self.correct_num += 1
        else:
            reward = -1

        self.problem_id += 1
        if self.problem_id == self.cfg.train_number:
            done = True
            info = {'eval_episode_return': self.correct_num / self.cfg.train_number}
        else:
            done = False
            info = {}

        train_sample = create_example_from_pid(pid, self.problems, self.cfg, test=True)
        obs = {'train_sample': train_sample, 'candidate_samples': self.cand_examples}

        return BaseEnvTimestep(obs, reward, done, info)

    def __repr__(self) -> str:
        return "DI-engine tabmwp Env"


if __name__ == '__main__':
    from easydict import EasyDict
    env_cfg = EasyDict(
        dict(
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
        )
    )
    env = TabMWP(env_cfg)
    env.seed(0)
    env.reset()
    env.parse_all_answers()
    env.search_answer('22976', ['32889', '8044'])
