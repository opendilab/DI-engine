from typing import Any, Dict, Union, Callable, Iterable,List
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.distributed import get_rank
from transformers import AutoTokenizer
from typing import List, Tuple, Optional, Any
import os
import uuid
import asyncio
import numpy as np
from loguru import logger
from easydict import EasyDict
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams, RequestOutput
from ding.utils import SERIAL_COLLECTOR_REGISTRY
from ding.worker.collector.base_serial_collector import ISerialCollector
from datasets import load_dataset
from ding.utils.data import OnlineRLDataset
import copy
import concurrent.futures


class VllmActor:
    def __init__(self, model_path: str,mm_processor_kwargs: dict,free_gpus:list) -> None:
        """
        Overview:
            Initialize the vLLM actor. For more details, please refer to https://docs.vllm.ai/en/stable.
        Arguments:
            - model_path (str): The path to the language model.
        """
        self.free_gpus = free_gpus
        self.num_gpus = len(self.free_gpus)
        assert self.num_gpus > 0, "No GPUs found"
        # Set CUDA_VISIBLE_DEVICES to use only free GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.free_gpus))
        self.model_path = model_path
        self.mm_processor_kwargs=mm_processor_kwargs
        self._initialize()

    def _initialize(self) -> None:
        """
        Overview:
            Initialize the vLLM actor with a series of arguments.
        """
        logger.info("Initializing vLLM")
        # TODO: Try other options in https://docs.vllm.ai/en/stable/models/engine_args.html#engine-args.
        engine_args = AsyncEngineArgs(
            model=self.model_path,
            tensor_parallel_size=self.num_gpus,
            max_num_batched_tokens=8192,
            max_model_len=8192,
            # enable_chunked_prefill=True,
            max_num_seqs=5,
            # Note - mm_processor_kwargs can also be passed to generate/chat calls
            mm_processor_kwargs=self.mm_processor_kwargs,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def generate(self, prompt, num_samples: int, max_tokens: int, temperature: float = 0) -> RequestOutput:
        """
        Overview:
            Generate tactics for the current state.
        Arguments:
            - prompt : The prompt to generate tactics.
            - num_samples (int): The number of tactics to generate.
            - max_tokens (int): The maximum number of tokens to generate.
            - temperature (float): The temperature for the language model, default to 0.
        Returns:
            - RequestOutput: The generated tactics and their log-probabilities.
        """
        sampling_params = SamplingParams(
            n=num_samples,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        # Using async iterator to handle vLLM's generation process
        # 1. vLLM's generate method is asynchronous to prevent blocking while waiting for model outputs
        # 2. async for allows streaming the generated outputs incrementally instead of waiting for all results
        # 3. This approach is particularly suitable for LLM inference which can be time-consuming
        # 4. The request_id ensures unique identification for each generation request
        async for oup in self.engine.generate(
            prompt, sampling_params, request_id=str(uuid.uuid4().hex)
        ):
            final_output = oup
        return final_output


@SERIAL_COLLECTOR_REGISTRY.register('vllm')
class VllmCollector(ISerialCollector):
    """
    Overview:
        Collector implementation for vLLM-based language models (LLM/VLM).
        This collector manages the interaction with vLLM models for text generation tasks.
    """
    config = dict(
        # (str) LLM/VLM model path
        model_path='',
        # (int) Maximum number of tokens to generate per request
        max_tokens=1024,
        # (float) Temperature for sampling, 0 means greedy decoding
        temperature=0.0,
        # (dict) Multimodal processor kwargs for vision-language models
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        },
        # Dataset related configs
        # (str) Key to access the input data in the dataset
        input_key='input',
        # (bool) Whether to apply a chat template to the input
        apply_chat_template=False,
        # (str) Template for the input
        input_template=None,
        # (bool) Whether to shuffle the dataset
        shuffle=True,
    )

    def __init__(self, cfg: EasyDict) -> None:
        """
        Overview:
            Initialize the VllmCollector with configuration.
        Arguments:
            - cfg (:obj:`EasyDict`): Configuration for the collector including model path, generation parameters,
              and dataset configuration
        """
        super().__init__()
        self._cfg = cfg
        self._envstep = 0

        # Initialize the tokenizer and dataset
        self._tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
        self._dataset = OnlineRLDataset(
            dataset=cfg.dataset,
            tokenizer=self._tokenizer,
            input_key=cfg.input_key,
            apply_chat_template=cfg.apply_chat_template,
            input_template=cfg.input_template,
            extra_input_keys=cfg.extra_input_keys
        )

        self._model = VllmActor(model_path=cfg.model_path, mm_processor_kwargs=cfg.mm_processor_kwargs,free_gpus=cfg.free_gpus)
        self.reset()

    def reset(self) -> None:
        """
        Overview:
            Reset the collector, including the dataset index.
        """
        self._index = np.arange(len(self._dataset))
        if self._cfg.shuffle:
            np.random.shuffle(self._index)

    def reset_policy(self, _model: Optional[str] = None) -> None:
        """
        Overview:
            Since LLM generation does not require a explicit policy and env, this function is empty.
        """
        pass

    def reset_env(self, _env: Optional[Any] = None) -> None:
        """
        Overview:
            Since LLM generation does not require a explicit policy and env, this function is empty.
        """
        pass
    async def _generate_for_prompt(self, prompt: str, num_samples_per_prompt: int) -> List[Tuple[str, float]]:
        return await self._model.generate(
            prompt=prompt,
            num_samples=num_samples_per_prompt,
            max_tokens=self._cfg.max_tokens,
            temperature=self._cfg.temperature
        )
    def collect(
            self,
            n_samples: int = 100,
            num_samples_per_prompt: int = 1,
            train_iter: int = 0,
    ) -> List[Tuple[str, float]]:
        """
        Overview:
            Collect generated responses from the vLLM model.
        Arguments:
            - n_samples (:obj:`int`): Number of prompts to generate.
            - num_samples_per_prompt (:obj:`int`): Number of samples to generate per prompt.
            - train_iter (:obj:`int`): Current training iteration, used for logging.
        Returns:
            - responses (:obj:`List[Tuple[str, float]]`): List of (generated_text, confidence_score) pairs
        """
        if self._model is None:
            raise RuntimeError("Model not initialized. Call `reset` method first.")

        prompts=[]
        for id in self._index[:n_samples]:
            prompts.append(self._dataset[id])
        # recusively update the index
        self._index = np.concatenate((self._index[n_samples:],self._index[:n_samples]))

        self._envstep += n_samples

        # Get the current event loop or create a new one
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Run the async generate method in the event loop
        # Create a list of tasks for each prompt
        tasks = [self._generate_for_prompt(prompt, num_samples_per_prompt) for prompt in prompts]

        # Run all tasks concurrently and collect results
        results = loop.run_until_complete(asyncio.gather(*tasks))

        # Map prompts to their corresponding results
        responses = {prompt["prompt"]: result for prompt, result in zip(prompts, results)}

        return responses
    
    def sync_collect(
            self,
            n_samples: int = 100,
            num_samples_per_prompt: int = 1,
            train_iter: int = 0,
    ) -> List[Tuple[str, float]]:
        """
        Overview:
            Collect generated responses from the vLLM model.
        Arguments:
            - n_samples (:obj:`int`): Number of prompts to generate.
            - num_samples_per_prompt (:obj:`int`): Number of samples to generate per prompt.
            - train_iter (:obj:`int`): Current training iteration, used for logging.
        Returns:
            - responses (:obj:`List[Tuple[str, float]]`): List of (generated_text, confidence_score) pairs
        """
        if self._model is None:
            raise RuntimeError("Model not initialized. Call `reset` method first.")

        prompts=[]
        for id in self._index[:n_samples]:
            prompts.append(self._dataset[id])
        # recusively update the index
        self._index = np.concatenate((self._index[n_samples:],self._index[:n_samples]))

        self._envstep += n_samples

        # Get the current event loop or create a new one
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Run the async generate method in the event loop
        results = {}
        for prompt in prompts:
        # Run the async generate method in the event loop for each prompt
            result = loop.run_until_complete(
                self._model.generate(
                    prompt=prompt,
                    num_samples=num_samples_per_prompt,
                    max_tokens=self._cfg.max_tokens,
                    temperature=self._cfg.temperature
                )
            )
            results[prompt['prompt']] = result

        return results    
    
    def collect_prompts(
            self,
            n_samples: int = 100,
            num_samples_per_prompt: int = 1,
            train_iter: int = 0,
    ) -> List[Tuple[str, float]]:
        """
        Overview:
            Collect generated responses from the vLLM model.
        Arguments:
            - n_samples (:obj:`int`): Number of prompts to generate.
            - num_samples_per_prompt (:obj:`int`): Number of samples to generate per prompt.
            - train_iter (:obj:`int`): Current training iteration, used for logging.
        Returns:
            - responses (:obj:`List[Tuple[str, float]]`): List of (generated_text, confidence_score) pairs
        """
        if self._model is None:
            raise RuntimeError("Model not initialized. Call `reset` method first.")

        prompts=[]
        for id in self._index[:n_samples]:
            prompts.append(self._dataset[id])
        # recusively update the index
        self._index = np.concatenate((self._index[n_samples:],self._index[:n_samples]))

        self._envstep += n_samples

        # Get the current event loop or create a new one
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Run the async generate method in the event loop
        results = {}
        tasks=[]
        for prompt in prompts:
            for _ in range(num_samples_per_prompt): 
            # Run the async generate method in the event loop for each prompt
                tasks.append(self._generate_for_prompt(prompt, num_samples_per_prompt=1))
        results_list = loop.run_until_complete(asyncio.gather(*tasks))
        for i,prompt in enumerate(prompts):
            results[prompt['prompt']]=[]
            for result in results_list[i*num_samples_per_prompt:(i+1)*num_samples_per_prompt]:
                results[prompt['prompt']].append(result.outputs[0].text)
        return results    



    @property
    def envstep(self) -> int:
        """
        Overview:
            Get the current environment step count.
        Returns:
            - count (:obj:`int`): Current environment step count
        """
        return self._envstep

    @envstep.setter
    def envstep(self, value: int) -> None:
        """
        Overview:
            Set the current environment step count.
        """
        self._envstep = value

    def close(self) -> None:
        """
        Overview:
            Close the collector.
        """
        pass

    def __del__(self) -> None:
        """
        Overview:
            Destructor for the collector.
        """
        self.close()
        
        
        
        
def get_free_gpus() -> List[int]:
    """
    Overview:
        Get IDs of GPUs with free memory.
    Returns:
        - List[int]: The IDs of the free GPUs.
    """
    try:
        # Get GPU memory usage using nvidia-smi
        gpu_stats = os.popen('nvidia-smi --query-gpu=memory.used,memory.total --format=csv,nounits,noheader')\
            .readlines()
        free_gpus = []

        for gpu_id, stats in enumerate(gpu_stats):
            mem_used, mem_total = map(int, stats.strip().split(','))
            # Consider GPU as free if less than 5% memory is used
            if mem_used / mem_total < 0.05:
                free_gpus.append(gpu_id)

        return free_gpus if free_gpus else [0]  # Default to GPU 0 if no free GPUs found
    except Exception:
        logger.warning("Failed to get GPU stats, defaulting to GPU 0")
        return [0]
    
def chunk_list(original_list, t):
    # chunk a list into sub_lists
    new_list = [original_list[i:i + t] for i in range(0, len(original_list), t)]
    return new_list


# prepare dataset
IMG_START_TOKEN = '<|vision_start|>'
IMG_END_TOKEN = '<|vision_end|>'
PLACE_HOLDER='<|image_pad|>'
def dataset(num=None):
    # Load the dataset
    hf_dataset = load_dataset("/mnt/afs/wangqijian/data/rlhf_dataset_test/VL-RewardBench",split='test')
    hf_dataset0 = hf_dataset.map(
        lambda x: {
            "query": f"{IMG_START_TOKEN}{PLACE_HOLDER}{IMG_END_TOKEN}{x['query']}",
            "image": x["image"],
        }
    )
    # shuffle the dataset
    hf_dataset = hf_dataset0.shuffle(seed=42)
    if num is None:
        return hf_dataset
    else:
        ret_data=[]
        for i in range(0,num):
            ret_data.append(hf_dataset[i])
        return ret_data


def run_vllm_collector(config):
    # set GPU for current process
    gpu_ids = ",".join(map(str, config.free_gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    collector = VllmCollector(config)  # 实例化模型
    #ret=collector.collect(n_samples=2,num_samples_per_prompt=4)
    ret=collector.collect(n_samples=2,num_samples_per_prompt=4)
    return ret


def start_collector(config):
    # collect within the process
    # results:a dict, basic form:
    #{"prompt_0":[ans_0,ans_1,...,ans_n],"prompt_1":[ans_0,ans_1,...,ans_n],...}
    results = run_vllm_collector(config)
    return results

def main(tot_dataset, free_gpus,config):
    num_tot=len(tot_dataset)
    num_gpu=len(free_gpus)
    num_per_gpu=num_tot//num_gpu
    prompts_per_gpu=chunk_list(tot_dataset,num_per_gpu)
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(free_gpus)) as executor:
        futures = []
        for gpu_id,prompts_gpu in zip(free_gpus,prompts_per_gpu):
            config_per_gpu=copy.deepcopy(config)
            config_per_gpu.dataset=prompts_gpu
            config_per_gpu.free_gpus=[gpu_id]
            futures.append(executor.submit(start_collector, config_per_gpu))

        # collect all results
        all_results = []
        for future in concurrent.futures.as_completed(futures):
            all_results.append(future.result())

    # save results
    with open(config.save_path, "w") as f:
        for response in all_results:
            print(response)
            for prompt in list(response.keys()):
                f.write(f"{prompt}:\n")
                for i,output in enumerate(response[prompt].outputs):
                    f.write(f'output_{i}:\n')
                    f.write(f"{output.text}\n")
                    
                    
test_dataset=dataset(num=96)
free_gpus=get_free_gpus()                   
config = EasyDict(
        # (str) LLM/VLM model path
        model_path='/mnt/afs/share/Qwen2-VL-7B',
        # (int) Maximum number of tokens to generate per request
        max_tokens=4096,
        # (float) Temperature for sampling, 0 means greedy decoding
        temperature=1.0,
        # (dict) Multimodal processor kwargs for vision-language models
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        },# defaul set to align with Qwen2-VL-7B
        # Dataset related configs
        # dataset=test_dataset,
        # dataset is defined for each gpu respectively
        # (str) Key to access the input data in the dataset
        input_key='query',
        # (bool) Whether to apply a chat template to the input
        apply_chat_template=True,
        # (str) Template for the input
        input_template=None,
        # (bool) Whether to shuffle the dataset
        shuffle=True,
        extra_input_keys=['image'],
        # free_gpus is defined for each gpu respectively
        # save_path is the file to store the output
        save_path="your_save_path"
    )




main(test_dataset,free_gpus,config)