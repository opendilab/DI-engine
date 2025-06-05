from transformers import AutoTokenizer
from typing import List, Tuple, Optional, Any
import os
from easydict import EasyDict
from datasets import load_dataset
from ding.worker.collector.vllm_collector import VllmCollector, get_free_gpus
import copy
import concurrent.futures
import pytest


def chunk_list(original_list: List, t: int) -> List[List]:
    # chunk a list into sub_lists
    # base length of sublists
    base_length = len(original_list) // t
    # remaind length of some sub_lists
    remainder = len(original_list) % t
    new_list = []
    index = 0
    for i in range(t):
        if i < remainder:
            sublist_length = base_length + 1
        else:
            sublist_length = base_length
        new_list.append(original_list[index:index + sublist_length])
        index += sublist_length
    return new_list


# prepare dataset
IMG_START_TOKEN = '<|vision_start|>'
IMG_END_TOKEN = '<|vision_end|>'
PLACE_HOLDER = '<|image_pad|>'


def dataset(num: int = None) -> List:
    # Load the dataset
    hf_dataset = load_dataset("MMInstruction/VL-RewardBench", split='test')
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
        ret_data = []
        for i in range(0, num):
            ret_data.append(hf_dataset[i])
        return ret_data


def run_vllm_collector(config: EasyDict) -> List[dict]:
    '''
    ret:[
        {
        "prompt_i":output([output_text_0,output_text_1,...,])
        }
    ]
    '''
    # set GPU for current process
    gpu_ids = ",".join(map(str, config.free_gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    collector = VllmCollector(config)
    #ret=collector.collect(n_samples=2,num_samples_per_prompt=4)
    ret = collector.collect(n_samples=config.n_samples, num_samples_per_prompt=config.num_samples_per_prompt)
    return ret


def start_collector(config: EasyDict):
    # collect within the process
    # results:a dict, basic form:
    #{"prompt_0":[ans_0,ans_1,...,ans_n],"prompt_1":[ans_0,ans_1,...,ans_n],...}
    results = run_vllm_collector(config)
    return results


def multi_vllm_main(tot_dataset, free_gpus: list, config: EasyDict):
    '''
    tot_dataset: the total dataset to process
    free_gpus: list of total gpus available for the task
    config: user defined config about how to do the task
    '''
    num_gpu_per_collector = config.num_gpus_per_collector
    # how many collector to use
    num_collector = len(free_gpus) // num_gpu_per_collector
    # list of list, each list contains the gpus the collecor can use
    gpu_per_collector = chunk_list(free_gpus, num_collector)
    prompts_per_gpu = chunk_list(tot_dataset, num_collector)
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_collector) as executor:
        futures = []
        for gpu_list, prompts_per_collector in zip(gpu_per_collector, prompts_per_gpu):
            config_per_gpu = copy.deepcopy(config)
            config_per_gpu.dataset = prompts_per_collector
            config_per_gpu.free_gpus = gpu_list
            #config_per_gpu.n_samples = len(prompts_per_collector)
            config_per_gpu.n_samples = 2
            futures.append(executor.submit(start_collector, config_per_gpu))

        # collect all results
        all_results = []
        for future in concurrent.futures.as_completed(futures):
            all_results.append(future.result())
        return all_results

    # # save results
    # with open(config.save_path, "w") as f:
    #     for response in all_results:
    #         #print(response)
    #         for prompt in list(response.keys()):
    #             f.write(f"{prompt}:\n")
    #             for i, output in enumerate(response[prompt].outputs):
    #                 f.write(f'output_{i}:\n')
    #                 f.write(f"{output.text}\n")


@pytest.mark.unittest
def test_multi_vllm():
    test_dataset = dataset(num=16)
    free_gpus = get_free_gpus()
    config = EasyDict(
        # (str) LLM/VLM model path
        model_path='Qwen/Qwen2-VL-7B',
        # (int) Maximum number of tokens to generate per request
        max_tokens=4096,
        # (float) Temperature for sampling, 0 means greedy decoding
        temperature=1.0,
        # (dict) Multimodal processor kwargs for vision-language models
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        },  # defaul set to align with Qwen2-VL-7B
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
        save_path="your_path",
        # how many gpus a collector can use
        num_gpus_per_collector=1,
        num_samples_per_prompt=4
    )
    result = multi_vllm_main(test_dataset, free_gpus, config)
    collector_num = len(free_gpus) // config.num_gpus_per_collector
    assert len(result) == collector_num
    for response in result:
        prompts = list(response.keys())
        for prompt in prompts:
            assert config.num_samples_per_prompt == len(response[prompt].outputs)
