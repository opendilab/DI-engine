from typing import List, Tuple, Optional
from ding.worker.collector.vllm_collector import HuggingFaceModelGenerator, get_free_gpus
from vllm.assets.image import ImageAsset
from enum import Enum
from datasets import load_dataset
import asyncio
from PIL import Image
import os
import concurrent.futures
import pytest


class Modality(Enum):
    IMAGE = "image"
    TEXT = "text"
    VIDEO = "video"


def chunk_list(original_list: List, t: int):
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


def get_prompts_qwen(questions: list, modality: Modality) -> Tuple[List[str], Optional[List[int]]]:
    if modality == Modality.IMAGE:
        placeholder = "<|image_pad|>"
    elif modality == Modality.VIDEO:
        placeholder = "<|video_pad|>"
    else:
        msg = f"Modality {modality} is not supported."
        raise ValueError(msg)

    prompts = [
        (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ) for question in questions
    ]
    stop_token_ids = None
    return prompts, stop_token_ids


def get_multi_modal_input(modality: Modality, filenames: list, questions: list) -> dict:
    """
    return {
        "data": image or video,
        "question": question,
    }
    """
    if modality == Modality.IMAGE:
        # Input image and question
        ret = {'data': [], 'question': []}
        for filename, question in zip(filenames, questions):
            if isinstance(filename, str):
                image = ImageAsset(filename) \
                .pil_image.convert("RGB")
            #img_question = "What is the content of this image?"
            elif isinstance(filename, Image.Image):
                image = filename
            else:
                raise ValueError(f"Unsupported type in filenames: {type(filename)}")
            img_question = question
            ret["data"].append(image)
            ret["question"].append(img_question)
    else:
        msg = f"Modality {modality} is not supported."
        raise ValueError(msg)
    return ret


# -----------------testing single gpu vllm_actor --------------------------------
async def single_main(model_path: str, gpu: list, temperature: float, modality: str, prompts: list, data: list):
    # note that HFModelGenerator has a parameter
    # "mm_processor_kwargs" set to align with the settings of Qwen in default
    model = HuggingFaceModelGenerator(model_path=model_path, free_gpus=gpu, temperature=temperature)
    inputs = [{"prompt": prompt, "multi_modal_data": {modality: data}} for prompt, data in zip(prompts, data)]
    # generate responses
    response_ret = []
    for in_data in inputs:
        responses = await model.generate(prompt=in_data, num_samples=3)
        # print response
        response_per_prompt = []
        for response, confidence in responses:
            response_per_prompt.append(response)
        response_ret.append(response_per_prompt)
    return response_ret


# run main
@pytest.mark.unittest
def test_single_main():
    # set a temperature > 0 to get multiple responses
    free_gpus = get_free_gpus()
    model_path = 'Qwen/Qwen2-VL-7B'
    temperature = 0.5
    questions = []
    img_names = []
    sample_num = 4
    hf_dataset = load_dataset("MMInstruction/VL-RewardBench", split='test')
    for i in range(sample_num):
        img_names.append(hf_dataset[i]["image"])
        questions.append(hf_dataset[i]["query"])
    assert len(img_names) == len(questions)
    modality = Modality.IMAGE
    mm_input = get_multi_modal_input(modality, img_names, questions)
    data = mm_input["data"]
    question = mm_input["question"]
    prompts, stop_token_ids = get_prompts_qwen(question, modality)
    responses = asyncio.run(
        single_main(
            model_path=model_path,
            gpu=[free_gpus[0]],
            temperature=temperature,
            modality=modality.value,
            prompts=prompts,
            data=data
        )
    )
    assert len(responses) == len(questions)


# -----------------testing multi gpu vllm_actor --------------------------------
async def run_vllm_collector(gpu_list: list, prompts: List, model_path: str, temperature: float) -> List[str]:
    # set visible gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_list))
    # get a model on a single gpu
    model = HuggingFaceModelGenerator(model_path, free_gpus=gpu_list, temperature=temperature)

    # get response for each prompts (can be improved later using async generation)
    responses_list = []
    for prompt in prompts:
        responses = await model.generate(prompt, num_samples=3)
        for response in responses:
            responses_list.append(response)
            #print(f"[GPU {gpu_list}] Response: {response}")

    return responses_list


def start_collector(gpu_list: list, prompts: list, model_path: str, temperature: float) -> List[str]:
    # event loop in a process
    results = asyncio.run(run_vllm_collector(gpu_list, prompts, model_path, temperature))
    return results


def multi_main(
        prompts: list, model_path: str, free_gpus: List[int], temperature: float, num_per_gpus_collector: int
) -> None:
    # solve how mant collectors to use
    num_collector = len(free_gpus) // num_per_gpus_collector
    # slove how many gpus a collector should use
    gpus_per_collector = chunk_list(free_gpus, num_collector)
    # split input_prompts to collectors equally
    prompts_per_gpu = chunk_list(prompts, num_collector)
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_collector) as executor:
        futures = []
        for gpu_list, prompts_gpu in zip(gpus_per_collector, prompts_per_gpu):
            futures.append(executor.submit(start_collector, gpu_list, prompts_gpu, model_path, temperature))

        # get all results
        all_results = []
        for future in concurrent.futures.as_completed(futures):
            all_results.append(future.result())

    return all_results


@pytest.mark.unittest
def test_multi_main():
    # get dataset
    hf_dataset = load_dataset("MMInstruction/VL-RewardBench", split='test')
    img_names = []
    questions = []
    num = 16
    for i in range(num):
        img_names.append(hf_dataset[i]["image"])
        questions.append(hf_dataset[i]["query"])
    assert len(img_names) == len(questions)
    #get gpus
    free_gpus = get_free_gpus()
    # set modality
    modality = Modality.IMAGE
    # get input
    mm_input = get_multi_modal_input(modality, img_names, questions)
    data = mm_input["data"]
    question = mm_input["question"]
    # get prompts
    prompts, stop_token_ids = get_prompts_qwen(question, modality)
    # set necessary parameters
    model_path = 'Qwen/Qwen2-VL-7B'
    temperature = 0.5
    num_gpus_per_collector = 1
    assert len(free_gpus) >= num_gpus_per_collector
    # set inputs
    inputs = [{"prompt": prompt, "multi_modal_data": {modality.value: data}} for prompt, data in zip(prompts, data)]
    # get results
    result = multi_main(inputs, model_path, free_gpus, temperature, num_gpus_per_collector)
    # default num_smaples is 3, can be modified in line 93
    assert len(result) == len(questions)
