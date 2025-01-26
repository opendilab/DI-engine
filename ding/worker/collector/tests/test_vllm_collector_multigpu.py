from typing import List, Tuple
import os
import uuid
from loguru import logger
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams, RequestOutput
from typing import List, Tuple, Optional
from vllm.assets.image import ImageAsset
from enum import Enum
import concurrent.futures
import asyncio


class VllmActor:

    def __init__(self, model_path: str, mm_processor_kwargs: dict, free_gpus: list) -> None:
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
        self.mm_processor_kwargs = mm_processor_kwargs
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


class HuggingFaceModelGenerator:
    """
    Overview:
        A LLM/VLM generator that uses Hugging Face models with vLLM as the backend.
    """

    def __init__(
            self,
            model_path: str,
            free_gpus: list,
            max_tokens: int = 1024,
            temperature: float = 0,
            mm_processor_kwargs: dict = {
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
            }
    ) -> None:
        """
        Overview:
            Initialize the Hugging Face model generator.
        Arguments:
            - model_path (str): The path to the language model.
            - max_tokens (int): The maximum number of tokens to generate, default to 1024.
            - temperature (float): The temperature for the language model, default to 0.
        """
        self.vllm_actor = VllmActor(model_path, mm_processor_kwargs, free_gpus)
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def generate(
            self,
            prompt,
            num_samples: int,
    ) -> List[Tuple[str, float]]:
        """
        Overview:
            Generate tactics for the current state.
        Arguments:
            - prompt : The prompt to generate tactics.
            - num_samples (int): The number of tactics to generate.
        Returns:
            - List[Tuple[str, float]]: The generated tactics and their log-probabilities.

        .. note::
            This method is asynchronous and returns a coroutine.
        """
        response = await self.vllm_actor.generate(prompt, num_samples, self.max_tokens, self.temperature)
        # Use raw logprobs as confidence scores
        confidence_scores = [x.cumulative_logprob for x in response.outputs]
        return [(x.text.strip(), conf) for x, conf in zip(response.outputs, confidence_scores)]


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


def chunk_list(original_list: list, t: int) -> List[list]:
    # chunk the list into sub_lists
    new_list = [original_list[i:i + t] for i in range(0, len(original_list), t)]
    return new_list


class Modality(Enum):
    IMAGE = "image"
    TEXT = "text"
    VIDEO = "video"


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
            image = ImageAsset(filename) \
                .pil_image.convert("RGB")
            #img_question = "What is the content of this image?"
            img_question = question
            ret["data"].append(image)
            ret["question"].append(img_question)
    else:
        msg = f"Modality {modality} is not supported."
        raise ValueError(msg)
    return ret


async def run_vllm_collector(gpu_id: int, prompts: List, model_path: str, temperature: float) -> List[str]:
    # set visible gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # get a model on a single gpu
    model = HuggingFaceModelGenerator(model_path, free_gpus=[gpu_id], temperature=temperature)

    responses_list = []
    for prompt in prompts:
        responses = await model.generate(prompt, num_samples=3)
        for response in responses:
            responses_list.append(response)
            print(f"[GPU {gpu_id}] Response: {response}")

    return responses_list


def start_collector(gpu_id: int, prompts: list, model_path: str, temperature: float) -> List[str]:
    # event loop in a process
    results = asyncio.run(run_vllm_collector(gpu_id, prompts, model_path, temperature))
    return results


def main(prompts: list, model_path: str, free_gpus: List[int], temperature: float) -> None:
    num_tot = len(prompts)
    num_gpu = len(free_gpus)
    num_per_gpu = num_tot // num_gpu
    prompts_per_gpu = chunk_list(prompts, num_per_gpu)
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(free_gpus)) as executor:
        futures = []
        for gpu_id, prompts_gpu in zip(free_gpus, prompts_per_gpu):
            futures.append(executor.submit(start_collector, gpu_id, prompts_gpu, model_path, temperature))

        # get all results
        all_results = []
        for future in concurrent.futures.as_completed(futures):
            all_results.extend(future.result())

    # save results
    with open("/mnt/afs/wangqijian/tests/vllm_multi_gpu.txt", "w") as f:
        for response in all_results:
            f.write(f"{response}\n")


if __name__ == "__main__":
    questions = [
        'Please describe the image.', 'Please describe the image.', 'What\'s the text in the image?',
        'What\'s the text in the image?', 'What is in the image?', 'What is in the image?',
        'How many people are in the image?', 'How many people are in the image?',
        'What is the emotion of the main character of the image?',
        'What is the emotion of the main character of the image?', 'How many animals are in the image?',
        'How many animals are in the image?', 'What is the place of the image?', 'What is the place of the image?',
        'What is the peroson doing?', 'What is the peroson doing?'
    ]
    img_names = [
        '/mnt/afs/niuyazhe/data/meme/data/Cimages/Cimages/Cimages/Image_(2127)',
        '/mnt/afs/niuyazhe/data/meme/data/Cimages/Cimages/Cimages/Image_(5394)',
        '/mnt/afs/niuyazhe/data/meme/data/Cimages/Cimages/Cimages/Image_(1160)',
        '/mnt/afs/niuyazhe/data/meme/data/Cimages/Cimages/Cimages/Image_(4956)',
        '/mnt/afs/niuyazhe/data/meme/data/Cimages/Cimages/Cimages/Image_(2212)',
        '/mnt/afs/niuyazhe/data/meme/data/Cimages/Cimages/Cimages/Image_(3387)',
        '/mnt/afs/niuyazhe/data/meme/data/Cimages/Cimages/Cimages/Image_(4086)',
        '/mnt/afs/niuyazhe/data/meme/data/Cimages/Cimages/Cimages/Image_(4384)',
        '/mnt/afs/niuyazhe/data/meme/data/Cimages/Cimages/Cimages/Image_(5000)',
        '/mnt/afs/niuyazhe/data/meme/data/Cimages/Cimages/Cimages/Image_(1237)',
        '/mnt/afs/niuyazhe/data/meme/data/Cimages/Cimages/Cimages/Image_(766)',
        '/mnt/afs/niuyazhe/data/meme/data/Cimages/Cimages/Cimages/Image_(6031)',
        '/mnt/afs/niuyazhe/data/meme/data/Cimages/Cimages/Cimages/Image_(6)',
        '/mnt/afs/niuyazhe/data/meme/data/Cimages/Cimages/Cimages/Image_(2284)',
        '/mnt/afs/niuyazhe/data/meme/data/Cimages/Cimages/Cimages/Image_(4533)',
        '/mnt/afs/niuyazhe/data/meme/data/Cimages/Cimages/Cimages/Image_(5495)'
    ]
    free_gpus = get_free_gpus()
    modality = Modality.IMAGE
    mm_input = get_multi_modal_input(modality, img_names, questions)
    data = mm_input["data"]
    question = mm_input["question"]
    prompts, stop_token_ids = get_prompts_qwen(question, modality)
    model_path = '/mnt/afs/share/Qwen2-VL-7B'
    temperature = 0.5
    main(prompts, model_path, free_gpus, temperature)
