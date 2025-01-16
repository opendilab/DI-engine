from typing import List, Tuple
import os
import uuid
from loguru import logger
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams, RequestOutput
from vllm.assets.image import ImageAsset

class VllmActor:
    def __init__(self, model_path: str) -> None:
        """
        Overview:
            Initialize the vLLM actor. For more details, please refer to https://docs.vllm.ai/en/stable.
        Arguments:
            - model_path (str): The path to the language model.
        """
        self.free_gpus = self.get_free_gpus()
        self.num_gpus = len(self.free_gpus)
        assert self.num_gpus > 0, "No GPUs found"
        # Set CUDA_VISIBLE_DEVICES to use only free GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.free_gpus))
        self.model_path = model_path
        self._initialize()

    def get_free_gpus(self) -> List[int]:
        """
        Overview:
            Get IDs of GPUs with free memory.
        Returns:
            - List[int]: The IDs of the free GPUs.
        """
        try:
            # Get GPU memory usage using nvidia-smi
            gpu_stats = os.popen('nvidia-smi --query-gpu=memory.used,memory.total --format=csv,nounits,noheader').readlines()
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
            #max_model_len=4096,  #see if 8192 works
            #max_num_batched_tokens=4096,
            #max_num_batched_tokens=2048,
            #max_model_len=2048,
            # enable_chunked_prefill=True,
            max_num_seqs=5,
            # Note - mm_processor_kwargs can also be passed to generate/chat calls
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
            },
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

    def __init__(self, model_path: str, max_tokens: int = 1024, temperature: float = 0) -> None:
        """
        Overview:
            Initialize the Hugging Face model generator.
        Arguments:
            - model_path (str): The path to the language model.
            - max_tokens (int): The maximum number of tokens to generate, default to 1024.
            - temperature (float): The temperature for the language model, default to 0.
        """
        self.vllm_actor = VllmActor(model_path)
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
        return [
            (x.text.strip(), conf)
            for x, conf in zip(response.outputs, confidence_scores)
        ]
        

model=HuggingFaceModelGenerator('/mnt/afs/share/Qwen2-VL-7B',temperature=0.5) #设置一个temperature就好了,可以做到生成多个候选答案

def get_prompts_qwen(questions: list,modality: str):
    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"

    prompts = [("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
              f"{question}<|im_end|>\n"
              "<|im_start|>assistant\n") for question in questions]
    stop_token_ids = None
    return prompts,stop_token_ids

def get_multi_modal_input(modality,filenames,questions):
    """
    return {
        "data": image or video,
        "question": question,
    }
    """
    if modality == "image":
        # Input image and question
        ret={}
        ret["data"]=[]
        ret["question"]=[]
        for filename,question in zip(filenames,questions):
            image = ImageAsset(filename) \
                .pil_image.convert("RGB")
            #img_question = "What is the content of this image?"
            img_question=question
            ret["data"].append(image)
            ret["question"].append(img_question)
        return ret


questions=["What is the content of this image?","Please describe the image.","How many people are there in the image? What are they doing?"]
img_names=['/mnt/afs/niuyazhe/data/meme/data/Eimages/Eimages/Eimages/image_ (2)','/mnt/afs/niuyazhe/data/meme/data/Cimages/Cimages/Cimages/Image_(3127)','/mnt/afs/wangqijian/data/test/test']
#questions=["What is the content of this image?"]
#img_names=['/mnt/afs/niuyazhe/data/meme/data/Eimages/Eimages/Eimages/image_ (2)']
num_prompts=len(questions)
image_repeat_prob=None
modality = 'image'

mm_input = get_multi_modal_input(modality,img_names,questions)
data = mm_input["data"]
question = mm_input["question"]
batch_inference_mine=True
prompts, stop_token_ids = get_prompts_qwen(question,modality)


import asyncio
import nest_asyncio
nest_asyncio.apply()
async def main():
    inputs = [
        {
            "prompt":prompt,
            "multi_modal_data":{
                modality:data
            }
        } for prompt,data in zip(prompts,data)
    ]
    # 调用 generate 方法
    for in_data in inputs:
        tactics = await model.generate(prompt=in_data, num_samples=3)
        # 打印返回结果
        for tactic, confidence in tactics:
            print(f"Tactic: {tactic}")
    

# 运行主程序
if __name__ == "__main__":
    asyncio.run(main())