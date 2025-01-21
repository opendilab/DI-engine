from typing import List, Tuple
import os
import uuid
from loguru import logger
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams, RequestOutput


class VllmActor:

    def __init__(self, model_path: str, mm_processor_kwargs: dict) -> None:
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
        self.mm_processor_kwargs = mm_processor_kwargs
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
        self.vllm_actor = VllmActor(model_path, mm_processor_kwargs)
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
