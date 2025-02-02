from typing import List, Tuple, Optional
import os
import uuid
from loguru import logger
from ..vllm_collector import HuggingFaceModelGenerator
from vllm.assets.image import ImageAsset
from enum import Enum
import asyncio
import nest_asyncio
# set a temperature > 0 to get multiple responses
# note that HFModelGenerator has a parameter "mm_processor_kwargs" set to align with the settings of Qwen in default
model = HuggingFaceModelGenerator('/mnt/afs/share/Qwen2-VL-7B', temperature=0.5)


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


questions = [
    "What is the content of this image?", "Please describe the image.",
    "How many people are there in the image? What are they doing?"
]
img_names = [
    '/mnt/afs/niuyazhe/data/meme/data/Eimages/Eimages/Eimages/image_ (2)',
    '/mnt/afs/niuyazhe/data/meme/data/Cimages/Cimages/Cimages/Image_(3127)', '/mnt/afs/wangqijian/data/test/test'
]

num_prompts = len(questions)
image_repeat_prob = None

modality = Modality.IMAGE

mm_input = get_multi_modal_input(modality, img_names, questions)
data = mm_input["data"]
question = mm_input["question"]
prompts, stop_token_ids = get_prompts_qwen(question, modality)

nest_asyncio.apply()


async def main():
    inputs = [{"prompt": prompt, "multi_modal_data": {modality.value: data}} for prompt, data in zip(prompts, data)]
    # generate responses
    for in_data in inputs:
        responses = await model.generate(prompt=in_data, num_samples=3)
        # print response
        for response, confidence in responses:
            print(f"Response: {response}")


# run main
if __name__ == "__main__":
    asyncio.run(main())
