import random
from enum import Enum
from typing import List, Dict, Any, Union
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import numpy as np

from clivis import preference
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,  # 新增
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from qwen_vl_utils import process_vision_info
from PIL import Image
from clivis.models.llm import basic_llm_chat, count_tokens

"""
Unified chat interface for multiple model types.
"""

TOTAL_TOKENS = 0  # Global token counter

__all__ = [
    "VlmModelType",
    "basic_llm_chat",
    "basic_vlm_chat",
    "init_vlm_model",
    "set_gpu",
]

# Print available GPU info
print("Available GPU count:", torch.cuda.device_count())
print("Available GPU names:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

GPU_ID = 0
DEVICE = f"cuda:{GPU_ID}"

def set_gpu(gpu_id: int):
    global GPU_ID, DEVICE
    GPU_ID = int(gpu_id)
    DEVICE = f"cuda:{GPU_ID}"
    print(f"Switched to GPU: {DEVICE}")

# Define VLM model enum
class VlmModelType(Enum):
    QWEN = "qwen"
    QWEN3 = "qwen3"  # Added
    VIDEO_LLAMA = "video-llama"
    INTERNVL = "internvl"


# Constant settings
min_pixels = 128 * 28 * 28
max_pixels = 1024 * 28 * 28
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Global model and processor variables
CURRENT_MODEL = None
CURRENT_PROCESSOR = None
CURRENT_MODEL_TYPE = None


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def init_vlm_model(model_type: Union[str, VlmModelType] = VlmModelType.QWEN):
    """
    Initialize the model at program start.

    Args:
        model_type: Model type as VlmModelType enum or string.

    Returns:
        None (sets global variables)
    """
    global CURRENT_MODEL, CURRENT_PROCESSOR, CURRENT_MODEL_TYPE

    # Convert string to enum
    if isinstance(model_type, str):
        model_type = VlmModelType(model_type.lower())

    # Return early if the requested model is already loaded
    if CURRENT_MODEL_TYPE == model_type and CURRENT_MODEL is not None and CURRENT_PROCESSOR is not None:
        print(f"Model {model_type.value} is already loaded")
        return

    print(f"Loading {model_type.value} model...")

    # Release previous model memory (if any)
    if CURRENT_MODEL is not None:
        del CURRENT_MODEL
        del CURRENT_PROCESSOR
        torch.cuda.empty_cache()

    # Load model based on model type
    if model_type == VlmModelType.QWEN:
        model_name = preference.VLM_MODEL_NAME_QWEN25VL
        CURRENT_MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            load_in_4bit=False,
            device_map=DEVICE,
            low_cpu_mem_usage=True,
            force_download=False,
        )
        CURRENT_PROCESSOR = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            use_fast=True
        )
    elif model_type == VlmModelType.QWEN3:  # Added branch
        model_name = preference.VLM_MODEL_NAME_QWEN3VL  # Defined in preference.py
        CURRENT_MODEL = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=DEVICE
        )
        CURRENT_PROCESSOR = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            use_fast=True
        )
    elif model_type == VlmModelType.VIDEO_LLAMA:
        model_name = preference.VLM_MODEL_NAME_VIDEOLLAMA3
        CURRENT_MODEL = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            load_in_4bit=True,
            device_map=DEVICE,
        )
        CURRENT_PROCESSOR = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            use_fast=True
        )
    elif model_type == VlmModelType.INTERNVL:
        model_name = preference.VLM_MODEL_NAME_INTERNVL3
        device_map = DEVICE
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        CURRENT_MODEL = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map
        ).eval()

        CURRENT_PROCESSOR = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    CURRENT_MODEL_TYPE = model_type
    print(f"{model_type.value} model loaded")


# Video processing helpers
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices


def load_video_for_internvl(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    """Load video and prepare inputs for InternVL."""
    try:
        from decord import VideoReader, cpu
        import numpy as np
    except ImportError:
        print("Please install decord: pip install decord")
        return None, None

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

def convert_messages_for_videollama(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert Qwen VL messages to VideoLLaMA format.
    Note: The converted messages are passed directly to the VideoLLaMA processor.
    """
    converted_messages = []

    for msg in messages:
        role = msg["role"]
        new_msg = {"role": role}

        if isinstance(msg.get("content"), str):
            new_msg["content"] = msg["content"]
        elif isinstance(msg.get("content"), list):
            new_content = []
            for item in msg["content"]:
                if item["type"] == "text":
                    new_content.append({"type": "text", "text": item["text"]})
                elif item["type"] == "video":
                    video_path = item["video"]
                    fps = item.get("fps", 1)
                    video_item = {
                        "type": "video",
                        "video": {
                            "video_path": video_path,
                            "fps": fps,
                            "max_frames": 128,  # VideoLLaMA3 default
                            "size": 360,
                        }
                    }
                    new_content.append(video_item)
                elif item["type"] == "image":
                    new_content.append({"type": "image", "image": item["image"]})
            new_msg["content"] = new_content

        converted_messages.append(new_msg)

    return converted_messages

def prepare_internvl_video_input(video_path, num_segments=32, max_num=1):
    """Prepare InternVL video input."""
    pixel_values, num_patches_list = load_video_for_internvl(video_path, num_segments=num_segments, max_num=max_num)
    if pixel_values is not None:
        pixel_values = pixel_values.to(torch.bfloat16).to(DEVICE)
    return pixel_values, num_patches_list


def extract_video_paths_from_messages(messages):
    """Extract video paths from messages."""
    video_paths = []
    for message in messages:
        if isinstance(message.get("content"), list):
            for item in message["content"]:
                if item.get("type") == "video" and item.get("video"):
                    if isinstance(item["video"], str):
                        video_paths.append(item["video"])
                    elif isinstance(item["video"], dict) and "video_path" in item["video"]:
                        video_paths.append(item["video"]["video_path"])
    return video_paths


def extract_image_paths_from_messages(messages):
    """Extract image paths from messages."""
    image_paths = []
    for message in messages:
        if isinstance(message.get("content"), list):
            for item in message["content"]:
                if item.get("type") == "image" and item.get("image"):
                    image_paths.append(item["image"])
    return image_paths


def get_internvl_prompt_from_messages(messages):
    """Build InternVL prompt from messages."""
    system_prompt = ""
    user_prompt = ""
    assistant_history = []

    for message in messages:
        role = message["role"]
        if role == "system":
            if isinstance(message.get("content"), str):
                system_prompt += message["content"] + "\n"
        elif role == "user":
            if isinstance(message.get("content"), str):
                user_prompt = message["content"]
            elif isinstance(message.get("content"), list):
                text_parts = []
                for item in message["content"]:
                    if item["type"] == "text":
                        text_parts.append(item["text"])
                user_prompt = " ".join(text_parts)
        elif role == "assistant" and len(assistant_history) > 0:
            if isinstance(message.get("content"), str):
                assistant_history.append({"user": "", "assistant": message["content"]})

    return system_prompt, user_prompt, assistant_history


def basic_vlm_chat(
        messages,
        temperature=0.3,
        top_p=0.95,
        top_k=100,
        max_tokens=4096,
        fps=0.5
):
    """
    Unified VLM chat function.
    """
    global CURRENT_MODEL, CURRENT_PROCESSOR, CURRENT_MODEL_TYPE, TOTAL_TOKENS
    # Clear GPU cache
    torch.cuda.empty_cache()

    try:
        # Ensure the model is initialized
        if CURRENT_MODEL is None or CURRENT_PROCESSOR is None:
            print("Error: model is not initialized. Call init_vlm_model() first.")
            return "Model is not initialized. Call init_vlm_model() first."

        # Validate message format
        if not messages or not isinstance(messages, list):
            print("Error: invalid message format")
            return None

        # Handle messages and inputs by model type
        if CURRENT_MODEL_TYPE == VlmModelType.INTERNVL:
            # InternVL handling
            generation_config = {
                "max_new_tokens": max_tokens,
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": 1.05,
            }

            # Extract prompt info
            _, user_prompt, history = get_internvl_prompt_from_messages(messages)

            # Check for video inputs
            video_paths = extract_video_paths_from_messages(messages)
            if video_paths:
                # Handle video input
                video_path = video_paths[0]  # Only handle the first video for now
                pixel_values, num_patches_list = prepare_internvl_video_input(video_path)

                if pixel_values is None:
                    return "Unable to process the video file. Ensure the format is correct."

                # Build video frame prefix
                video_prefix = ''.join([f'Frame{i + 1}: <image>\n' for i in range(len(num_patches_list))])
                question = video_prefix + user_prompt

                # Run model with video inputs
                response = CURRENT_MODEL.chat(
                    CURRENT_PROCESSOR,
                    pixel_values,
                    question,
                    generation_config,
                    num_patches_list=num_patches_list,
                    history=history if history else None
                )

                # Count tokens
                token_count = count_tokens(str(response))
                TOTAL_TOKENS += token_count

                return response

            # Check for image inputs
            image_paths = extract_image_paths_from_messages(messages)
            if image_paths:
                # Handle single or multiple images
                if len(image_paths) == 1:
                    # Single image
                    pixel_values = load_image(image_paths[0], max_num=12).to(torch.bfloat16).to(DEVICE)
                    question = '<image>\n' + user_prompt
                    response = CURRENT_MODEL.chat(
                        CURRENT_PROCESSOR,
                        pixel_values,
                        question,
                        generation_config,
                        history=history if history else None
                    )
                else:
                    # Multiple images (up to 2)
                    pixel_values1 = load_image(image_paths[0], max_num=12).to(torch.bfloat16).to(DEVICE)
                    pixel_values2 = load_image(image_paths[1], max_num=12).to(torch.bfloat16).to(DEVICE)
                    pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
                    num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

                    question = 'Image-1: <image>\nImage-2: <image>\n' + user_prompt
                    response = CURRENT_MODEL.chat(
                        CURRENT_PROCESSOR,
                        pixel_values,
                        question,
                        generation_config,
                        num_patches_list=num_patches_list,
                        history=history if history else None
                    )

                token_count = count_tokens(str(response))
                TOTAL_TOKENS += token_count

                return response

            # Text-only chat
            response = CURRENT_MODEL.chat(
                CURRENT_PROCESSOR,
                None,
                user_prompt,
                generation_config,
                history=history if history else None
            )

            token_count = count_tokens(str(response))
            TOTAL_TOKENS += token_count
            print(f"Total tokens so far: {TOTAL_TOKENS}")

            return response

        elif CURRENT_MODEL_TYPE == VlmModelType.VIDEO_LLAMA:
            # VideoLLaMA3 flow
            vlm_messages = convert_messages_for_videollama(messages)

            # VideoLLaMA3 uses the conversation parameter
            inputs = CURRENT_PROCESSOR(conversation=vlm_messages, return_tensors="pt")

            # Move tensors to GPU and convert format
            inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        elif CURRENT_MODEL_TYPE == VlmModelType.QWEN3:
            # Qwen3 input construction
            text = CURRENT_PROCESSOR.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            try:
                images, videos, video_kwargs = process_vision_info(
                    messages,
                    image_patch_size=16,
                    return_video_kwargs=True,
                    return_video_metadata=True
                )

                # Unpack (video, metadata) pairs
                if videos is not None:
                    videos, video_metadatas = zip(*videos)
                    videos, video_metadatas = list(videos), list(video_metadatas)
                else:
                    video_metadatas = None
            except Exception as e:
                print(f"Error processing vision info: {e}")
                return f"Error processing vision inputs: {str(e)}"

            inputs = CURRENT_PROCESSOR(
                text=text,
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                return_tensors="pt",
                do_resize=False,
                **video_kwargs
            )

            # Move inputs to the actual model device
            model_device = next(CURRENT_MODEL.parameters()).device
            inputs = inputs.to(model_device)

        else:
            # Qwen VL flow
            # Build text template
            text = CURRENT_PROCESSOR.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Process vision inputs
            try:
                image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

                # Validate input validity
                if (video_inputs and len(video_inputs) > 0 and
                        all(v is None or (isinstance(v, torch.Tensor) and v.numel() == 0) for v in video_inputs)):
                    print("Warning: all video inputs are invalid")
                    return "Unable to process video content. Ensure the file exists and format is correct."

                if (image_inputs and len(image_inputs) > 0 and
                        all(img is None or (isinstance(img, torch.Tensor) and img.numel() == 0) for img in
                            image_inputs)):
                    print("Warning: all image inputs are invalid")
                    return "Unable to process image content. Ensure the file exists and format is correct."
            except Exception as e:
                print(f"Error processing vision info: {e}")
                return f"Error processing vision inputs: {str(e)}"

            # Build inputs
            inputs = CURRENT_PROCESSOR(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs
            )

            # Check input validity
            if 'input_ids' not in inputs or inputs['input_ids'].numel() == 0:
                print("Error: generated input is empty")
                return "Unable to process input. Provide valid text and media."

            # Move to GPU
            inputs = inputs.to(DEVICE)

        # Generation flow for non-InternVL models
        if CURRENT_MODEL_TYPE != VlmModelType.INTERNVL:
            # Set random seed
            random_seed = random.randint(0, 100000)
            torch.manual_seed(random_seed)

            # Generate response
            with torch.no_grad():
                generated_ids = CURRENT_MODEL.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,

                )

            # Decode text - VideoLLaMA and Qwen differ
            if CURRENT_MODEL_TYPE == VlmModelType.VIDEO_LLAMA:
                output_text = CURRENT_PROCESSOR.batch_decode(generated_ids, skip_special_tokens=True)
            else:
                # Trim generated IDs for Qwen
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = CURRENT_PROCESSOR.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

            result = output_text[0].strip() if output_text else None
                # Count tokens
            token_count = count_tokens(str(result))
            TOTAL_TOKENS += token_count
            return result

    except Exception as e:
            print(f"Error during VLM processing: {e}")
            return f"Error processing request: {str(e)}"

        # Example: initialize the model at startup
init_vlm_model(VlmModelType.INTERNVL)





