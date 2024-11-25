from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import argparse
import os
import pickle
import random
import numpy as np
import torch


def set_seed(seed=5775709):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_chat_template(trajectory, narrative, add_generation_prompt=False):
    prompt = f"### Narrative:\n{narrative}\n\n"
    for turn in trajectory:
        role = turn.get('role', 'user')  # Default role is 'user'
        content = turn.get('content', '')
        # 명확한 역할 구분을 위해 Role 추가
        prompt += f"{role.capitalize()}:\n{content}\n\n"
    if add_generation_prompt:
        prompt += "User:\n"  # 항상 user의 역할을 유도
    return prompt



def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature for LLM")
    parser.add_argument("--maxlen", type=int, default=1024, help="Maximum token length for generation")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--world_size", type=int, default=4, help="Number of GPUs for model parallelism")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset (Pickle file)")
    parser.add_argument("--model", type=str, required=True, help="Pretrained model name or path")
    parser.add_argument("--num_turns", type=int, default=5, help="Number of conversation turns to include in the prompt")
    parser.add_argument("--dtype", type=str, default="float32", help="Data type for the LLM")
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments and set random seed
    args = parse_arguments()
    set_seed(args.seed)

    # Initialize tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.world_size,
            dtype=args.dtype,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize model or tokenizer: {e}")

    # Load dataset
    try:
        with open(args.dataset, 'rb') as handle:
            combined_data = pickle.load(handle)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")

    # Validate dataset format
    if not all(isinstance(entry, dict) for entry in combined_data):
        # check combined_data's type
        print("combined_data's type: ", type(combined_data))
        print("entry's type: ", type(combined_data[0]))
        raise ValueError("Dataset must be a list of dictionaries.")

    # Prepare prompts
    prompts = []
    prompt_idx_to_data_idx = {}
    for i, entry in enumerate(combined_data):
        trajectory = entry.get('trajectory', [])
        narrative = entry.get('narrative', "Default narrative")
        if len(trajectory) >= args.num_turns * 2:
            continue
        prompt = apply_chat_template(trajectory, narrative, add_generation_prompt=True)
        prompts.append(prompt)
        prompt_idx_to_data_idx[len(prompts) - 1] = i

    # Generate responses
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.maxlen,
        seed=args.seed,
    )
    try:
        responses = llm.generate(prompts, sampling_params)
    except Exception as e:
        raise RuntimeError(f"Failed to generate responses: {e}")

    # Update dataset with generated responses
    outputs = [resp.outputs[0].text.strip() for resp in responses]
for idx, output in enumerate(outputs):
    data_idx = prompt_idx_to_data_idx[idx]
    # role을 명시적으로 'assistant'로 고정
    combined_data[data_idx]['trajectory'].append({"role": "assistant", "content": output})


    # Save updated dataset
    try:
        with open(args.dataset, 'wb') as handle:
            pickle.dump(combined_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Responses successfully generated and saved to {args.dataset}")
    except Exception as e:
        raise RuntimeError(f"Failed to save dataset: {e}")
