from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import argparse
import os
import pickle
import random
import numpy as np
import torch


def set_seed(seed=5775709):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="User Generator")
    parser.add_argument("--temperature", type=float, default=0.01, help="Sampling temperature")
    parser.add_argument("--maxlen", type=int, default=1024, help="Maximum length of generated tokens")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--world_size", type=int, default=4, help="Number of parallel processes")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset pickle file")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--num_turns", type=int, default=5, help="Number of turns to generate")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"], help="Data type for model")
    return parser.parse_args()


def get_prompt(trajectory, narrative):
    prompt = f'### Narrative:\n{narrative}\n\n'
    prompt += 'Below is a dialogue among multiple speakers. Pretend you are the user in this conversation. What would you say next?\n\n'
    for turn in trajectory:
        role = turn['role']  # 'user' 또는 'assistant'
        content = turn['content']
        prompt += f"{role.capitalize()}:\n{content}\n\n"
    prompt += 'Assistant:\n'  # 다음 응답은 항상 user로 가정
    return prompt



if __name__ == "__main__":
    # Initialize arguments and components
    args = parse_arguments()
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.world_size,
        dtype=args.dtype,
    )

    # Load dataset
    with open(args.dataset, 'rb') as handle:
        combined_data = pickle.load(handle)

    prompts = []
    prompt_i_to_data_i = {}
    for i, entry in enumerate(combined_data):
        trajectory = entry['trajectory']
        narrative = entry['narrative']
        if len(trajectory) < args.num_turns * 2:
            prompt = get_prompt(trajectory, narrative)
            prompts.append(prompt)
            prompt_i_to_data_i[len(prompts) - 1] = i

    # Generate responses
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.maxlen,
        seed=args.seed
    )
    responses = llm.generate(prompts, sampling_params)
    outputs = [resp.outputs[0].text for resp in responses]

    # Merge generated messages into trajectory
    for idx, output in enumerate(outputs):
        data_idx = prompt_i_to_data_i[idx]
        # 출력 내용을 바로 추가
        combined_data[data_idx]['trajectory'].append({"role": "user", "content": output.strip()})
        print(f"Data index {data_idx}: Added output as user: {output.strip()}")


    # Save the updated dataset
    with open(args.dataset, 'wb') as handle:
        pickle.dump(combined_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Messages generated and saved to {args.dataset}')

