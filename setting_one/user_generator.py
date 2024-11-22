# user_generator.py (수정된 코드)

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
    parser = argparse.ArgumentParser(description="User Generator for REFUEL")
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
    prompt += 'Below is a dialogue among multiple speakers. Pretend you are the user in this conversation. What question would you ask next?\n\n'
    for turn in trajectory:
        prompt += '### ' + turn['role'].capitalize()
        prompt += ': '
        prompt += turn['content']
        prompt += '\n\n'
    prompt += '### Instructions:\nFIRST provide a justification of the question you want to ask.\nSECOND, on a new line, state only the question. Your response should use the format:\nJustification:\nQuestion:'
    return [{"role": "user", "content": prompt}]


if __name__ == "__main__":

    # Initialize arguments and components
    args = parse_arguments()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.world_size,
        dtype=args.dtype,
    )

    set_seed(args.seed)

    # Load dataset
    with open(args.dataset, 'rb') as handle:
        combined_data = pickle.load(handle)

    prompts, prompt_i_to_traj_i = [], {}
    for i, entry in enumerate(combined_data):
        trajectory = entry['trajectory']
        narrative = entry['narrative']
        if len(trajectory) < args.num_turns * 2:
            prompts.append(get_prompt(trajectory, narrative))
            prompt_i_to_traj_i[len(prompts) - 1] = i

    # Apply chat template (assuming apply_chat_template is correctly defined)
    prompts = [tokenizer.apply_chat_template(t, tokenize=False, add_generation_prompt=True) for t in prompts]

    # Generate responses
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.maxlen,
        seed=args.seed
    )
    response = llm.generate(prompts, sampling_params)
    output = [x.outputs[0].text for x in response]

    # Merge generated questions into trajectory
    for r in range(len(output)):
        try:
            # Extract the generated question after 'Question:'
            question = output[r].rsplit('Question:', 1)[1].strip()
            combined_data[prompt_i_to_traj_i[r]]['trajectory'].append({"role": "user", "content": question})
        except IndexError:
            # If 'Question:' is not found, append the entire output
            print(prompt_i_to_traj_i[r], 'added all outputs')
            combined_data[prompt_i_to_traj_i[r]]['trajectory'].append({"role": "user", "content": output[r].strip()})

    # Save the updated dataset
    with open(args.dataset, 'wb') as handle:
        pickle.dump(combined_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Questions generated and saved to {args.dataset}')
