from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature for generation.")
    parser.add_argument("--maxlen", type=int, default=1024, help="Maximum length of generated tokens.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--world_size", type=int, default=4, help="Tensor parallel size for the model.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset file or dataset type (e.g., 'allenai/soda').")
    parser.add_argument("--model", type=str, required=True, help="Model name or path.")
    parser.add_argument("--num_turns", type=int, default=5, help="Number of turns to use for context.")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"], help="Precision for model inference.")
    return parser.parse_args()


def preprocess_soda_to_trajectory(soda_data):
    """
    Convert SODA dataset to trajectory format.
    Args:
        soda_data (list): List of dialogue entries from SODA dataset.
    Returns:
        trajectory (list): List of trajectories, each containing user and assistant turns.
    """
    trajectory = []
    for item in soda_data:
        turns = item['trajectory']
        formatted_turns = []
        for turn in turns:
            formatted_turns.append({"role": turn['from'], "content": turn['value']})
        trajectory.append(formatted_turns)
    return trajectory


def refine_soda_prompts(trajectory, num_turns):
    """
    Refine SODA trajectories to create prompts for model generation.
    Args:
        trajectory (list): List of processed trajectories.
        num_turns (int): Number of context turns to use.
    Returns:
        prompts (list): List of refined prompts for generation.
        prompt_i_to_traj_i (dict): Mapping of prompt indices to trajectory indices.
    """
    prompts, prompt_i_to_traj_i = [], {}
    for i, t in enumerate(trajectory):
        if len(t) < num_turns * 2:  # Ensure enough turns exist
            dialogue_history = " ".join(
                [turn["content"] for turn in t if turn["role"] == "user"]
            )
            assistant_turns = [turn["content"] for turn in t if turn["role"] == "assistant"]
            prompts.append(f"{dialogue_history} {assistant_turns[-1] if assistant_turns else ''}")
            prompt_i_to_traj_i[len(prompts) - 1] = i
    return prompts, prompt_i_to_traj_i


if __name__ == "__main__":

    # Initialize arguments and model
    args = parse_arguments()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.world_size,
        dtype=args.dtype,  # Use the dtype argument
    )

    # Set random seed
    set_seed(args.seed)

    # Load dataset and preprocess
    with open(args.dataset, 'rb') as handle:
        soda_data = pickle.load(handle)
    trajectory = preprocess_soda_to_trajectory(soda_data)
    print("Dataset preprocessed into trajectory format.")

    # Refine prompts
    prompts, prompt_i_to_traj_i = refine_soda_prompts(trajectory, args.num_turns)

    # Generate responses
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.maxlen,
        seed=args.seed,
    )
    response = llm.generate(prompts, sampling_params)
    output = list(map(lambda x: x.outputs[0].text, response))

    # Merge generated responses into trajectory
    for r in range(len(output)):
        trajectory[prompt_i_to_traj_i[r]].append({"role": "assistant", "content": output[r]})

    # Save updated trajectory back to file
    save_path = args.dataset if args.dataset != "allenai/soda" else "updated_soda.pkl"
    with open(save_path, 'wb') as handle:
        pickle.dump(trajectory, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Responses saved to {save_path}")
