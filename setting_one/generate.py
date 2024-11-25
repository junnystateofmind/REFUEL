from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import argparse
import os
import pickle
import random
import time
import subprocess
import numpy as np


def set_seed(seed=5775709):
    random.seed(seed)
    np.random.seed(seed)

# SODA 데이터셋에 맞게 형식 변경
def change_format(row):
    new_trajectory = []
    if "narrative" in row and row["narrative"]:  # narrative 추가
        new_trajectory.append({
            "role": "system",
            "content": row["narrative"]
        })
    for i, (speaker, utterance) in enumerate(zip(row["speakers"], row["dialogue"])):
        role = "user" if i % 2 == 0 else "assistant"  # 역할을 user/assistant로 고정
        new_trajectory.append({'role': role, 'content': utterance})
    row["trajectory"] = new_trajectory
    return row

def restructure_data_for_generator(data):
    """
    SODA 데이터셋의 list 또는 Dataset 객체를 list[dict] 형태로 변환.
    """
    structured_data = []
    for entry in data:
        # 각 entry가 dict인지 확인
        if isinstance(entry, dict) and "trajectory" in entry:
            # 이미 `trajectory` 필드가 있다면 그대로 추가
            structured_data.append({
                "trajectory": entry["trajectory"],
                "narrative": entry.get("narrative", "Default narrative")  # narrative 추가
            })
        else:
            print("[DEBUG] Unexpected entry format:", entry)
    
    # 변환된 데이터 디버깅 출력
    if structured_data:
        print("[DEBUG] First entry in structured_data:", structured_data[0])
    else:
        print("[DEBUG] No valid entries found in dataset.")
    
    return structured_data


# 중복 제거 함수
unique_traj = {}
def check_redundant(tokenizer, row):
    try:
        dialogue_text = " ".join([f"{msg['role']}: {msg['content']}" for msg in row["trajectory"]])
        tokenized = tokenizer(dialogue_text, truncation=True, padding=True, return_tensors="pt").input_ids.tolist()
        tokenized = tuple(tokenized[0])
        if tokenized in unique_traj:
            return False
        unique_traj[tokenized] = 1
        return True
    except Exception as e:
        print("[DEBUG] Error in check_redundant with row:", row)
        raise e

def call_scripts(args, seed, gen_type):
    if gen_type == 'response':
        try:
            subprocess.run(['python', './setting_one/response_generator.py', \
                            '--dataset', f'{os.path.join(args.output_dir, "temp.pkl")}', \
                            '--temperature', f'{args.temperature}', \
                            '--maxlen', f'{args.maxlen}', \
                            '--world_size', f'{args.world_size}', \
                            '--model', f'{args.model}', \
                            '--seed', f'{seed}', \
                            '--num_turns', f'{args.num_turns}', \
                            '--dtype', f'{args.dtype}'], check=True)
        except:
            return False
    else:
        try:
            subprocess.run(['python', './setting_one/user_generator.py', \
                            '--dataset', f'{os.path.join(args.output_dir, "temp.pkl")}', \
                            '--temperature', f'{args.temperature}', \
                            '--maxlen', f'{args.maxlen}', \
                            '--world_size', f'{args.world_size}', \
                            '--model', f'{args.user_model}', \
                            '--seed', f'{seed}', \
                            '--num_turns', f'{args.num_turns}', \
                            '--dtype', f'{args.dtype}'], check=True)
        except:
            return False
    return True

def call_scripts_wrapper(args, seed, gen_type):
    while not call_scripts(args, seed, gen_type):
        time.sleep(20)
        print(f'error when generating {gen_type}')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--user_model", type=str, default="meta-llama/Meta-Llama-3.1-70B-Instruct")

    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--output_repo", type=str, default="")

    parser.add_argument("--dataset", type=str, default="allenai/soda")
    parser.add_argument("--dataset_split", type=str, default="train")

    parser.add_argument("--num_turns", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--maxlen", type=int, default=1024)
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])

    parser.add_argument("--num_data", type=int, default=0)
    return parser.parse_args()

if __name__ == "__main__":

    # 초기화
    args = parse_arguments()
    set_seed(args.seed)
    dataset = load_dataset(args.dataset, split=args.dataset_split)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # 전처리
    if args.num_data != 0:
        dataset = dataset.select(range(args.num_data))
    dataset = dataset.map(change_format)

    # list[list] 구조를 list[dict] 구조로 변환
    dataset = restructure_data_for_generator(dataset)

    # 중복 제거
    dataset = [row for row in dataset if check_redundant(tokenizer, row)]

    # 초기 프롬프트 저장
    trajectory = []
    for entry in dataset:
        if entry['trajectory']:
            trajectory.append({
                "trajectory": [entry['trajectory'][0]],  # 첫 발화만 저장
                "narrative": entry.get("narrative", "Default narrative")  # narrative 추가
            })
    with open(os.path.join(args.output_dir, 'temp.pkl'), 'wb') as handle:
        pickle.dump(trajectory, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Initial prompt saved to {os.path.join(args.output_dir, "temp.pkl")}')

    # 생성 루프
    # 생성 루프에서 중간 저장 시 trajectory만 업로드
    for turn in range(args.num_turns):
        call_scripts_wrapper(args, args.seed, gen_type='response')
    
        # 중간 결과 저장
        with open(os.path.join(args.output_dir, 'temp.pkl'), 'rb') as handle:
            trajectory = pickle.load(handle)
        
        # trajectory 데이터만 필터링
        temp_dataset = Dataset.from_dict({"trajectory": [entry['trajectory'] for entry in trajectory]})
        temp_dataset.push_to_hub(args.output_repo + f'_{args.num_turns}_turns_ckp_{turn}')
    
        if turn < args.num_turns - 1:
            call_scripts_wrapper(args, args.seed, gen_type='user')
    
    # 최종 저장 시 trajectory만 업로드
    with open(os.path.join(args.output_dir, 'temp.pkl'), 'rb') as handle:
        trajectory = pickle.load(handle)
    
    # trajectory 데이터만 필터링
    generated = Dataset.from_dict({"trajectory": [entry['trajectory'] for entry in trajectory]})
    generated.push_to_hub(args.output_repo)

