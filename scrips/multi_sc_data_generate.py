from reward.reward import result_stats
from argparse import ArgumentParser
from utils.config import llama3_path_a100, llama3_path_a800
from transformers import AutoTokenizer
from train.inference import inference
from dataloader.dataloader import (
    DataloaderForHotpotQA,
    DataloaderForMWHQA,
    DataloaderForCBT,
    DataloaderForGSM8K,
    DataloaderForMATH,
    DataloaderForTrivalQA,
    DataloaderForARC,
    DataloaderForMMLU,
)
import time
import os
import subprocess
import requests

argumentParser = ArgumentParser()
argumentParser.add_argument("--model_path", type=str, required=True)
argumentParser.add_argument("--tokenizer_path", type=str, default=llama3_path_a800)
argumentParser.add_argument("--device", type=int, default=0)
argumentParser.add_argument("--port", type=int, default=8000)
argumentParser.add_argument("--dataset_type", type=str, default="hotpot_qa")
argumentParser.add_argument("--num_thread", type=int, default=1024)
argumentParser.add_argument("--output_path", type=str, required=True)
argumentParser.add_argument("--temperature", type=float, default=0.5)
args = argumentParser.parse_args()

if __name__ == "__main__":
    loader = None
    if args.dataset_type == "hotpot_qa":
        print("hotpot")
        loader = DataloaderForHotpotQA(split="validation")
    elif args.dataset_type == "mwh_qa":
        print("mwh")
        loader = DataloaderForMWHQA(split="dev")
    elif args.dataset_type == "cbt":
        print("cbt")
        loader = DataloaderForCBT(split="test")
    elif args.dataset_type == "gsm8k":
        print("gsm8k")
        loader = DataloaderForGSM8K(split="test")
    elif args.dataset_type == "math":
        print("math")
        loader = DataloaderForMATH(split="test")
    elif args.dataset_type == "trival_qa":
        print("trival_qa")
        loader = DataloaderForTrivalQA(split="validation")
    elif args.dataset_type == "arc":
        print("arc")
        loader = DataloaderForARC(split="test")
    elif args.dataset_type == "mmlu":
        print("mmlu")
        loader = DataloaderForMMLU(split="test")
    ports = []
    for i in range(8):
        process = subprocess.Popen(
            f"""
        source ~/.bashrc && \
        conda activate vllm-0.6.2 && \
        CUDA_VISIBLE_DEVICES={args.device+i} vllm serve {args.model_path} --host 0.0.0.0 --port {args.port+i} --served-model-name "Llama-3" --enable-prefix-caching
        """,
            shell=True,
        )
        ports.append(args.port + i)
    while True:
        try:
            message_input = [{"role": "assistant", "content": "hello!"}]
            headers = {"Content-Type": "application/json"}
            data_json = {
                "model": "Llama-3",
                "messages": message_input,
            }
            response = requests.post(
                f"http://0.0.0.0:{args.port}/v1/chat/completions",
                headers=headers,
                json=data_json,
            )
            content = (response.json()["choices"][0]["message"]["content"],)
            print(f"ready to generate data: {content}")
            break
        except:
            continue
    time.sleep(20)
    loader.current_task_id = 0
    inference(
        "Llama-3",
        "Llama-3",
        f"http://0.0.0.0:{args.port}/v1/chat/completions",
        f"http://0.0.0.0:{args.port}/v1/chat/completions",
        args.tokenizer_path,
        args.tokenizer_path,
        sample_count=1000,
        explore_count=50,
        output_path=args.output_path,
        thread_count=args.num_thread,
        prompt_pool_path="",
        train_data_path="",
        dataloader=loader,
        temperature=args.temperature,
        no_use_prompt_pool=True,
        ports=ports,
    )
    for i in range(8):
        os.system(
            f"""pkill -f "vllm serve {args.model_path} --host 0.0.0.0 --port {args.port+i} --served-model-name Llama-3 --enable-prefix-caching" """
        )
    time.sleep(5)
