# from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from train.datagenerate import vllm_data_generate
from train.monte_carlo_deploy import monte_carlo_data_generate_deploy
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
from datasets import load_from_disk
import os
import yaml
import subprocess
import requests
import torch
import json
import numpy as np
import time


def sft_dpo_train_v2(
    origin_sft_yaml_path: str,
    origin_dpo_yaml_path: str,
    initial_model_path: str,
    initial_dataset_path: str,
    dataset_type: str,
    mid_yaml_root_path: str,
    mid_sft_jsonl_root_path: str,
    mid_dpo_jsonl_root_path: str,
    mid_sft_dataset_root_path: str,
    mid_dpo_dataset_root_path: str,
    check_point_root_path: str,
    initial_episilon: float,
    initial_dpo_min_value: float,
    initial_dpo_episilon: float,
    iteration_times: int,
    port: int,
    devices: str,
    tokenizer_first_path: str,
    tokenizer_second_path: str,
    sample_count: int,
    monte_sample_count: int,
    explore_count: int,
    thread_count: int,
    prompt_pool_path: str,
    skipping: int,
    cal_ppl: int = 1,
    lambda1: float = -0.6,
    lambda2: float = 1,
    from_initial: bool = False,
    vllm_env:str = "",
    alignment_env:str = ""
):
    model_path = initial_model_path
    episilon = initial_episilon
    dpo_episilon = initial_dpo_episilon
    dpo_min_value = initial_dpo_min_value
    loader = None
    score_type = "f1-score"
    no_use_prompt_pool = False
    is_math = False
    if dataset_type == "hotpot_qa":
        loader = DataloaderForHotpotQA(split="train")
    elif dataset_type == "mwh_qa":
        loader = DataloaderForMWHQA(split="train")
    elif dataset_type == "cbt":
        print("cbt")
        loader = DataloaderForCBT(split="train")
    elif dataset_type == "gsm8k":
        print("gsm8k")
        is_math = True
        score_type = "exact-match"
        loader = DataloaderForGSM8K(split="train")
    elif dataset_type == "math":
        is_math = True
        print("math")
        loader = DataloaderForMATH(split="train")
        score_type = "math"
        # no_use_prompt_pool = True
    elif dataset_type == "trival_qa":
        print("trival_qa")
        loader = DataloaderForTrivalQA(split="train")
    elif dataset_type == "arc":
        print("arc")
        loader = DataloaderForARC(split="train")
        score_type = "exact-match"
    elif dataset_type == "mmlu":
        print("mmlu")
        loader = DataloaderForMMLU(split="auxiliary_train")
        score_type = "exact-match"

    for _ in range(skipping):
        loader.sample_once()
    for i in range(iteration_times):
        ports = []
        process = subprocess.Popen(
            f"""
        source ~/.bashrc && \
        conda activate {vllm_env} && \
        CUDA_VISIBLE_DEVICES={devices} vllm serve {model_path} --host 0.0.0.0 --port {port} --served-model-name "Llama-3" --enable-prefix-caching
        """,
            shell=True,
        )
        ports.append(port)
        for j in range(7):
            process_occupy = subprocess.Popen(
                f"""
            source ~/.bashrc && \
            conda activate {vllm_env} && \
            CUDA_VISIBLE_DEVICES={j} vllm serve {model_path} --host 0.0.0.0 --port {port+j+1} --served-model-name "Llama-3" --enable-prefix-caching
            """,
                shell=True,
            )
            ports.append(port + j + 1)
        while True:
            try:
                message_input = [{"role": "assistant", "content": "hello!"}]
                headers = {"Content-Type": "application/json"}
                data_json = {
                    "model": "Llama-3",
                    "messages": message_input,
                }
                response = requests.post(
                    f"http://0.0.0.0:{port}/v1/chat/completions",
                    headers=headers,
                    json=data_json,
                )
                content = (response.json()["choices"][0]["message"]["content"],)
                print(f"ready to generate data: {content}")
                break
            except:
                continue
        pid = process.pid
        vllm_data_generate(
            "Llama-3",
            "Llama-3",
            url_first=f"http://0.0.0.0:{port}/v1/chat/completions",
            url_second=f"http://0.0.0.0:{port}/v1/chat/completions",
            tokenizer_path_first=tokenizer_first_path,
            tokenizer_path_second=tokenizer_second_path,
            sample_count=sample_count,
            explore_count=explore_count,
            output_path=os.path.join(mid_sft_jsonl_root_path, f"iteration_{i}.jsonl"),
            thread_count=thread_count,
            prompt_pool_path=prompt_pool_path,
            dataloader=loader,
            no_use_prompt_pool=((i != 0) or no_use_prompt_pool),
            temperature=0.7 if ((i != 0) or no_use_prompt_pool or is_math) else 0.3,
            ports=ports,
            iteration=i,
        )
        time.sleep(10)
        for j in range(8):
            os.system(
                f"""pkill -f "vllm serve {model_path} --host 0.0.0.0 --port {port+j} --served-model-name Llama-3 --enable-prefix-caching" """
            )
        time.sleep(30)
        process = subprocess.Popen(
            f"""source ~/.bashrc && \
                conda activate {alignment_env} && \
                python ppl_deploy.py 
                """,
            shell=True,
        )

        while True:
            try:
                ret = requests.post(
                    "http://localhost:8000/ppl", json={"texts": ["hi hi" * 5000]}
                )
                if ret.status_code == 200:
                    break
            except:
                continue

        # reward
        process = subprocess.Popen(
            f"""source ~/.bashrc && \
                conda activate {alignment_env} && \
                HF_DATASETS_CACHE="../huggingface_cache/huggingface_dataset_cache" python reward_main.py --raw_data_path {os.path.join(mid_sft_jsonl_root_path, f"iteration_{i}.jsonl")}\
                --rewarded_output_path {os.path.join(mid_sft_jsonl_root_path, f"rewarded_iteration_{i}.jsonl")}\
                --cleaned_output_path {os.path.join(mid_sft_jsonl_root_path, f"cleaned_iteration_{i}.jsonl")}\
                --model_path {initial_model_path}\
                --sft_dataset_output_path {os.path.join(mid_sft_dataset_root_path,f"iteration_{i}")}\
                --score 1 --clean 1\
                --episilon {episilon}\
                --deploy 1\
                --num_thread {thread_count}\
                --score_type {score_type}\
                --cal_ppl {cal_ppl}\
                --lambda1 {lambda1}\
                --lambda2 {lambda2}\
                --prompt_type {loader.data_type}
                """,
            shell=True,
        )
        process.wait()
        os.system("""pkill -f "python ppl_deploy.py" """)
        dataset_path = os.path.join(mid_sft_dataset_root_path, f"iteration_{i}")
        time.sleep(20)
        # train
        with open(origin_sft_yaml_path, "r") as f:
            config = yaml.safe_load(f)
        if not from_initial:
            config["model_name_or_path"] = "."+model_path
        else:
            config["model_name_or_path"] = initial_model_path
        config["dataset_mixer"] = {"."+dataset_path: 1.0}
        config["output_dir"] = os.path.join("."+check_point_root_path, f"sft_iteration_{i}")

        try:
            with open(
                os.path.join(mid_yaml_root_path, f"sft_iteration_{i}.yaml"), "r"
            ) as f:
                pass
        except:
            with open(
                os.path.join(mid_yaml_root_path, f"sft_iteration_{i}.yaml"), "w"
            ) as fout:
                fout.write(yaml.safe_dump(config))
            process = subprocess.Popen(
                f"""source ~/.bashrc &&\
                conda activate {alignment_env} && \
                ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_sft.py  {os.path.join(mid_yaml_root_path[21:],f"sft_iteration_{i}.yaml")} """,
                shell=True,
                cwd="./alignment-handbook",
            )
            process.wait()
        model_path = os.path.join(check_point_root_path, f"sft_iteration_{i}")
        time.sleep(15)
        # deploy model for dpo
        ports = []
        for j in range(7):
            process_occupy = subprocess.Popen(
                f"""
            source ~/.bashrc && \
            conda activate {vllm_env} && \
            CUDA_VISIBLE_DEVICES={j+1} vllm serve {model_path} --host 0.0.0.0 --port {port+1+j} --served-model-name "Llama-3" --enable-prefix-caching
            """,
                shell=True,
            )
            ports.append(port + 1 + j)

        process = subprocess.Popen(
            f"""source ~/.bashrc && \
                conda activate {alignment_env} && \
                python ppl_deploy.py --num_replicas 1
                """,
            shell=True,
        )
        while True:
            try:
                ret = requests.post(
                    "http://localhost:8000/ppl", json={"texts": ["hi hi" * 5000]}
                )
                if ret.status_code == 200:
                    break
            except:
                continue

        while True:
            try:
                message_input = [{"role": "assistant", "content": "hello!"}]
                headers = {"Content-Type": "application/json"}
                data_json = {
                    "model": "Llama-3",
                    "messages": message_input,
                }
                response = requests.post(
                    f"http://0.0.0.0:{port+1}/v1/chat/completions",
                    headers=headers,
                    json=data_json,
                )
                print(response)
                content = (response.json()["choices"][0]["message"]["content"],)
                print(f"ready to generate data: {content}")
                break
            except:
                continue
        # sample 100
        process = subprocess.Popen(
            f"""
            source ~/.bashrc && \
            conda activate {alignment_env} && \
            HF_DATASETS_CACHE="../huggingface_cache/huggingface_dataset_cache" python inference_main.py\
            --output_path {os.path.join(mid_dpo_jsonl_root_path,f"tmp_iteration_{i}.jsonl")}\
            --sample_count 100\
            --num_thread 24\
            --url_first "http://0.0.0.0:{port+1}/v1/chat/completions"\
            --url_second "http://0.0.0.0:{port+1}/v1/chat/completions"\
            --dataset_type {dataset_type}\
            --skipping {loader.current_task_id+1}\
            --temperature 0.7\
            --ports [{",".join([str(p) for p in ports])}]\
            --add_name 1
            """,
            shell=True,
        )
        process.wait()
        # calculate quantile
        token_count_list = []
        with open(
            os.path.join(mid_dpo_jsonl_root_path, f"tmp_iteration_{i}.jsonl"), "r"
        ) as f:
            for line in f:
                data = json.loads(line)
                token_count_list.append(data["results"][0]["token_count"])
        percentiles = [80]
        quantile = int(np.percentile(token_count_list, percentiles)[0]) + 1
        # monte_carlo data generate
        time.sleep(10)
        try:
            monte_carlo_data_generate_deploy(
                max_token=quantile,
                model="http://localhost:8000/ppl",
                tokenizer_path=initial_model_path,
                sft_data_path="",
                output_path=os.path.join(
                    mid_dpo_jsonl_root_path, f"iteration_{i}.jsonl"
                ),
                sample_count=monte_sample_count,
                num_thread=thread_count,
                dataloader=loader,
                prompt_pool_path=prompt_pool_path,
                model_url=f"http://0.0.0.0:{port}/v1/chat/completions",
                min_value=dpo_episilon,
                incremental_threshold=dpo_min_value,
                ports=ports,
                score_type=score_type,
                cal_ppl=(cal_ppl == 1),
                lambda1=lambda1,
                lambda2=lambda2,
            )
        except:
            for j in range(7):
                os.system(
                    f"""pkill -f "vllm serve {model_path} --host 0.0.0.0 --port {port+j+1} --served-model-name Llama-3 --enable-prefix-caching" """
                )
            os.system("""pkill -f "python ppl_deploy.py --num_replicas 1" """)
            torch.cuda.empty_cache()
            raise
            exit(0)
        for j in range(7):
            os.system(
                f"""pkill -f "vllm serve {model_path} --host 0.0.0.0 --port {port+j+1} --served-model-name Llama-3 --enable-prefix-caching" """
            )
        os.system("""pkill -f "python ppl_deploy.py --num_replicas 1" """)
        torch.cuda.empty_cache()
        time.sleep(30)
        # dpo dataset
        process = subprocess.Popen(
            f"""
            source ~/.bashrc && \
            conda activate {alignment_env} && \
            HF_DATASETS_CACHE="../huggingface_cache/huggingface_dataset_cache" python reward_main.py --cleaned_output_path {os.path.join(mid_dpo_jsonl_root_path,f"iteration_{i}_dpo_format.jsonl")}\
            --dpo_dataset_output_path {os.path.join(mid_dpo_dataset_root_path,f"iteration_{i}")}
            """,
            shell=True,
        )
        process.wait()
        dataset_path = os.path.join(mid_dpo_dataset_root_path, f"iteration_{i}")
        # dpo train
        with open(origin_dpo_yaml_path, "r") as f:
            config = yaml.safe_load(f)
        config["model_name_or_path"] = "."+model_path
        config["dataset_mixer"] = {"."+dataset_path: 1.0}
        config["output_dir"] = os.path.join("."+check_point_root_path, f"dpo_iteration_{i}")

        try:
            with open(
                os.path.join(mid_yaml_root_path, f"dpo_iteration_{i}.yaml"), "r"
            ) as f:
                pass
        except:
            with open(
                os.path.join(mid_yaml_root_path, f"dpo_iteration_{i}.yaml"), "w"
            ) as fout:
                fout.write(yaml.safe_dump(config))
            process = subprocess.Popen(
                f"""source ~/.bashrc &&\
                conda activate {alignment_env} && \
                ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py  {os.path.join(mid_yaml_root_path[21:],f"dpo_iteration_{i}.yaml")} """,
                shell=True,
                cwd="./alignment-handbook",
            )
            process.wait()
        model_path = os.path.join(check_point_root_path, f"dpo_iteration_{i}")
