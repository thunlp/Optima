from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from train.datagenerate import vllm_data_generate
from dataloader.dataloader import (
    DataloaderForHotpotQA,
    DataloaderForMWHQA,
    DataloaderForCBT,
    DataloaderForGSM8K,
    DataloaderForMATH,
    DataloaderForTrivalQA,
    DataloaderForARC,
    DataloaderForMix,
    DataloaderForMMLU,
)
import os
import yaml
import subprocess
import requests
import time
from reward.deploy_reward import (
    serve_reward_model,
    process_raw_conversation_data_based_on_deploy,
)


def sft_train(
    origin_yaml_path: str,
    initial_model_path: str,
    initial_dataset_path: str,
    dataset_type: str,
    mid_yaml_root_path: str,
    mid_jsonl_root_path: str,
    mid_dataset_root_path: str,
    check_point_root_path: str,
    initial_episilon: float,
    iteration_times: int,
    port: int,
    devices: str,
    tokenizer_first_path: str,
    tokenizer_second_path: str,
    sample_count: int,
    explore_count: int,
    thread_count: int,
    prompt_pool_path: str,
    skipping: int,
):
    pass


def sft_train_v2(
    origin_yaml_path: str,
    initial_model_path: str,
    initial_dataset_path: str,
    dataset_type: str,
    mid_yaml_root_path: str,
    mid_jsonl_root_path: str,
    mid_dataset_root_path: str,
    check_point_root_path: str,
    initial_episilon: float,
    iteration_times: int,
    port: int,
    devices: str,
    tokenizer_first_path: str,
    tokenizer_second_path: str,
    sample_count: int,
    explore_count: int,
    thread_count: int,
    prompt_pool_path: str,
    skipping: int,
    cal_ppl: int = 1,
    skip_iteration: int = 0,
    from_initial: bool = False,
    lambda1: float = -0.6,
    lambda2: float = 1,
    mix_dataset: list = [],
    vllm_env:str = "",
    alignment_env:str = ""
):
    model_path = initial_model_path
    episilon = initial_episilon
    score_type = "f1-score"
    loader = None
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
        loader = DataloaderForGSM8K(split="train")
        score_type = "exact-match"
        # no_use_prompt_pool =True
    elif dataset_type == "math":
        score_type = "math"
        is_math = True
        # no_use_prompt_pool = True
        print("math")
        loader = DataloaderForMATH(split="train")
    elif dataset_type == "trival_qa":
        print("trival_qa")
        loader = DataloaderForTrivalQA(split="train")
    elif dataset_type == "arc":
        print("arc")
        score_type = "exact-match"
        loader = DataloaderForARC(split="train")
    elif dataset_type == "mmlu":
        print("mmlu")
        score_type = "exact-match"
        loader = DataloaderForMMLU(split="auxiliary_train")
    elif dataset_type == "mix":
        print("mix")
        score_type = "mix"
        loader = DataloaderForMix(datasets=mix_dataset, splits="train")
        # no_use_prompt_pool =True
    for _ in range(skipping):
        loader.sample_once()
    for i in range(iteration_times):
        if i < skip_iteration:
            model_path = os.path.join(check_point_root_path, f"iteration_{i}")
            continue
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
        # check deploy
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
        time.sleep(15)
        # generate raw data
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
            output_path=os.path.join(mid_jsonl_root_path, f"iteration_{i}.jsonl"),
            thread_count=thread_count,
            prompt_pool_path=prompt_pool_path,
            dataloader=loader,
            no_use_prompt_pool=((i != 0) or no_use_prompt_pool),
            temperature=0.7 if ((i != 0) or no_use_prompt_pool or is_math) else 0.3,
            ports=ports,
            iteration=i,
        )
        for j in range(8):
            os.system(
                f"""pkill -f "vllm serve {model_path} --host 0.0.0.0 --port {port+j} --served-model-name Llama-3 --enable-prefix-caching" """
            )
        time.sleep(30)
        process1 = subprocess.Popen(
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

        # reward and generate dataset
        process = subprocess.Popen(
            f"""source ~/.bashrc && \
                conda activate {alignment_env} && \
                HF_DATASETS_CACHE="../huggingface_cache/huggingface_dataset_cache" python reward_main.py --raw_data_path {os.path.join(mid_jsonl_root_path, f"iteration_{i}.jsonl")}\
                --rewarded_output_path {os.path.join(mid_jsonl_root_path, f"rewarded_iteration_{i}.jsonl")}\
                --cleaned_output_path {os.path.join(mid_jsonl_root_path, f"cleaned_iteration_{i}.jsonl")}\
                --model_path {initial_model_path}\
                --sft_dataset_output_path {os.path.join(mid_dataset_root_path,f"iteration_{i}")}\
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
        time.sleep(20)
        dataset_path = os.path.join(mid_dataset_root_path, f"iteration_{i}")
        # sft train
        with open(origin_yaml_path, "r") as f:
            config = yaml.safe_load(f)
        config["model_name_or_path"] = (
            initial_model_path if from_initial else "."+model_path
        )
        config["dataset_mixer"] = {"."+dataset_path: 1.0}
        config["output_dir"] = os.path.join("."+check_point_root_path, f"iteration_{i}")
        try:
            with open(
                os.path.join(mid_yaml_root_path, f"iteration_{i}.yaml"), "r"
            ) as f:
                pass
        except:
            with open(
                os.path.join(mid_yaml_root_path, f"iteration_{i}.yaml"), "w"
            ) as fout:
                fout.write(yaml.safe_dump(config))
            
            process = subprocess.Popen(
                f"""source ~/.bashrc &&\
                conda activate {alignment_env} && \
                ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_sft.py  {os.path.join(mid_yaml_root_path[21:],f"iteration_{i}.yaml")} """,
                shell=True,
                cwd="./alignment-handbook",
            )
            process.wait()
        # update
        model_path = os.path.join(check_point_root_path, f"iteration_{i}")
