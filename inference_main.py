from transformers import AutoTokenizer
from train.datagenerate import vllm_data_generate
from train.inference import inference
from utils.config import llama3_path_a100
from argparse import ArgumentParser
from utils.config import llama3_path_a100, llama3_path_aistation, llama3_path_a800
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


argumentParser = ArgumentParser()
argumentParser.add_argument("--train_data_path", type=str, default="")
argumentParser.add_argument(
    "--tokenizer_path",
    type=str,
    default=llama3_path_a800,
)
argumentParser.add_argument(
    "--output_path", type=str, default="inference_results/hotpot_qa.jsonl"
)
argumentParser.add_argument("--sample_count", type=int, default=1000)
argumentParser.add_argument("--num_thread", type=int, default=24)
argumentParser.add_argument(
    "--url_first", type=str, default="http://0.0.0.0:8000/v1/chat/completions"
)
argumentParser.add_argument(
    "--url_second", type=str, default="http://0.0.0.0:8000/v1/chat/completions"
)
argumentParser.add_argument("--dataset_type", type=str, default="hotpot_qa")
argumentParser.add_argument("--skipping", type=int, default=0)
argumentParser.add_argument("--temperature", type=float, default=0.0)
argumentParser.add_argument("--no_use_prompt_pool", type=int, default=1)
argumentParser.add_argument("--split", type=str, default="train")
argumentParser.add_argument("--explore_count", type=int, default=1)
argumentParser.add_argument("--ports", type=str, default="[8000]")
argumentParser.add_argument("--add_name", type=int, default=0)
args = argumentParser.parse_args()

if __name__ == "__main__":
    ports = args.ports.strip("[").strip("]").split(",")
    ports = [int(port) for port in ports]
    loader = None
    if args.dataset_type == "hotpot_qa":
        print("hotpot")
        loader = DataloaderForHotpotQA(split=args.split)
    elif args.dataset_type == "mwh_qa":
        print("mwh")
        loader = DataloaderForMWHQA(split=args.split)
    elif args.dataset_type == "cbt":
        print("cbt")
        loader = DataloaderForCBT(split=args.split)
    elif args.dataset_type == "gsm8k":
        print("gsm8k")
        loader = DataloaderForGSM8K(split=args.split)
    elif args.dataset_type == "math":
        print("math")
        loader = DataloaderForMATH(split=args.split)
    elif args.dataset_type == "trival_qa":
        print("trival_qa")
        loader = DataloaderForTrivalQA(split=args.split)
    elif args.dataset_type == "arc":
        print("arc")
        loader = DataloaderForARC(split=args.split)
    elif args.dataset_type == "mmlu":
        print("mmlu")
        loader = DataloaderForMMLU(
            split="auxiliary_train" if args.split == "train" else "test"
        )
    for _ in range(args.skipping):
        loader.sample_once()
    inference(
        "Llama-3",
        "Llama-3",
        args.url_first,
        args.url_second,
        args.tokenizer_path,
        args.tokenizer_path,
        sample_count=args.sample_count,
        explore_count=args.explore_count,
        output_path=args.output_path,
        thread_count=args.num_thread,
        prompt_pool_path="",
        train_data_path=args.train_data_path,
        dataloader=loader,
        temperature=args.temperature,
        no_use_prompt_pool=(args.no_use_prompt_pool == 1),
        ports=ports,
        add_name=args.add_name,
    )
