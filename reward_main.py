from transformers import AutoModelForCausalLM, AutoTokenizer
from argparse import ArgumentParser
from reward.reward import (
    process_raw_conversation_data,
    process_raw_conversation_data_based_on_vllm,
    process_raw_conversation_data_batch,
)
from dataloader.dataloader import (
    process_dataloader_for_sft,
    preprocess_dataloader_for_dpo,
    data_clean,
    process_dpo_format_to_dataset,
)
from utils.config import llama3_path_a100, llama3_path_aistation, llama3_path_a800
import time
from collections import Counter
from reward.deploy_reward import process_raw_conversation_data_based_on_deploy

argumentParser = ArgumentParser()
argumentParser.add_argument(
    "--raw_data_path", type=str, default="results/vllm_test.jsonl"
)
argumentParser.add_argument("--use_vllm", type=int, default=0)
argumentParser.add_argument(
    "--rewarded_output_path", type=str, default="results/check/vllm_rewarded_data.jsonl"
)
argumentParser.add_argument(
    "--cleaned_output_path", type=str, default="results/check/vllm_cleaned_data.jsonl"
)
argumentParser.add_argument(
    "--model_path",
    type=str,
    default=llama3_path_a800,
)
argumentParser.add_argument("--sft_dataset_output_path", type=str, default=None)
argumentParser.add_argument("--dpo_dataset_output_path", type=str, default=None)
argumentParser.add_argument("--score", type=int, default=0)
argumentParser.add_argument("--clean", type=int, default=0)
argumentParser.add_argument("--model_url", type=str, default="http://0.0.0.0:8002/v1")
argumentParser.add_argument("--lambda1", type=float, default=-0.6)
argumentParser.add_argument("--lambda2", type=float, default=1)
argumentParser.add_argument("--episilon", type=float, default=0.5)
argumentParser.add_argument("--cal_ppl", type=int, default=1)
argumentParser.add_argument("--score_type", type=str, default="f1-score")
argumentParser.add_argument("--deploy", type=int, default=0)
argumentParser.add_argument("--num_thread", type=int, default=24)
argumentParser.add_argument("--prompt_type", type=str, default="qa")
args = argumentParser.parse_args()


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, torch_dtype="auto", attn_implementation="flash_attention_2"
    )
    if args.score != 0:
        if args.use_vllm != 1:
            model = None
            startTime = time.time()
            if args.deploy == 1:
                process_raw_conversation_data_based_on_deploy(
                    model_url="http://localhost:8000/ppl",
                    tokenizer=tokenizer,
                    input_path=args.raw_data_path,
                    output_path=args.rewarded_output_path,
                    _lambda1=args.lambda1,
                    _lambda2=args.lambda2,
                    cal_ppl=(args.cal_ppl == 1),
                    score_type=args.score_type,
                    num_thread=args.num_thread,
                )
            else:
                if args.cal_ppl == 1:
                    model = AutoModelForCausalLM.from_pretrained(
                        args.model_path,
                        torch_dtype="auto",
                        attn_implementation="flash_attention_2",
                    ).to("cuda:0")
                process_raw_conversation_data(
                    model,
                    tokenizer,
                    args.raw_data_path,
                    args.rewarded_output_path,
                    "cuda:0",
                    args.lambda1,
                    args.lambda2,
                    cal_ppl=(args.cal_ppl == 1),
                    score_type=args.score_type,
                )
            endTime = time.time()
            print(f"time: {endTime-startTime}")
        else:
            process_raw_conversation_data_based_on_vllm(
                args.model_url,
                args.raw_data_path,
                args.rewarded_output_path,
                thread_count=16,
            )
    if args.clean != 0:
        data_clean(args.rewarded_output_path, args.cleaned_output_path)
    if args.sft_dataset_output_path is not None:
        process_dataloader_for_sft(
            tokenizer,
            args.sft_dataset_output_path,
            args.cleaned_output_path,
            episilon=args.episilon,
            prompt_type=args.prompt_type,
        )
    if args.dpo_dataset_output_path is not None:
        process_dpo_format_to_dataset(
            args.cleaned_output_path, args.dpo_dataset_output_path
        )
