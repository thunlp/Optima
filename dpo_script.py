from train.dpo import dpo_train, dpo_train_v2
from argparse import ArgumentParser
import yaml
import os

argumentParser = ArgumentParser()
argumentParser.add_argument(
    "--train_config_path",
    type=str,
    default="/home/test/test04/yuanjiarui/project/src/train/dpo_recipes/hotpot_qa_v3.yaml",
)
argumentParser.add_argument(
    "--skipping",
    type=int,
    default=0,
)
argumentParser.add_argument("--vllm_env", type=str, required=True)
argumentParser.add_argument("--alignment_env", type=str, required=True)

args = argumentParser.parse_args()

if __name__ == "__main__":
    with open(args.train_config_path, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(config["mid_yaml_root_path"], exist_ok=True)
    os.makedirs(config["mid_dpo_jsonl_root_path"], exist_ok=True)
    os.makedirs(config["mid_dpo_dataset_root_path"], exist_ok=True)

    dpo_train_v2(
        config["origin_dpo_yaml_path"],
        config["initial_model_path"],
        config["initial_dataset_path"],
        config["dataset_type"],
        config["mid_yaml_root_path"],
        config["mid_dpo_jsonl_root_path"],
        config["mid_dpo_dataset_root_path"],
        config["check_point_root_path"],
        config["initial_episilon"],
        config["initial_dpo_min_value"],
        config["initial_dpo_episilon"],
        config["iteration_times"],
        config["port"],
        config["devices"],
        config["tokenizer_first_path"],
        config["tokenizer_second_path"],
        config["sample_count"],
        config["monte_sample_count"],
        config["explore_count"],
        config["thread_count"],
        config["prompt_pool_path"],
        skipping=args.skipping,
        cal_ppl=config["cal_ppl"],
        from_initial=config["from_initial"],
        lambda1=config["lambda1"],
        lambda2=config["lambda2"],
        vllm_env=args.vllm_env,
        alignment_env=args.alignment_env,
    )