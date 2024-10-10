from train.sft import sft_train, sft_train_v2
from argparse import ArgumentParser
from utils.config import llama3_path_a800
import yaml

argumentParser = ArgumentParser()
argumentParser.add_argument(
    "--train_config_path",
    type=str,
    default="train/sft_recipes/hotpot_qa.yaml",
)
argumentParser.add_argument(
    "--skipping",
    type=int,
    default=0,
)
argumentParser.add_argument("--skip_iteration", type=int, default=0)
args = argumentParser.parse_args()

if __name__ == "__main__":
    with open(args.train_config_path, "r") as f:
        config = yaml.safe_load(f)
    sft_train_v2(
        config["origin_yaml_path"],
        config["initial_model_path"],
        config["initial_dataset_path"],
        config["dataset_type"],
        config["mid_yaml_root_path"],
        config["mid_jsonl_root_path"],
        config["mid_dataset_root_path"],
        config["check_point_root_path"],
        config["initial_episilon"],
        config["iteration_times"],
        config["port"],
        config["devices"],
        config["tokenizer_first_path"],
        config["tokenizer_second_path"],
        config["sample_count"],
        config["explore_count"],
        config["thread_count"],
        config["prompt_pool_path"],
        skipping=args.skipping,
        cal_ppl=config["cal_ppl"],
        skip_iteration=args.skip_iteration,
        from_initial=config["from_initial"],
        lambda1=config["lambda1"],
        lambda2=config["lambda2"],
        mix_dataset=[],
    )
