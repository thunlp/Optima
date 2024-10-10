from analysis.analysis import multi_sc_analysis
from argparse import ArgumentParser
from utils.config import llama3_path_a800
from transformers import AutoTokenizer

argumentParser = ArgumentParser()
argumentParser.add_argument("--input_path", type=str, required=True)
argumentParser.add_argument("--max_rounds", type=int, required=True)
argumentParser.add_argument("--output_root_path", type=str, required=True)
argumentParser.add_argument("--score_type", type=str, default="f1-score")
args = argumentParser.parse_args()


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(llama3_path_a800)

    multi_sc_analysis(
        input_path=args.input_path,
        max_rounds=args.max_rounds,
        output_root_path=args.output_root_path,
        tokenizer=tokenizer,
        score_type=args.score_type,
    )
