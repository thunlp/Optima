import numpy as np
import json
from reward.reward import cal_f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
from answerParser.parser import is_equiv
from reward.reward import result_stats, cal_f1_score


def get_best_division(input_path, output_path, episilon, tokenizer, score_type):
    with open(input_path, "r") as f:
        for line in f:
            data = json.loads(line)
            divisions = []
            check_divisions = []
            results = data["results"]
            final_answers = [result["final_answer"] for result in results]
            messages = data["results"]
            for final_answer, message in zip(final_answers, messages):
                mid_division = None
                mid_check_division = None
                for division, check_division in zip(divisions, check_divisions):
                    for reference_answer, reference_message in division:
                        similarity = 0
                        try:
                            if score_type == "exact-match":
                                similarity = (
                                    1
                                    if reference_answer.strip().lower()
                                    == final_answer.strip().lower()
                                    else 0
                                )
                                if similarity == 0:
                                    pass
                            elif score_type == "f1-score":
                                similarity = cal_f1_score(
                                    reference_answer, final_answer, tokenizer
                                )
                            elif score_type == "gsm8k":
                                try:
                                    final_answer = re.findall(
                                        r"-?\d+(?:\.\d+)?", final_answer
                                    )[0]
                                except:
                                    pass

                                similarity = (
                                    1
                                    if reference_answer.strip().lower()
                                    == final_answer.strip().lower()
                                    else 0
                                )
                            elif score_type == "math":
                                similarity = (
                                    1
                                    if is_equiv(
                                        reference_answer.strip().lower(),
                                        final_answer.strip().lower(),
                                    )
                                    else 0
                                )
                        except:
                            similarity = 0
                        if similarity >= episilon:
                            mid_division = division
                            mid_check_division = check_division
                            break
                    if mid_division is not None:
                        division.append((final_answer, message))
                        check_division.append(final_answer)
                        break
                if mid_division is None:
                    divisions.append([(final_answer, message)])
                    check_divisions.append([final_answer])
            divisions_length_list = [len(division) for division in divisions]
            longest_index = np.argmax(divisions_length_list)

            best_division = divisions[longest_index]
            best_division = [result[1] for result in best_division]
            with open(output_path, "a") as fout:
                fout.write(
                    json.dumps({"task_id": data["task_id"], "results": best_division})
                    + "\n"
                )


def get_best_answer(input_path, score_type, tokenizer):
    count = 0
    average_acc = 0
    with open(input_path, "r") as f:
        for line in f:
            data = json.loads(line)
            acc_list = []
            for result in data["results"]:
                if isinstance(result["answer"], list):
                    if score_type == "f1-score":
                        all_score = [
                            cal_f1_score(answer, result["final_answer"], tokenizer)
                            for answer in result["answer"]
                        ]
                        correct_score = max(all_score)
                    elif score_type == "exact-match":
                        answers = [answer.strip().lower for answer in result["answer"]]
                        correct_score = (
                            1 if result["answer"].strip().lower() in answers else 0
                        )
                else:
                    if score_type == "f1-score":
                        correct_score = cal_f1_score(
                            result["answer"], result["final_answer"], tokenizer
                        )
                    elif score_type == "exact-match":
                        correct_score = (
                            1
                            if result["answer"].strip().lower()
                            == result["final_answer"].strip().lower()
                            else 0
                        )
                    elif score_type == "gsm8k":
                        try:
                            result["final_answer"] = re.findall(
                                r"-?\d+(?:\.\d+)?", result["final_answer"]
                            )[0]
                        except:
                            pass
                        try:
                            correct_score = (
                                1
                                if result["answer"].strip().lower()
                                == result["final_answer"].strip().lower()
                                else 0
                            )
                        except:
                            correct_score = 0
                    elif score_type == "math":
                        try:
                            correct_score = (
                                1
                                if is_equiv(
                                    result["answer"].strip().lower(),
                                    result["final_answer"].strip().lower(),
                                )
                                else 0
                            )
                        except:
                            correct_score = (
                                1
                                if result["answer"].strip().lower()
                                == result["final_answer"].strip().lower()
                                else 0
                            )
                acc_list.append(correct_score)
            average_acc += max(acc_list)
            count += 1
    print(f"best: {average_acc/count}")
    return average_acc / count


def multi_sc_analysis(input_path, max_rounds, output_root_path, tokenizer, score_type):
    """
    Performs multi-step analysis,computes accuracy scores, and saves the analysis results.

    Args:
        input_path (str): The path to the input file.
        max_rounds (int): The number of rounds to run the analysis, each round progressively adds more results.
        output_root_path (str): The root directory where the output files will be stored.
        tokenizer (Tokenizer): The tokenizer used for tokenizing text.
        score_type (str): The type of scoring metric to use for evaluation (e.g., 'f1-score', 'gsm8k', 'exact-match').

    Returns:
        None: The function saves results in files.
    """
    acc_list = []
    best_acc_list = []
    token_list = []
    multi_sc_path = os.path.join(output_root_path, "mid_multi_sc.jsonl")
    multi_sc_best_division_path = os.path.join(
        output_root_path, "mid_multi_sc_best_division.jsonl"
    )

    for i in range(max_rounds):
        print(f"------------------{i}------------------------")
        with open(multi_sc_path, "w") as fout:
            fout.write("")
        with open(multi_sc_best_division_path, "w") as fout:
            fout.write("")

        with open(input_path, "r") as f:
            for line in f:
                data = json.loads(line)
                data["results"] = data["results"][: i + 1]
                with open(multi_sc_path, "a") as fout:
                    fout.write(json.dumps(data) + "\n")
        best_acc = get_best_answer(
            multi_sc_path,
            (
                score_type
                if score_type == "f1-score" or score_type == "gsm8k"
                else "exact-match"
            ),
            tokenizer,
        )
        best_acc_list.append(best_acc)
        get_best_division(
            multi_sc_path,
            multi_sc_best_division_path,
            episilon=0.9,
            tokenizer=tokenizer,
            score_type=score_type if score_type == "f1-score" else "exact-match",
        )
        acc, _ = result_stats(multi_sc_best_division_path, tokenizer, score_type, False)
        _, token = result_stats(multi_sc_path, tokenizer, "exact-match", True)
        acc_list.append(acc)
        token_list.append(token)

    print(f"acc_list: {acc_list}")
    print(f"best_acc_list: {best_acc_list}")
    print(f"token_list: {token_list}")
    with open(multi_sc_path, "w") as fout:
        fout.write("")
    with open(multi_sc_best_division_path, "w") as fout:
        fout.write("")
    with open(os.path.join(output_root_path, "record.jsonl"), "a") as fout:
        fout.write(
            json.dumps(
                {
                    "acc_list": acc_list,
                    "best_acc_list": best_acc_list,
                    "token_list": token_list,
                }
            )
        )
