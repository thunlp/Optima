from typing import List
from utils import lambda1, lambda2, reward_device, llama3_path
import torch
import torch.nn as nn
import torch.nn.functional as F
from rouge import Rouge
import json
import requests
import math
from openai import OpenAI
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import re
from collections import Counter
from answerParser.parser import is_equiv

file_lock = threading.Lock()


def get_score(
    result: dict,
    max_token_count: int,
    tokenizer,
    model,
    device: str,
    _lambda1: float = lambda1,
    _lambda2: float = lambda2,
):
    rouger = Rouge()
    try:
        correct_score = rouger.get_scores(
            result["answer"],
            (
                result["final_answer"]
                if result["final_answer"] != "" and result["final_answer"] != "..."
                else "UNKNOW"
            ),
        )
        correct_score = correct_score[0]["rouge-l"]["f"]
    except:
        correct_score = 0
    token_score = _lambda1 * result["token_count"] / max_token_count

    ppl_list = []
    for sentence in result["conversation"]:
        sentence = [{"role": "assistant", "content": sentence}]
        sentence = tokenizer.apply_chat_template(sentence, tokenize=False)
        message_input = tokenizer(sentence, return_tensors="pt")["input_ids"].to(device)
        target_output = message_input.clone()
        with torch.no_grad():
            output = model(message_input, labels=target_output)
            nll_loss = output.loss
        ppl_list.append(nll_loss.item())
    max_ppl = max(ppl_list)
    ppl_score = _lambda2 / max_ppl
    reward = correct_score + token_score + ppl_score
    print(
        f"correct: {correct_score},token: {token_score},ppl:{ppl_score},reward:{correct_score + token_score + ppl_score}"
    )
    return reward, correct_score, ppl_score


def reward(
    task_results: dict,
    model,
    tokenizer,
    device: str,
    _lambda1: float = lambda1,
    _lambda2: float = lambda2,
):

    results_list: List[dict] = task_results["results"]
    rewarded_results_list: List[dict] = []
    max_token_count = max([result["token_count"] for result in results_list])
    rouger = Rouge()
    for result in results_list:
        result["reward"], _, _ = get_score(
            result, max_token_count, tokenizer, model, device, _lambda1, _lambda2
        )
        rewarded_results_list.append(result)

    return rewarded_results_list


def cal_ppl_list(
    tokenizer,
    model,
    tokenized_sentences: List,
    device: str = "cuda:0",
    max_input_length: int = 100,
):
    record = 0
    ppl_list = []
    finish = False
    while True:
        if record + max_input_length >= len(tokenized_sentences):
            finish = True
            end = len(tokenized_sentences)
        else:
            end = record + max_input_length
        batch_input = tokenizer(
            tokenized_sentences[record:end],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )["input_ids"].to(device)

        with torch.inference_mode():
            output = model(input_ids=batch_input)
            logits = output.logits

        shift_logits = logits[..., 4:-1, :].contiguous()
        shift_labels = batch_input[..., 5:].contiguous()

        loss_fct = nn.CrossEntropyLoss(
            reduction="none", ignore_index=tokenizer.pad_token_id
        )
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        loss = loss.view(shift_labels.size())
        ppl_list.append(
            loss.sum(dim=1) / (shift_labels != tokenizer.pad_token_id).sum(dim=1)
        )
        if finish:
            break
        record += max_input_length
    return torch.cat(ppl_list)


def cal_f1_score(ref: str, tar: str, tokenizer):
    ref_token = tokenizer.tokenize(ref.lower().strip().strip("m=").strip("m = "))
    tar_token = tokenizer.tokenize(tar.lower().strip().strip("m=").strip("m = "))
    all_tokens = set(ref_token + tar_token)

    counter_ref = Counter(ref_token)
    counter_tar = Counter(tar_token)

    tp = sum((counter_ref & counter_tar).values())
    fp = sum((counter_tar - counter_ref).values())
    fn = sum((counter_ref - counter_tar).values())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def reward_batch(
    task_results: dict,
    model,
    tokenizer,
    device: str,
    _lambda1: float,
    _lambda2: float,
    score_type: str = "f1-score",
    cal_ppl: bool = True,
):
    tokenizer.pad_token_id = 128002
    results_list: List[dict] = task_results["results"]
    rewarded_results_list: List[dict] = []
    max_token_count = max([result["token_count"] for result in results_list])
    for result in results_list:
        try:
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
        except:
            correct_score = 0
        token_score = 0
        try:
            token_score = _lambda1 * result["token_count"] / max_token_count
        except:
            token_score = 0
        ppl_score = 0

        if cal_ppl:
            if result["token_count"] > 500 or len(result["conversation"]) == 0:
                ppl_score = -1
            else:
                tokenized_sentences = []
                for sentence in result["conversation"]:
                    sentence = [{"role": "assistant", "content": sentence}]
                    tokenized_sentence = tokenizer.apply_chat_template(
                        sentence, tokenize=False
                    )
                    tokenized_sentences.append(tokenized_sentence)
                batch_input = tokenizer(
                    tokenized_sentences,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )["input_ids"].to(device)

                target_output = batch_input.clone()
                with torch.no_grad():
                    output = model(input_ids=batch_input)
                    logits = output.logits

                shift_logits = logits[..., 4:-1, :].contiguous()
                shift_labels = target_output[..., 5:].contiguous()
                loss_fct = nn.CrossEntropyLoss(
                    reduction="none", ignore_index=tokenizer.pad_token_id
                )
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )

                loss = loss.view(shift_labels.size())

                ppl_list = loss.sum(dim=1) / (
                    shift_labels != tokenizer.pad_token_id
                ).sum(dim=1)

                max_ppl = np.max(ppl_list.cpu().numpy())
                ppl_score = _lambda2 / max_ppl

        reward = correct_score + token_score + ppl_score
        result["reward"] = reward
        rewarded_results_list.append(result)
        print(
            f"correct: {correct_score},token: {token_score},ppl:{ppl_score},reward:{correct_score + token_score + ppl_score}"
        )
    return rewarded_results_list


def reward_list_batch(
    task_results_list: List[dict],
    model,
    tokenizer,
    device: str,
    _lambda1: float,
    _lambda2: float,
):
    tokenizer.pad_token_id = 128002
    explore_time = len(task_results_list[0]["results"])
    tokenized_sentences = []
    rouge_score_list = []
    token_score_list = []
    conversation_length_list = []
    rouger = Rouge()
    for task_results in task_results_list:
        results_list = task_results["results"]
        max_token_count = max([result["token_count"] for result in results_list])
        for result in results_list:
            try:
                correct_score = rouger.get_scores(
                    result["answer"],
                    (
                        result["final_answer"]
                        if result["final_answer"] != ""
                        and result["final_answer"] != "..."
                        else "UNKNOW"
                    ),
                )
                correct_score = correct_score[0]["rouge-l"]["f"]
            except:
                correct_score = 0
            rouge_score_list.append(correct_score)
            token_score_list.append(result["token_count"] / max_token_count)
            for sentence in result["conversation"]:
                sentence = [{"role": "assistant", "content": sentence}]
                tokenized_sentence = tokenizer.apply_chat_template(
                    sentence, tokenize=False
                )
                tokenized_sentences.append(tokenized_sentence)
            conversation_length_list.append(len(result["conversation"]))
    ppl_list = cal_ppl_list(tokenizer, model, tokenized_sentences, device)
    current_index = 0
    ppl_start_index = 0

    for i in range(0, len(rouge_score_list), explore_time):
        for j in range(i, i + explore_time):
            max_ppl = torch.max(
                ppl_list[
                    ppl_start_index : ppl_start_index + conversation_length_list[j]
                ]
            ).item()
            ppl_score = _lambda2 / max_ppl
            task_results_list[current_index]["results"][j - i]["reward"] = (
                rouge_score_list[j] + _lambda1 * token_score_list[j] + ppl_score
            )
            print(
                f"rouge-l: {rouge_score_list[j]} token_score: {_lambda1 * token_score_list[j]} ppl_score: {ppl_score}  reward: {rouge_score_list[j] + _lambda1 * token_score_list[j] + ppl_score}"
            )
            ppl_start_index += conversation_length_list[j]
        current_index += 1
    return task_results_list


def process_raw_conversation_data_batch(
    model,
    tokenizer,
    input_path,
    output_path,
    device: str = "cuda:0",
    _lambda1: float = lambda1,
    _lambda2: float = lambda2,
    batch_size: int = 1,
):
    record_set = set()
    skipping = 0
    with open(output_path, "r") as fin:
        for line in fin:
            data = json.loads(line)
            skipping += 1
            record_set.add(data["task_id"])

    print(f"skipping: {skipping}")
    print(f"{record_set}")

    record = 0
    task_results_list = []

    full_line = subprocess.run(["wc", "-l", input_path], capture_output=True, text=True)
    full_line = int(full_line.stdout.split()[0])
    current_line = 0
    with open(input_path, "r") as f:
        for line in f:
            data = json.loads(line)
            if data["task_id"] in record_set:
                continue
            record += 1
            current_line += 1
            task_results_list.append(data)
            if record == batch_size or current_line == full_line:
                print(f"-------------------{current_line}---------------------")
                rewarded_data_list = reward_list_batch(
                    task_results_list, model, tokenizer, device, _lambda1, _lambda2
                )
                with open(output_path, "a") as fout:
                    for rewarded_data in rewarded_data_list:
                        fout.write(json.dumps(rewarded_data) + "\n")
                record = 0
                task_results_list = []


def reward_based_on_vllm(task_results: dict, model_url: str, output_path: str):
    results_list: List[dict] = task_results["results"]
    rewarded_results_list: List[dict] = []
    max_token_count = max([result["token_count"] for result in results_list])
    rouger = Rouge()
    for result in results_list:
        try:
            correct_score = rouger.get_scores(
                result["answer"],
                (
                    result["final_answer"]
                    if result["final_answer"] != "" and result["final_answer"] != "..."
                    else "UNKNOW"
                ),
            )
            correct_score = correct_score[0]["rouge-l"]["f"]
        except:
            correct_score = 0
        token_score = lambda1 * result["token_count"] / max_token_count

        ppl_list = []
        for sentence in result["conversation"]:
            with torch.no_grad():
                client = OpenAI(api_key="EMPTY", base_url=model_url)
                models = client.models.list()
                model = models.data[0].id
                completion = client.completions.create(
                    model=model,
                    prompt=sentence,
                    echo=True,
                    max_tokens=0,
                    logprobs=1,
                )
                response = completion.choices[0].dict()
                token_logprobs = response["logprobs"]["token_logprobs"][1:]
                ppl = math.exp(-np.sum(token_logprobs) / len(token_logprobs))
            ppl_list.append(ppl)

        max_ppl = max(ppl_list)
        ppl_score = lambda2 / max_ppl
        reward = correct_score + token_score + ppl_score
        result["reward"] = reward
        rewarded_results_list.append(result)
        print(
            f"correct: {correct_score},token: {token_score},ppl:{ppl_score},reward:{correct_score + token_score + ppl_score}"
        )
    with file_lock:
        with open(output_path, "a") as fout:
            fout.write(
                json.dumps(
                    {
                        "task_id": task_results["task_id"],
                        "results": rewarded_results_list,
                    }
                )
                + "\n"
            )


def process_raw_conversation_data(
    model,
    tokenizer,
    input_path,
    output_path,
    device: str,
    _lambda1: float = lambda1,
    _lambda2: float = lambda2,
    cal_ppl: bool = True,
    score_type: str = "f1-score",
):
    record_set = set()
    skipping = 0
    try:
        with open(output_path, "r") as fin:
            for line in fin:
                data = json.loads(line)
                skipping += 1
                record_set.add(data["task_id"])
    except:
        pass
    print(f"skipping: {skipping}")
    print(f"{record_set}")

    fout = open(output_path, "a")
    with open(input_path, "r") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            if data["task_id"] in record_set:
                continue
            print(f"-----------{data['task_id']}-------------")
            data["results"] = reward_batch(
                data,
                model,
                tokenizer,
                device,
                _lambda1,
                _lambda2,
                score_type=score_type,
                cal_ppl=cal_ppl,
            )
            try:
                fout.write(json.dumps(data) + "\n")
            except:
                pass
    fout.close()


def process_raw_conversation_data_based_on_vllm(
    model_url: str, input_path: str, output_path: str, thread_count: int
):
    args_list = []
    with open(input_path, "r") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            args_list.append((data, model_url))

    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        futures = [
            executor.submit(reward_based_on_vllm, task_results, model_url_, output_path)
            for (task_results, model_url_) in args_list
        ]
        for future in as_completed(futures):
            pass


def result_stats(input_path: str, tokenizer, score_type, is_consistence: bool = False):
    """
    Calculate average F1 score (or other scores) and average token count from the results.

    Parameters:
        input_path (str): Path to the input data file containing results.
        tokenizer: The tokenizer used to process the answers.
        score_type (str): The type of score to calculate (e.g., "f1-score", "exact-match").
        is_consistence (bool): Whether to calculate average token count consistently.

    Returns:
        tuple: Average acc score and average token count.
    """
    average_rouge = 0
    average_token = 0
    count = 0
    with open(input_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
            except:
                print(line)

                continue
            results = data["results"]
            tmp_count = 0
            tmp_average_token = 0
            tmp_average_rouge = 0
            count += 1
            for result in results:
                tmp_count += 1
                if (
                    "large" in result["conversation"]
                    or "error" in result["conversation"]
                ):
                    continue

                tmp_average_token += result["token_count"]
                try:
                    if isinstance(result["answer"], list):
                        if score_type == "f1-score":
                            all_score = [
                                cal_f1_score(answer, result["final_answer"], tokenizer)
                                for answer in result["answer"]
                            ]
                            correct_score = max(all_score)
                        elif score_type == "exact-match":
                            answers = [
                                answer.strip().lower() for answer in result["answer"]
                            ]
                            correct_score = (
                                1
                                if result["final_answer"].strip().lower() in answers
                                else 0
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
                            result["final_answer"] = re.findall(
                                r"-?\d+(?:\.\d+)?", result["final_answer"]
                            )[0]

                            correct_score = (
                                1
                                if result["answer"].strip().lower()
                                == result["final_answer"].strip().lower()
                                else 0
                            )
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
                except:
                    correct_score = 0
                tmp_average_rouge += correct_score
                if score_type == "math":
                    break
            average_rouge += tmp_average_rouge / tmp_count
            if not is_consistence:
                average_token += tmp_average_token / tmp_count
            else:
                average_token += tmp_average_token
    average_token /= count
    average_rouge /= count
    print(f"average_f1_score: {average_rouge} ,  average_token: {average_token}")
    return average_rouge, average_token
