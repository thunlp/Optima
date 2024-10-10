from datasets import load_dataset, DatasetDict, Dataset
from string import Template
from pydantic import BaseModel
from typing import Union, Any
from utils.prompt_template import *
from answerParser.parser import normalize_final_answer
import random
import json
import numpy as np
import datasets
from tqdm import tqdm
import re
import os

split_map = {
    "train": {
        "hotpot_qa": "train",
        "mwh_qa": "train",
        "cbt": "train",
        "gsm8k": "train",
        "math": "train",
        "trival_qa": "train",
        "arc": "train",
    },
    "test": {
        "hotpot_qa": "validation",
        "mwh_qa": "dev",
        "cbt": "test",
        "gsm8k": "test",
        "math": "test",
        "trival_qa": "validation",
        "arc": "test",
    },
}


class BaseDataloader:
    def sample_once(self):
        pass

    def sample(self, count: int):
        pass


class DataloaderForHotpotQA(BaseDataloader):

    def __init__(
        self, dataset: str = "hotpot_qa", name: str = "distractor", split: str = "train"
    ):
        self.dataset = dataset
        self.name = name
        self.data_type = "qa"
        self.split = split
        self.train_set = load_dataset(
            path=dataset,
            name=name,
            split=split,
            trust_remote_code=True,
        )
        self.total = len(self.train_set)
        self.current_task_id: int = 0
        self.dataset_name = "hotpot_qa"

    def sample_once(self):
        data = self.train_set[self.current_task_id]
        question = data["question"]
        answer = data["answer"]
        supporting_facts_title: list = data["supporting_facts"]["title"]
        other_context_title = [
            title
            for title in data["context"]["title"]
            if title not in supporting_facts_title
        ]
        context_sentences = {}

        for i, title in enumerate(data["context"]["title"]):
            context_sentences[title] = data["context"]["sentences"][i][0]

        context1 = []
        context2 = []
        count = len(supporting_facts_title)
        for i in range(count):
            sample_title = random.sample(supporting_facts_title, 1)[0]
            if i % 2 == 0:
                context1.append(f"{context_sentences[sample_title]}\n")
            else:
                context2.append(f"{context_sentences[sample_title]}\n")
                supporting_facts_title
            supporting_facts_title.remove(sample_title)

        for title in other_context_title:
            judge = random.randint(1, 2)
            if judge == 1:
                context1.append(f"{context_sentences[title]}\n")
            else:
                context2.append(f"{context_sentences[title]}\n")

        self.current_task_id += 1
        if self.current_task_id >= self.total:
            self.current_task_id = 0

        return question, answer, context1, context2


class DataloaderForMWHQA(BaseDataloader):
    def __init__(
        self,
        dataset_path: str = "/home/test/test04/yuanjiarui/project/json_dataset/2MultiWikiQA",
        name: str = "",
        split: str = "train",
    ):
        with open(os.path.join(dataset_path, f"{split}.json"), "r") as f:
            self.data = json.load(f)
        self.data_type = "qa"
        self.total: int = len(self.data)
        self.current_task_id: int = 0
        self.dataset_name = "mwh_qa"

    def sample_once(self):
        question = self.data[self.current_task_id]["question"]
        answer = self.data[self.current_task_id]["answer"]
        contexts = {
            pair[0]: pair[1] for pair in self.data[self.current_task_id]["context"]
        }
        supporting_fact_titles = [
            supporting_fact[0]
            for supporting_fact in self.data[self.current_task_id]["supporting_facts"]
        ]
        context_first = []
        context_second = []
        for i, supporting_fact in enumerate(supporting_fact_titles):
            if i % 2 == 0:
                context_first.append("".join(contexts[supporting_fact]))
            else:
                context_second.append("".join(contexts[supporting_fact]))
        other_contexts = [
            pair[1]
            for pair in self.data[self.current_task_id]["context"]
            if pair[0] not in supporting_fact_titles
        ]
        for other_context in other_contexts:
            judge = random.randint(1, 2)
            if judge == 1:
                context_first.append("".join(other_context))
            else:
                context_second.append("".join(other_context))
        self.current_task_id += 1
        if self.current_task_id >= self.total:
            self.current_task_id = 0
        return question, answer, context_first, context_second


class DataloaderForTrivalQA(BaseDataloader):
    def __init__(
        self,
        dataset: str = "/home/test/test04/yuanjiarui/project/huggingface_cache/trivia_qa_dataset",
        name: str = "rc",
        split: str = "train",
    ):
        self.dataset = dataset
        self.name = name
        self.split = split
        self.train_set = load_dataset(
            path=dataset,
            name=name,
            split=split,
            trust_remote_code=True,
        )
        self.data_type = "qa"
        self.total = len(self.train_set)
        self.current_task_id: int = 0
        self.dataset_name = "trival_qa"

    def sample_once(self):
        data = self.train_set[self.current_task_id]
        search_results: list = data["search_results"]["description"]
        question = data["question"]
        answer = data["answer"]["aliases"]
        context1 = []
        context2 = []
        for i in range(len(search_results)):
            context = random.choice(search_results)
            search_results.remove(context)
            if i % 2 == 0:
                context1.append(context)
            else:
                context2.append(context)
        self.current_task_id += 1
        if self.current_task_id >= self.total:
            self.current_task_id = 0
        return question, answer, context1, context2


class DataloaderForCBT(BaseDataloader):
    def __init__(
        self,
        dataset: str = "/home/test/test04/yuanjiarui/project/huggingface_cache/cbt_dataset",
        name: str = "CN",
        split: str = "train",
    ):
        self.dataset = dataset
        self.name = name
        self.split = split
        self.train_set = load_dataset(
            path=dataset,
            name=name,
            split=split,
            trust_remote_code=True,
        )
        self.data_type = "qa"
        self.total = len(self.train_set)
        self.current_task_id: int = 0
        self.dataset_name = "cbt"

    def sample_once(self):
        data = self.train_set[self.current_task_id]
        options = data["options"]
        question = (
            data["question"] + "\n" + f"Please choose your answer from {options} "
        )
        answer = data["answer"]
        context1 = data["sentences"][: int(len(data["sentences"]) / 2)]
        context2 = data["sentences"][int(len(data["sentences"]) / 2) :]
        self.current_task_id += 1
        if self.current_task_id >= self.total:
            self.current_task_id = 0
        return question, answer, context1, context2


class DataloaderForGSM8K(BaseDataloader):
    def __init__(
        self,
        dataset: str = "/home/test/test04/yuanjiarui/project/huggingface_cache/gsm8k_dataset",
        name: str = "main",
        split: str = "train",
    ):
        self.dataset = dataset
        self.name = name
        self.split = split
        self.train_set = load_dataset(
            path=dataset,
            name=name,
            split=split,
            trust_remote_code=True,
        )
        self.data_type = "math"
        self.dataset_name = "gsm8k"
        self.total = len(self.train_set)
        self.current_task_id: int = 0

    def sample_once(self):
        data = self.train_set[self.current_task_id]
        question = data["question"]
        splited_answer = [answer.strip() for answer in data["answer"].split("####")]
        solving_process = splited_answer[0]
        answer = splited_answer[1].strip()
        self.current_task_id += 1
        if self.current_task_id >= self.total:
            self.current_task_id = 0
        return question, answer, [], []


class DataloaderForMATH(BaseDataloader):
    def __init__(
        self,
        dataset: str = "/home/test/test04/yuanjiarui/project/json_dataset/MATH",
        split="train",
    ):
        self.split = split
        root_path = os.path.join(dataset, split)
        question_types = os.listdir(root_path)
        self.train_set = []
        for question_type in question_types:
            type_path = os.path.join(root_path, question_type)
            questions = os.listdir(type_path)
            for question in questions:
                try:
                    with open(os.path.join(type_path, question), "r") as f:
                        data = json.load(f)
                        self.train_set.append(data)
                except:
                    print(os.path.join(type_path, question))
        self.total = len(self.train_set)
        self.data_type = "math"
        self.dataset_name = "math"
        self.current_task_id = 0

    def extract_boxed_content(self, text):
        start_idx = text.rfind(r"\boxed{")
        if start_idx == -1:
            return None

        start_idx += 7
        brace_count = 1
        end_idx = start_idx

        while brace_count > 0 and end_idx < len(text):
            if text[end_idx] == "{":
                brace_count += 1
            elif text[end_idx] == "}":
                brace_count -= 1
            end_idx += 1

        if brace_count == 0:
            return text[start_idx : end_idx - 1]
        else:
            return None

    def sample_once(self):
        data = self.train_set[self.current_task_id]
        question = data["problem"]
        answer = self.extract_boxed_content(text=data["solution"])
        if answer is not None:
            answer = normalize_final_answer(answer)
        with open("look.txt", "a") as fout:
            fout.write(f"data: {data}\n answer: {answer}\n")
        # print(f"data: {data}\n answer: {answer}")
        # breakpoint()
        self.current_task_id += 1
        if self.current_task_id >= self.total:
            self.current_task_id = 0
        return question, answer, [], []


class DataloaderForARC(BaseDataloader):
    def __init__(
        self,
        dataset: str = "/home/test/test04/yuanjiarui/project/huggingface_cache/arc_dataset",
        name: str = "ARC-Challenge",
        split: str = "train",
    ):
        self.dataset = dataset
        self.name = name
        self.split = split
        self.train_set = load_dataset(
            path=dataset,
            name=name,
            split=split,
            trust_remote_code=True,
        )
        self.data_type = "debate"
        self.total = len(self.train_set)
        self.current_task_id: int = 0
        self.dataset_name = "arc"

    def sample_once(self):
        data = self.train_set[self.current_task_id]
        choices_text = data["choices"]["text"]
        choices_label = data["choices"]["label"]
        label_to_text = {
            label: text for (label, text) in zip(choices_label, choices_text)
        }
        question = (
            data["question"] + "\n" + f"Please choose your answer from {choices_text}"
        )

        answer = label_to_text[data["answerKey"]]

        self.current_task_id += 1
        if self.current_task_id == self.total:
            self.current_task_id = 0

        return question, answer, [], []


class DataloaderForMMLU(BaseDataloader):
    def __init__(
        self,
        dataset: str = "/home/test/test04/yuanjiarui/project/huggingface_cache/mmlu_dataset",
        name: str = "all",
        split: str = "auxiliary_train",
    ):
        self.dataset = dataset
        self.name = name
        self.split = split
        self.train_set = load_dataset(
            path=dataset,
            name=name,
            split=split,
            trust_remote_code=True,
        )
        self.data_type = "debate"
        self.total = len(self.train_set)
        self.current_task_id: int = 0
        self.dataset_name = "mmlu"
        self.arc_filter = DataloaderForARC(split=split.strip("auxiliary_"))
        arc_question_list = [arc["question"] for arc in self.arc_filter.train_set]
        self.train_set = [
            (data["question"], data["subject"], data["choices"], data["answer"])
            for data in self.train_set
            if data["question"] not in arc_question_list
        ]
        local_random = random.Random(42)
        local_random.shuffle(self.train_set)

    def sample_once(self):
        question, subject, choices, answer = self.train_set[self.current_task_id]
        question = (
            question
            + "\n"
            + f"You need to select an answer from the options {choices} to fill in the _."
        )
        answer = choices[answer]

        self.current_task_id += 1
        if self.current_task_id >= self.total:
            self.current_task_id = 0
        return question, answer, [], []


class DataloaderForMix(BaseDataloader):
    def __init__(self, datasets: list = ["mwh_qa", "arc"], splits="train"):
        self.data_type = "mix"
        self.dataset_name = "mix"
        self.datasets = []
        self.total = 0
        self.split = splits
        self.current_task_id = 0
        for dataset in datasets:
            if dataset == "hotpot_qa":
                hotpot_qa_dataset = DataloaderForHotpotQA(
                    split=split_map[splits][dataset]
                )
                self.datasets.append(hotpot_qa_dataset)
                self.total += hotpot_qa_dataset.total
            elif dataset == "mwh_qa":
                mwh_qa_dataset = DataloaderForMWHQA(split=split_map[splits][dataset])
                self.datasets.append(mwh_qa_dataset)
                self.total += mwh_qa_dataset.total
            elif dataset == "trival_qa":
                trival_qa_dataset = DataloaderForTrivalQA(
                    split=split_map[splits][dataset]
                )
                self.datasets.append(trival_qa_dataset)
                self.total += trival_qa_dataset.total
            elif dataset == "cbt":
                cbt_dataset = DataloaderForCBT(split=split_map[splits][dataset])
                self.datasets.append(cbt_dataset)
                self.total += cbt_dataset.total
            elif dataset == "math":
                math_dataset = DataloaderForMATH(split=split_map[splits][dataset])
                self.datasets.append(math_dataset)
                self.total += math_dataset.total
            elif dataset == "gsm8k":
                gsm8k_dataset = DataloaderForGSM8K(split=split_map[splits][dataset])
                self.datasets.append(gsm8k_dataset)
                self.total += gsm8k_dataset.total
            elif dataset == "arc":
                arc_dataset = DataloaderForARC(split=split_map[splits][dataset])
                self.datasets.append(arc_dataset)
                self.total += arc_dataset.total

    def sample_once(self):
        dataset_index = self.current_task_id % len(self.datasets)
        random_dataset = self.datasets[dataset_index]
        question, answer, context1, context2 = random_dataset.sample_once()
        data_type = random_dataset.data_type
        task_id = 0
        for i in range(dataset_index - 1):
            task_id += self.datasets[i].total
        task_id += random_dataset.current_task_id
        self.current_task_id += 1
        if self.current_task_id == self.total:
            self.current_task_id = 0
        return (
            task_id,
            data_type,
            random_dataset.dataset_name,
            question,
            answer,
            context1,
            context2,
        )


def preprocess_dataloader_for_dpo(
    tokenizer=None, output_path: str = None, rawdata_path: str = None
):
    dataset_dict = {"prompt": [], "chosen": [], "rejected": []}
    with open(rawdata_path, "r") as f:
        for line in f:
            data = json.loads(line)
            results = data["results"]
            reward_list = [result["reward"] for result in results]
            chosen_index = int(np.argmax(reward_list))
            rejected_index = int(np.argmin(reward_list))

            dataset_dict["prompt"].append(
                Template(prompt).safe_substitute(
                    {
                        "name": "Alice",
                        "partner": "Bob",
                        "question": results[0]["question"],
                        "information": results[chosen_index]["context_first"],
                    }
                )
            )
            dataset_dict["prompt"].append(
                Template(prompt).safe_substitute(
                    {
                        "name": "Bob",
                        "partner": "Alice",
                        "question": results[0]["question"],
                        "information": results[chosen_index]["context_second"],
                    }
                )
            )

            chosen_conversation = []
            for sentence in results[chosen_index]["conversation"]:
                chosen_conversation.append({"role": "assistant", "content": sentence})

            rejected_conversation = []
            for sentence in results[rejected_index]["conversation"]:
                rejected_conversation.append({"role": "assistant", "content": sentence})
            chosen_conversation[0]["role"] = "system"
            dataset_dict["chosen"].append(chosen_conversation)
            dataset_dict["chosen"].append(chosen_conversation)
            rejected_conversation[0]["role"] = "system"
            dataset_dict["rejected"].append(rejected_conversation)
            dataset_dict["rejected"].append(rejected_conversation)
    train_dict = {
        "prompt": dataset_dict["prompt"][: int(0.9 * len(dataset_dict["prompt"]))],
        "chosen": dataset_dict["chosen"][: int(0.9 * len(dataset_dict["chosen"]))],
        "rejected": dataset_dict["rejected"][
            : int(0.9 * len(dataset_dict["rejected"]))
        ],
    }
    test_dict = {
        "prompt": dataset_dict["prompt"][int(0.9 * len(dataset_dict["prompt"])) :],
        "chosen": dataset_dict["chosen"][int(0.9 * len(dataset_dict["chosen"])) :],
        "rejected": dataset_dict["rejected"][
            int(0.9 * len(dataset_dict["rejected"])) :
        ],
    }
    train = Dataset.from_dict(train_dict)
    test = Dataset.from_dict(test_dict)
    datasetDict = DatasetDict({"train": train, "test": test})
    datasetDict.save_to_disk(output_path)
    return datasetDict


def load_dpo_dataloader(input_path: str):
    dataset = datasets.load_from_disk(input_path)
    return dataset


def process_dataloader_for_sft(
    tokenizer: None,
    output_path: str = None,
    rewarded_data_path: str = None,
    episilon: float = 0.45,
    prompt_type: str = "qa",
):
    """
    Processes a dataset for SFT by selecting data based on rewards,
    generating conversation pairs, and saving them as a dataset.

    Args:
        tokenizer (Tokenizer): Tokenizer used to process the text data (currently not used).
        output_path (str): Path where the processed dataset will be saved.
        rewarded_data_path (str): Path to the input dataset that contains reward annotations.
        episilon (float): Minimum reward threshold to select results.
        prompt_type (str): The type of prompt to use ('qa', 'math', 'debate', etc.).

    Returns:
        DatasetDict: The processed dataset with training and testing splits saved to disk.
    """

    dataset_dict = {"messages": []}
    average_token = 0
    average_reward = 0
    count = 0
    the_prompt = ""
    first_prompt = ""
    second_prompt = ""
    is_debate = False
    is_mix = False
    try:
        with open(rewarded_data_path, "r") as f:
            for line in f:
                data = json.loads(line)
                results = data["results"]
                if results[0]["dataset_name"] == "gsm8k":
                    prompt_type = "gsm8k"
                break
    except:
        pass
    if prompt_type == "qa":
        the_prompt = prompt
        is_debate = False
    elif prompt_type == "math":
        first_prompt = prompt_multi_the_math_first
        second_prompt = prompt_multi_math_second
        is_debate = True
    elif prompt_type == "gsm8k":
        first_prompt = prompt_multi_math_first
        second_prompt = prompt_multi_math_second
        is_debate = True
    elif prompt_type == "debate":
        the_prompt = prompt_multi_debate
        first_prompt = prompt_multi_arc_first
        second_prompt = prompt_multi_arc_second
        is_debate = True
    elif prompt_type == "mix":
        is_mix = True

    with open(rewarded_data_path, "r") as f:
        best_results = []
        if not is_mix:
            for line in tqdm(f):
                data = json.loads(line)
                results = data["results"]
                reward_list = [result["reward"] for result in results]
                index = int(np.argmax(reward_list))
                best_results.append(
                    (
                        reward_list[index],
                        results[index]["conversation"],
                        results[index]["question"],
                        results[index]["context_first"],
                        results[index]["context_second"],
                        results[index]["answer"],
                        results[index]["correct_score"],
                        results[index]["token_score"],
                        results[index]["ppl_score"],
                        prompt_type if not is_mix else results[index]["data_type"],
                        "not mix",
                    )
                )
                if reward_list[index] < episilon:
                    continue
                count += 1
                average_token += results[index]["token_count"]
                average_reward += reward_list[index]
            # sort and select
            best_results.sort(key=lambda x: x[0], reverse=True)
            if prompt_type == "math":
                best_results = best_results[
                    int(0.1 * len(best_results)) : int(0.8 * len(best_results))
                ]
            elif prompt_type == "arc":
                best_results = best_results[
                    int(0.1 * len(best_results)) : int(0.8 * len(best_results))
                ]
            else:
                best_results = best_results[: int(0.7 * len(best_results))]
            best_results = [result for result in best_results if result[0] >= episilon]
        else:
            mix_datasets: dict = {}
            for line in f:
                data = json.loads(line)
                results = data["results"]
                reward_list = [result["reward"] for result in results]
                index = int(np.argmax(reward_list))
                if results[0]["dataset_name"] not in mix_datasets.keys():
                    mix_datasets[results[0]["dataset_name"]] = []
                mix_datasets[results[0]["dataset_name"]].append(
                    (
                        reward_list[index],
                        results[index]["conversation"],
                        results[index]["question"],
                        results[index]["context_first"],
                        results[index]["context_second"],
                        results[index]["answer"],
                        results[index]["correct_score"],
                        results[index]["token_score"],
                        results[index]["ppl_score"],
                        prompt_type if not is_mix else results[index]["data_type"],
                        results[0]["dataset_name"],
                    )
                )
                count += 1
                if reward_list[index] < episilon:
                    continue
            for dataset_name in mix_datasets.keys():
                mix_datasets[dataset_name].sort(key=lambda x: x[0], reverse=True)
                if mix_datasets[dataset_name][0][-1] == "math":
                    mix_datasets[dataset_name] = mix_datasets[dataset_name][
                        int(0.1 * len(mix_datasets[dataset_name])) : int(
                            0.8 * len(mix_datasets[dataset_name])
                        )
                    ]
                elif mix_datasets[dataset_name][0][-1] == "arc":
                    mix_datasets[dataset_name] = mix_datasets[dataset_name][
                        int(0.1 * len(mix_datasets[dataset_name])) : int(
                            0.8 * len(mix_datasets[dataset_name])
                        )
                    ]
                else:
                    mix_datasets[dataset_name] = mix_datasets[dataset_name][
                        : int(0.7 * len(mix_datasets[dataset_name]))
                    ]
                best_results.extend(mix_datasets[dataset_name])
            best_results = [result for result in best_results if result[0] >= episilon]
        # generate dataset
        for (
            reward,
            conversation_list,
            question,
            context_first,
            context_second,
            answer,
            correct_score,
            token_score,
            ppl_score,
            prompt_type,
            dataset_name,
        ) in best_results:
            if prompt_type == "qa":
                the_prompt = prompt
                is_debate = False
            elif prompt_type == "math":
                first_prompt = prompt_multi_math_first
                second_prompt = prompt_multi_math_second
                is_debate = True
            elif prompt_type == "debate":
                the_prompt = prompt_multi_debate
                first_prompt = prompt_multi_arc_first
                second_prompt = prompt_multi_arc_second
                is_debate = True

            prompt_template_first = the_prompt if not is_debate else first_prompt
            conversationA = [
                {
                    "role": "system",
                    "content": Template(prompt_template_first).safe_substitute(
                        {
                            "name": "Alice",
                            "partner": "Bob",
                            "question": question,
                            "information": context_first,
                        },
                    ),
                }
            ]
            prompt_template_second = the_prompt if not is_debate else second_prompt
            conversationB = [
                {
                    "role": "system",
                    "content": Template(prompt_template_second).safe_substitute(
                        {
                            "name": "Bob",
                            "partner": "Alice",
                            "question": question,
                            "information": context_second,
                        },
                    ),
                }
            ]
            for i in range(len(conversation_list)):
                conversationA.append(
                    {"role": "assistant", "content": conversation_list[i]}
                )
                conversationB.append(
                    {"role": "assistant", "content": conversation_list[i]}
                )
            dataset_dict["messages"].append(conversationA)
            dataset_dict["messages"].append(conversationB)
            with open(f"{output_path}_record.jsonl", "a") as fout:
                fout.write(
                    json.dumps(
                        {
                            "conversationA": conversationA,
                            "conversationB": conversationB,
                            "answer": answer,
                            "reward": reward,
                            "correct_score": correct_score,
                            "token_score": token_score,
                            "ppl_score": ppl_score,
                            "dataset_name": dataset_name,
                        }
                    )
                    + "\n"
                )
    messages = dataset_dict["messages"]
    random.shuffle(messages)
    train_dict = {"messages": messages[: int(0.98 * len(messages))]}
    test_dict = {"messages": messages[int(0.98 * len(messages)) :]}

    train = Dataset.from_dict(train_dict)
    test = Dataset.from_dict(test_dict)
    datasetDict = DatasetDict({"train": train, "test": test})
    datasetDict.save_to_disk(output_path)
    print(
        f"average_token: {average_token/count} average_reward: {average_reward/count}  count:{count}"
    )
    return datasetDict


def data_clean(rewarded_data_path, clean_data_path):
    skipping = 0
    record_set = set()
    try:
        with open(clean_data_path, "r") as fin:
            for line in fin:
                data = json.loads(line)
                skipping += 1
                record_set.add(data["task_id"])
    except:
        pass
    print(f"skipping: {skipping}")
    print(f"{record_set}")
    fout = open(clean_data_path, "a")
    with open(rewarded_data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            if data["task_id"] in record_set:
                continue
            results = data["results"]

            for i in range(len(results)):
                conversation_list = results[i]["conversation"]
                # data clean
                for j in range(len(conversation_list)):
                    if re.search(r"Alice:", conversation_list[j]) and re.search(
                        r"Bob:", conversation_list[j]
                    ):
                        data["results"][i]["reward"] -= 10
                        break
                    elif (
                        len(re.findall(r"Alice:", conversation_list[j])) >= 2
                        or len(re.findall(r"Bob:", conversation_list[j])) >= 2
                    ):
                        data["results"][i]["reward"] -= 10
            # save
            fout.write(
                json.dumps({"task_id": data["task_id"], "results": results}) + "\n"
            )
    fout.close()


def process_dpo_format_to_dataset(dpo_format_data_path: str, output_path: str):
    """
    Processes a dataset in DPO format into a structured Hugging Face dataset.

    Args:
        dpo_format_data_path (str): Path to the input dataset in DPO format.
        output_path (str): Path where the processed dataset will be saved.

    Returns:
        DatasetDict: The processed dataset with training and testing splits saved to disk.
    """
    dataset_dict = {"chosen": [], "rejected": []}
    with open(dpo_format_data_path, "r") as fin:
        for line in fin:
            data = json.loads(line)
            results = data["dpo_results"]
            for result in results:
                chosen = result["chosen"]
                rejected = result["rejected"]
                chosen_conversation = [
                    {"role": "assistant", "content": chosen_speech}
                    for chosen_speech in chosen
                ]
                rejected_conversation = [
                    {"role": "assistant", "content": rejected_speech}
                    for rejected_speech in rejected
                ]
                chosen_conversation[0]["role"] = "system"
                rejected_conversation[0]["role"] = "system"
                dataset_dict["chosen"].append(chosen_conversation)
                dataset_dict["rejected"].append(rejected_conversation)
    train_dict = {
        "chosen": dataset_dict["chosen"][: int(0.9 * len(dataset_dict["chosen"]))],
        "rejected": dataset_dict["rejected"][
            : int(0.9 * len(dataset_dict["rejected"]))
        ],
    }
    test_dict = {
        "chosen": dataset_dict["chosen"][int(0.9 * len(dataset_dict["chosen"])) :],
        "rejected": dataset_dict["rejected"][
            int(0.9 * len(dataset_dict["rejected"])) :
        ],
    }
    train = Dataset.from_dict(train_dict)
    test = Dataset.from_dict(test_dict)
    datasetDict = DatasetDict({"train": train, "test": test})
    datasetDict.save_to_disk(output_path)
    return datasetDict
