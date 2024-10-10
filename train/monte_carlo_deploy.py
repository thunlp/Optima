from typing import List, Any, Set
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.prompt_template import *
from string import Template
import requests
from answerParser.parser import hotpot_qa_parser, math_parser

from utils.utils_token import cal_token
from message.message import llmMessage
import threading
import json
import random
import numpy as np
import time

from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from utils.prompt_template import get_prompt_pool
from reward.deploy_reward import get_score_deploy
from train.monte_carlo import treeNode

import editdistance
import multiprocessing

monte_carlo_file_lock_deploy = threading.Lock()
monte_multi_lock = multiprocessing.Lock()


class MonteCarloTreeDeploy:
    """
    Monte Carlo Tree Deployment for conversational agents.

    This class implements a Monte Carlo Tree Search (MCTS) approach to facilitate dialogue generation
    between two agents, leveraging a provided language model. It explores different conversation paths
    and evaluates the outcomes based on various scoring metrics to optimize the responses.

    Attributes:
        model_url (str): URL endpoint for the language model.
        root (treeNode): The root node of the search tree.
        contexts (list): Contexts for each agent in the conversation.
        question (str): The initial question posed to the agents.
        answer (str): Expected answer for evaluation.
        names (list): Names of the agents involved in the conversation.
        reward_highest_node (treeNode): The node with the highest reward during search.
        model (str): Name of the language model used.
        tokenizer (str): Tokenizer for the language model.
        my_model_name (str): Model name for deployment.
        max_depth (int): Maximum depth for the search tree.
        task_id (int): Identifier for the task being processed.
        max_token_count (int): Maximum token limit for responses.
        explore_time (int): Number of iterations for exploring the search tree.
        results (list): List to store results of the dialogue generation.
        dpo_results (list): List to store decision policy optimization results.
        format_results (list): Formatted results for output.
        valuable_nodes (set): Set of valuable nodes identified during search.
        min_value (float): Minimum value threshold for node evaluation.
        incremental_threshold (float): Threshold for evaluating improvements in values.
        cal_ppl (bool): Flag to indicate whether to calculate perplexity.
        score_type (str): Type of score to be used for evaluation (e.g., "f1-score").
        lambda1 (float): Weight parameter for scoring.
        lambda2 (float): Weight parameter for scoring.
        the_prompt (str): The initial prompt for dialogue.
        is_debate (bool): Flag indicating whether the conversation is a debate.
        debate_prompts (list): Specific prompts used if the conversation is a debate.
        data_type (str): Type of task being performed (e.g., "qa", "math").

    Methods:
        generate(): Runs the MCTS process to generate dialogue responses.
        search(): Explores the search tree and selects nodes based on their values.
        expand(node, origin_conversation_list): Expands the conversation tree from a given node.
        back_propagation(node, conversation_list, final_answer): Updates node values based on rewards and backpropagates results through the tree.
    """

    def __init__(
        self,
        _model_url: str,
        _question: str,
        _answer: str,
        context1: List[str],
        context2: List[str],
        model: str,
        tokenizer: str,
        _my_model_name: str,
        _max_depth: int,
        _task_id: int,
        _explore_time: int,
        _prompt_pool: list,
        _min_value: float = 0.15,
        _incremental_threshold: float = 0.3,
        _max_token: int = 111,
        cal_ppl: bool = True,
        score_type: str = "f1-score",
        lambda1: float = -0.6,
        lambda2: float = 1,
        the_prompt: str = prompt,
        is_debate: bool = False,
        first_prompt="",
        second_prompt="",
        data_type="qa",
    ):
        self.prompt_pool = _prompt_pool
        self.model_url = _model_url
        self.root = treeNode()
        self.the_prompt = the_prompt
        self.root.content = self.the_prompt
        self.contexts = []
        self.contexts.append(context1)
        self.contexts.append(context2)
        self.question = _question
        self.answer = _answer
        self.names = ["Alice", "Bob"]
        self.reward_highest_node = self.root
        self.model = model
        self.tokenizer = tokenizer
        self.my_model_name = _my_model_name
        self.max_depth = _max_depth
        self.task_id = _task_id
        self.max_token_count = _max_token
        self.explore_time = _explore_time
        self.results = []
        self.dpo_results = []
        self.format_results = []
        self.valuable_nodes: Set[treeNode] = set()
        self.min_value = _min_value
        self.incremental_threshold = _incremental_threshold
        self.cal_ppl = cal_ppl
        self.score_type = score_type
        self.all_nodes: List[treeNode] = [self.root]
        self.selected_nodes: Set[treeNode] = set()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.is_debate = is_debate
        self.debate_prompts = [first_prompt, second_prompt]
        if is_debate:
            self.root.content = first_prompt
        self.data_type = data_type

    def generate(self):
        for i in range(self.explore_time):
            start_time = time.time()
            self.search()
        self.all_nodes.extend(self.selected_nodes)
        for node in self.all_nodes:
            if len(node.children) >= 2:
                self.valuable_nodes.add(node)
        valuable_record = []
        for i, valuable_node in enumerate(self.valuable_nodes):
            valuable_record.append({"node_id": i, "all_conversation": []})
            for child in valuable_node.children:
                conversation_list = []
                tmpNode = child
                now_depth = 0
                while tmpNode != self.root:
                    conversation_list.append(tmpNode.content)
                    tmpNode = tmpNode.parent
                    now_depth += 1
                nodeType = now_depth % 2
                conversation_list.append(
                    Template(
                        self.root.content
                        if not self.is_debate
                        else self.debate_prompts[nodeType]
                    ).safe_substitute(
                        {
                            "name": self.names[nodeType],
                            "partner": self.names[(nodeType + 1) % 2],
                            "information": self.contexts[nodeType],
                            "question": self.question,
                        }
                    ),
                )
                conversation_list = conversation_list[::-1]
                valuable_record[i]["all_conversation"].append(
                    {"conversation": conversation_list, "value": child.value}
                )

            if (
                valuable_node.max_child.value > self.min_value
                and valuable_node.max_child.value - valuable_node.min_child.value
                > self.incremental_threshold
            ):
                conversation_list = []
                tmpNode = valuable_node
                now_depth = 0
                while tmpNode != self.root:
                    conversation_list.append(tmpNode.content)
                    tmpNode = tmpNode.parent
                    now_depth += 1
                nodeType = now_depth % 2
                conversation_list.append(
                    Template(
                        self.root.content
                        if not self.is_debate
                        else self.debate_prompts[nodeType]
                    ).safe_substitute(
                        {
                            "name": self.names[nodeType],
                            "partner": self.names[(nodeType + 1) % 2],
                            "information": self.contexts[nodeType],
                            "question": self.question,
                        }
                    ),
                )
                conversation_list = conversation_list[::-1]
                conversation_list.append(valuable_node.max_child.content)
                result = {}
                result["chosen"] = conversation_list
                reject_conversation_list = conversation_list.copy()
                reject_conversation_list[-1] = valuable_node.min_child.content
                result["rejected"] = reject_conversation_list
                result["chosen_value"] = valuable_node.max_child.value
                result["distance"] = (
                    valuable_node.max_child.value - valuable_node.min_child.value
                )
                self.dpo_results.append(result)

        if (
            self.root.max_child.value - self.root.min_child.value
            > self.incremental_threshold
            and self.root.max_child.value > self.min_value
        ):
            result = {}
            conversation_list = []
            conversation_list.append(
                Template(
                    self.root.content if not self.is_debate else self.debate_prompts[0]
                ).safe_substitute(
                    {
                        "name": self.names[nodeType],
                        "partner": self.names[(nodeType + 1) % 2],
                        "information": self.contexts[nodeType],
                        "question": self.question,
                    }
                ),
            )
            conversation_list.append(self.root.max_child.content)
            result["chosen"] = conversation_list
            # conversation_list.pop()
            reject_conversation_list = conversation_list.copy()
            reject_conversation_list[-1] = self.root.min_child.content
            result["rejected"] = reject_conversation_list
            self.format_results = [result]

        self.dpo_results.sort(key=lambda x: x["chosen_value"], reverse=True)
        if int(0.7 * len(self.dpo_results)) >= 1:
            self.dpo_results = self.dpo_results[: int(0.5 * len(self.dpo_results))]

        return (
            {"task_id": self.task_id, "results": self.results},
            {
                "task_id": self.task_id,
                "dpo_results": self.dpo_results,
            },
            {"task_id": self.task_id, "dpo_results": self.format_results},
            {
                "task_id": self.task_id,
                "question": self.question,
                "context_first": self.contexts[0],
                "context_second": self.contexts[1],
                "record": valuable_record,
            },
        )

    def search(self):
        start_time = time.time()
        candidate_nodes = [
            node
            for node in self.all_nodes
            if (
                node == self.root
                or (len(node.children) > 0 and len(node.children[0].children) > 0)
            )
        ]
        tmp_selected_nodes = set()
        for node in candidate_nodes:
            for ref_node in self.selected_nodes:
                judge = editdistance.eval(node.content, ref_node.content)
                if (judge / (max(len(node.content), len(ref_node.content)) + 1)) < 0.25:
                    tmp_selected_nodes.add(node)
                    break
        for node in tmp_selected_nodes:
            candidate_nodes.remove(node)
            self.all_nodes.remove(node)
        if len(candidate_nodes) > 0:
            candidate_nodes.sort(key=lambda x: x.value, reverse=True)
            candidate_nodes_value = [
                node.value for node in candidate_nodes[: min(len(self.all_nodes), 10)]
            ]
            exp_list = np.exp(candidate_nodes_value)
            exp_list = exp_list / np.sum(exp_list)
            distribution = np.cumsum(exp_list)
            random_number = np.random.rand()
            index = np.where(distribution >= random_number)[0][0]
            self.reward_highest_node = candidate_nodes[index]
        else:
            self.reward_highest_node = self.root

        mid_node = self.reward_highest_node
        self.selected_nodes.add(mid_node)
        nodeType = self.reward_highest_node.nodeType
        conversation_list = []
        while mid_node != self.root:
            conversation_list.append({"role": "assistant", "content": mid_node.content})
            mid_node = mid_node.parent
        conversation_list.append(
            {
                "role": "system",
                "content": Template(
                    self.root.content if not self.is_debate else self.debate_prompts[0]
                ).safe_substitute(
                    {
                        "name": self.names[nodeType],
                        "partner": self.names[(nodeType + 1) % 2],
                        "information": self.contexts[nodeType],
                        "question": self.question,
                    }
                ),
            }
        )
        conversation_list = conversation_list[::-1]
        self.expand(self.reward_highest_node, conversation_list)

    def expand(self, node: treeNode, origin_conversation_list: List[dict]):
        headers = {"Content-Type": "application/json"}

        for _ in range(3):
            midNode = node
            conversation_list = origin_conversation_list.copy()
            now_depth = len(conversation_list) - 1
            if self.data_type != "math":
                finalAnswer = hotpot_qa_parser(
                    llmMessage(
                        role="assistant", content=conversation_list[-1]["content"]
                    )
                )
            else:
                finalAnswer = math_parser(
                    llmMessage(
                        role="assistant", content=conversation_list[-1]["content"]
                    )
                )
            is_qa = False
            if (
                "You should start your utterance with" in self.the_prompt
                or "You must begin your response with" in self.the_prompt
            ):
                is_qa = True
            while True:
                if now_depth >= self.max_depth:
                    self.back_propagation(midNode, conversation_list, "error")
                    break
                now_depth += 1
                data_json = {
                    "model": self.my_model_name,
                    "messages": conversation_list,
                    "temperature": 0.7,
                    "chat_template": """{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}{%- if is_alice %}\n    {{- \'Alice:\' }}\n{%- endif %}\n{%- if is_bob %}\n    {{- \'Bob:\' }}\n{%- endif %}""",
                    "chat_template_kwargs": {
                        "is_alice": (
                            self.names[midNode.nodeType] == "Alice" and not is_qa
                        ),
                        "is_bob": (self.names[midNode.nodeType] == "Bob" and not is_qa),
                    },
                    "max_tokens": 2000,
                }
                start_time = time.time()
                response = requests.post(
                    self.model_url, headers=headers, json=data_json
                )
                response = response.json()["choices"][0]["message"]["content"]
                if not response.startswith(self.names[midNode.nodeType]):
                    response = f"{self.names[midNode.nodeType]}:{response}"
                print(
                    f"""
                    ---------------------------------
                    {response}  
                    ---------------------------------  
                    """
                )
                skip = False
                for child in midNode.children:
                    if (
                        editdistance.eval(child.content, response) / (len(response) + 1)
                        < 0.1
                    ):
                        midNode = child
                        skip = True
                        break
                if not skip:
                    newNode = treeNode()
                    newNode.content = response
                    midNode.add_child(newNode)
                    self.all_nodes.append(newNode)
                    midNode = newNode

                conversation_list[0] = {
                    "role": "system",
                    "content": Template(
                        self.the_prompt
                        if not self.is_debate
                        else self.debate_prompts[midNode.nodeType]
                    ).safe_substitute(
                        {
                            "name": self.names[midNode.nodeType],
                            "partner": self.names[(midNode.nodeType + 1) % 2],
                            "information": self.contexts[midNode.nodeType],
                            "question": self.question,
                        }
                    ),
                }
                conversation_list.append({"role": "assistant", "content": response})
                if self.data_type != "math":
                    tempAnswer = hotpot_qa_parser(
                        llmMessage(
                            role="assistant", content=conversation_list[-1]["content"]
                        )
                    )
                else:
                    tempAnswer = math_parser(
                        llmMessage(
                            role="assistant", content=conversation_list[-1]["content"]
                        )
                    )
                if tempAnswer is not None and tempAnswer != "":
                    if tempAnswer == finalAnswer:
                        self.back_propagation(midNode, conversation_list, finalAnswer)
                        break
                    else:
                        finalAnswer = tempAnswer

    def back_propagation(
        self, node: treeNode, conversation_list: List[dict], final_answer: str
    ):

        conversation = [speech["content"] for speech in conversation_list][1:]
        start_time = time.time()
        token_count = cal_token(conversation, [self.tokenizer, self.tokenizer])

        result = {
            "question": self.question,
            "answer": self.answer,
            "context_first": self.contexts[0],
            "context_second": self.contexts[1],
            "conversation": conversation,
            "final_answer": final_answer,
            "token_count": token_count,
        }

        start_time = time.time()
        value, rouge_l, ppl_score = get_score_deploy(
            result=result,
            max_token_count=self.max_token_count,
            tokenizer=self.tokenizer,
            model_url=self.model,
            cal_ppl=self.cal_ppl,
            _lambda1=self.lambda1,
            _lambda2=self.lambda2,
            score_type=self.score_type,
        )
        result["reward"] = value
        result["rouge_l"] = rouge_l
        result["ppl_score"] = ppl_score
        self.results.append(result)

        while node is not None:
            node.visitTimes += 1
            node.value = (node.value * (node.visitTimes - 1) + value) / node.visitTimes

            if node.parent is not None:
                node.parent.update_min_max(node)
            node = node.parent


def monte_carlo_data_generate_once_deploy(
    max_token: int,
    model_url: str,
    question: str,
    answer: str,
    context_first: List[str],
    context_second: List[str],
    model: Any,
    tokenizer: Any,
    my_model_name: str,
    max_depth: int,
    task_id: int,
    explore_time: int,
    output_path: str,
    prompt_pool: list,
    min_value: float = 0.15,
    incremental_threshold: float = 0.3,
    cal_ppl: bool = True,
    score_type: str = "f1-score",
    lambda1: float = -0.6,
    lambda2: float = 1,
    the_prompt: str = prompt,
    is_debate: bool = False,
    first_prompt: str = "",
    second_prompt: str = "",
    data_type: str = "qa",
):
    tree = MonteCarloTreeDeploy(
        model_url,
        question,
        answer,
        context_first,
        context_second,
        model,
        tokenizer,
        my_model_name,
        max_depth,
        task_id,
        explore_time,
        prompt_pool,
        _min_value=min_value,
        _incremental_threshold=incremental_threshold,
        _max_token=max_token,
        cal_ppl=cal_ppl,
        score_type=score_type,
        lambda1=lambda1,
        lambda2=lambda2,
        the_prompt=the_prompt,
        is_debate=is_debate,
        first_prompt=first_prompt,
        second_prompt=second_prompt,
        data_type=data_type,
    )

    result, dpo_result, structure_result, record_result = tree.generate()

    with monte_carlo_file_lock_deploy:
        try:
            with open(output_path, "a") as fout:
                fout.write(json.dumps(result) + "\n")
            with open(f"{output_path[:-6]}_dpo_format.jsonl", "a") as fout2:
                fout2.write(json.dumps(dpo_result) + "\n")
            with open(f"{output_path[:-6]}_dpo_structure.jsonl", "a") as fout3:
                fout3.write(json.dumps(structure_result) + "\n")
            with open(f"{output_path[:-6]}_dpo_record.jsonl", "a") as fout4:
                fout4.write(json.dumps(record_result) + "\n")
        except Exception as e:
            print(e)


def monte_carlo_data_generate_once_deploy_lock(
    max_token: int,
    model_url: str,
    question: str,
    answer: str,
    context_first: List[str],
    context_second: List[str],
    model: Any,
    tokenizer: Any,
    my_model_name: str,
    max_depth: int,
    task_id: int,
    explore_time: int,
    output_path: str,
    prompt_pool: list,
    min_value: float = 0.15,
    incremental_threshold: float = 0.3,
    cal_ppl: bool = True,
    score_type: str = "f1-score",
    lambda1: float = -0.6,
    lambda2: float = 1,
    the_prompt: str = prompt,
    is_debate: bool = False,
    first_prompt: str = "",
    second_prompt: str = "",
    data_type: str = "qa",
):
    tree = MonteCarloTreeDeploy(
        model_url,
        question,
        answer,
        context_first,
        context_second,
        model,
        tokenizer,
        my_model_name,
        max_depth,
        task_id,
        explore_time,
        prompt_pool,
        _min_value=min_value,
        _incremental_threshold=incremental_threshold,
        _max_token=max_token,
        cal_ppl=cal_ppl,
        score_type=score_type,
        lambda1=lambda1,
        lambda2=lambda2,
        the_prompt=the_prompt,
        is_debate=is_debate,
        first_prompt=first_prompt,
        second_prompt=second_prompt,
        data_type=data_type,
    )

    result, dpo_result, structure_result, record_result = tree.generate()

    with monte_multi_lock:
        with open(output_path, "a") as fout:
            fout.write(json.dumps(result) + "\n")
        with open(f"{output_path[:-6]}_dpo_format.jsonl", "a") as fout1:
            fout1.write(json.dumps(dpo_result) + "\n")
        with open(f"{output_path[:-6]}_dpo_structure.jsonl", "a") as fout1:
            fout1.write(json.dumps(structure_result) + "\n")
        with open(f"{output_path[:-6]}_dpo_record.jsonl", "a") as fout1:
            fout1.write(json.dumps(record_result) + "\n")


def monte_carlo_data_generate_deploy(
    max_token: int,
    model: str,
    tokenizer_path: str,
    sft_data_path: str,
    output_path: str,
    sample_count: int,
    num_thread: int,
    dataloader,
    prompt_pool_path: str,
    model_url: str = "http://0.0.0.0:8000/v1/chat/completions",
    min_value: float = 0.15,
    incremental_threshold: float = 0.3,
    ports: List = [],
    cal_ppl: bool = True,
    score_type: str = "f1-score",
    lambda1: float = -0.6,
    lambda2: float = 1,
):
    record_set = set()
    skipping = 0
    try:
        with open(sft_data_path, "r") as f:
            for line in f:
                data = json.loads(line)
                record_set.add(data["task_id"])
                skipping += 1
    except:
        pass
    try:
        with open(output_path, "r") as f:
            for line in f:
                data = json.loads(line)
                if data["task_id"] not in record_set:
                    record_set.add(data["task_id"])
                    skipping += 1
    except:
        pass
    args_list = []
    print(record_set)
    for i in range(sample_count):
        question, answer, context_first, context_second = dataloader.sample_once()
        if i in record_set:
            continue
        args_list.append((i, question, answer, context_first, context_second))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    prompt_pool = get_prompt_pool(prompt_pool_path)

    the_prompt = ""
    is_debate = False
    first_prompt = ""
    second_prompt = ""
    if dataloader.data_type == "qa":
        the_prompt = prompt
    elif dataloader.data_type == "math":
        is_debate = True
        if dataloader.dataset_name == "gsm8k":
            first_prompt = prompt_multi_math_first
        else:
            first_prompt = prompt_multi_the_math_first
        second_prompt = prompt_multi_math_second
    elif dataloader.data_type == "debate":
        is_debate = True
        first_prompt = prompt_multi_arc_first
        second_prompt = prompt_multi_arc_second

    if score_type != "math":
        with ThreadPoolExecutor(num_thread) as executor:
            futures = [
                executor.submit(
                    monte_carlo_data_generate_once_deploy,
                    max_token,
                    (
                        f"http://localhost:{random.choice(ports)}/v1/chat/completions"
                        if len(ports) > 0
                        else model_url
                    ),
                    _question,
                    _answer,
                    _context_first,
                    _context_second,
                    model,
                    tokenizer,
                    "Llama-3",
                    10,
                    _task_id,
                    8,
                    output_path,
                    prompt_pool,
                    min_value,
                    incremental_threshold,
                    cal_ppl,
                    score_type,
                    lambda1,
                    lambda2,
                    the_prompt,
                    is_debate,
                    first_prompt,
                    second_prompt,
                    dataloader.data_type,
                )
                for (
                    _task_id,
                    _question,
                    _answer,
                    _context_first,
                    _context_second,
                ) in args_list
            ]
            for future in as_completed(futures):
                print("-------------------------------------------")
    else:
        with ProcessPoolExecutor(max_workers=num_thread) as executor:
            futures = [
                executor.submit(
                    monte_carlo_data_generate_once_deploy_lock,
                    max_token,
                    (
                        f"http://localhost:{random.choice(ports)}/v1/chat/completions"
                        if len(ports) > 0
                        else model_url
                    ),
                    _question,
                    _answer,
                    _context_first,
                    _context_second,
                    model,
                    tokenizer,
                    "Llama-3",
                    10,
                    _task_id,
                    8,
                    output_path,
                    prompt_pool,
                    min_value,
                    incremental_threshold,
                    cal_ppl,
                    score_type,
                    lambda1,
                    lambda2,
                    the_prompt,
                    is_debate,
                    first_prompt,
                    second_prompt,
                    dataloader.data_type,
                )
                for (
                    _task_id,
                    _question,
                    _answer,
                    _context_first,
                    _context_second,
                ) in args_list
            ]
            for future in as_completed(futures):
                print("-------------------------------------------")
