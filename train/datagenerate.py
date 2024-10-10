from dataloader.dataloader import DataloaderForHotpotQA, load_dpo_dataloader
from agent.agent import Agent, VllmAgent, BaseAgent
from model.llm import Llama3
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from dataloader.dataloader import DataloaderForHotpotQA
from message.message import llmMessage
from answerParser.parser import hotpot_qa_parser, math_parser
from typing import List
from utils.utils_token import cal_token
from utils.prompt_template import *
import json
import random
import numpy as np

import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


max_round = 10
lock = threading.Lock()
prompt_template_list = [prompt]


def data_generate(
    model_path_first: str,
    model_path_second: str,
    device_first: str,
    device_second: str,
    sample_count: int,
    explore_count: int,
    output_path: str,
):
    model_first = AutoModelForCausalLM.from_pretrained(
        model_path_first, torch_dtype="auto", attn_implementation="flash_attention_2"
    ).to(device_first)
    model_second = AutoModelForCausalLM.from_pretrained(
        model_path_second, torch_dtype="auto", attn_implementation="flash_attention_2"
    ).to(device_second)

    tokenizer_first = AutoTokenizer.from_pretrained(
        model_path_first, torch_dtype="auto", attn_implementation="flash_attention_2"
    )
    tokenizer_second = AutoTokenizer.from_pretrained(
        model_path_second, torch_dtype="auto", attn_implementation="flash_attention_2"
    )

    llm_first = Llama3(
        device=device_first, model=model_first, tokenizer=tokenizer_first
    )
    llm_second = Llama3(
        device=device_second, model=model_second, tokenizer=tokenizer_second
    )

    agent_first = Agent(
        llm=llm_first,
        system_prompt=llmMessage(role="system", content=prompt),
        name="Alice",
        prompt_template=prompt,
    )
    agent_second = Agent(
        llm=llm_second,
        system_prompt=llmMessage(role="system", content=prompt),
        name="Bob",
        prompt_template=prompt,
    )

    dataloader = DataloaderForHotpotQA()
    prompt_template_list = [prompt]

    skipping = 0
    with open(output_path, "r") as f:
        for line in f:
            skipping += 1

    for i in range(sample_count):
        question, answer, context1, context2 = dataloader.sample_once()
        if i < skipping:
            continue
        results_list = []
        for j in range(explore_count):
            agent_first.reset()
            agent_second.reset()
            agent_first.init_system_prompt(
                (
                    prompt_template_list[j]
                    if j < len(prompt_template_list)
                    else random.sample(prompt_template_list, 1)[0]
                ),
                {
                    "name": agent_first.name,
                    "partner": agent_second.name,
                    "question": question,
                    "information": "".join(context1),
                },
            )
            agent_second.init_system_prompt(
                (
                    prompt_template_list[j]
                    if j < len(prompt_template_list)
                    else random.sample(prompt_template_list, 1)[0]
                ),
                {
                    "name": agent_second.name,
                    "partner": agent_first.name,
                    "question": question,
                    "information": "".join(context2),
                },
            )
            agent_list = [agent_first, agent_second]
            print(f"----------------answer_{j}: {answer}------------------")
            response_list, final_answer = conversation(agent_list)
            results_list.append(
                {
                    "question": question,
                    "answer": answer,
                    "context_first": context1,
                    "context_second": context2,
                    "conversation": response_list,
                    "final_answer": final_answer,
                    "token_count": cal_token(
                        response_list, [tokenizer_first, tokenizer_second]
                    ),
                }
            )
        with open(output_path, "a") as f:
            f.write(json.dumps({"task_id": i, "results": results_list}) + "\n")


def vllm_data_generate(
    model_first: str,
    model_second: str,
    url_first: str,
    url_second: str,
    tokenizer_path_first: str,
    tokenizer_path_second: str,
    sample_count: int,
    explore_count: int,
    output_path: str,
    thread_count: int,
    prompt_pool_path: str = "utils/prompts.jsonl",
    train_data_path: str = "",
    dataloader=None,
    no_use_prompt_pool: bool = False,
    temperature: float = 0,
    ports: List = [],
    iteration: int = 1,
):
    """
    Generates conversational data by querying two models with prompts.

    Parameters:
        model_first (str): The first model's name or path.
        model_second (str): The second model's name or path.
        url_first (str): API endpoint for the first model.
        url_second (str): API endpoint for the second model.
        tokenizer_path_first (str): Path to the tokenizer for the first model.
        tokenizer_path_second (str): Path to the tokenizer for the second model.
        sample_count (int): Number of samples to generate.
        explore_count (int): Exploration count for variations.
        output_path (str): Path to save the generated output.
        thread_count (int): Number of threads for concurrent processing.
        prompt_pool_path (str, optional): Path to the prompt pool. Default is "utils/prompts.jsonl".
        train_data_path (str, optional): Path to the training data. Default is an empty string.
        dataloader: DataLoader object to sample data from.
        no_use_prompt_pool (bool, optional): Flag to indicate whether to use the prompt pool. Default is False.
        temperature (float, optional): Sampling temperature for the models. Default is 0.
        ports (List, optional): List of ports for the model endpoints. Default is an empty list.
        iteration (int, optional): The iteration count for prompts. Default is 1.

    Returns:
        None
    """
    tokenizer_first = AutoTokenizer.from_pretrained(
        tokenizer_path_first,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
    )
    tokenizer_second = AutoTokenizer.from_pretrained(
        tokenizer_path_second,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
    )
    tokenizer_first.chat_template = """{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}{%- if is_alice %}\n    {{- \'Alice:\' }}\n{%- endif %}\n{%- if is_bob %}\n    {{- \'Bob:\' }}\n{%- endif %}"""
    tokenizer_second.chat_template = """{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}{%- if is_alice %}\n    {{- \'Alice:\' }}\n{%- endif %}\n{%- if is_bob %}\n    {{- \'Bob:\' }}\n{%- endif %}"""
    record_set = set()
    skipping = 0
    try:
        with open(output_path, "r") as f:
            for line in f:
                skipping += 1
                record_set.add(json.loads(line)["task_id"])
    except:
        print("except_output_path")

    if train_data_path != "":
        try:
            with open(train_data_path, "r") as f:
                for line in f:
                    if not json.loads(line)["task_id"] in record_set:
                        skipping += 1
                        record_set.add(json.loads(line)["task_id"])
        except:
            print("except_train_data_path")
    print(f"{skipping} record_set: {record_set}")
    the_prompt = ""
    is_debate = False
    first_prompt = ""
    second_prompt = ""
    if dataloader.data_type == "qa":
        the_prompt = prompt
    elif dataloader.data_type == "math":
        the_prompt = prompt_multi_math
        # no_use_prompt_pool = True
        is_debate = True
        if dataloader.dataset_name == "math":
            first_prompt = prompt_multi_the_math_first
        else:
            first_prompt = prompt_multi_math_first
        second_prompt = prompt_multi_math_second
        if iteration == 0:
            second_prompt = (
                second_prompt
                + """\n 3. You must begin your response with \"${name}:\"."""
            )
            if (
                "You must begin your response with" not in first_prompt
                and "You should start your utterance with" not in first_prompt
            ):
                first_prompt = (
                    first_prompt
                    + """\n 3. You must begin your response with \"${name}:\"."""
                )
    elif dataloader.data_type == "debate":
        the_prompt = prompt_multi_debate
        # no_use_prompt_pool = True
        is_debate = True
        first_prompt = prompt_multi_arc_first
        second_prompt = prompt_multi_arc_second
        if iteration == 0:
            second_prompt = (
                second_prompt
                + """\n 3. You must begin your response with \"${name}:\"."""
            )
            if (
                "You must begin your response with" not in first_prompt
                and "You should start your utterance with" not in first_prompt
            ):
                first_prompt = (
                    first_prompt
                    + """\n 3. You must begin your response with \"${name}:\"."""
                )
    prompt_pool = []
    if not no_use_prompt_pool:
        with open(prompt_pool_path, "r") as fin:
            for line in fin:
                prompt_pool.append(parse_prompt_template(json.loads(line)["prompt"]))
    args_list = []
    if (
        dataloader.data_type != "qa"
        and dataloader.data_type != "mix"
        and dataloader.split == "test"
    ):
        sample_count = dataloader.total
    for i in range(0, sample_count):
        if dataloader.data_type != "mix":
            question, answer, context1, context2 = dataloader.sample_once()
            if i in record_set:
                continue
            args_list.append(
                (
                    i,
                    question,
                    answer,
                    context1,
                    context2,
                    the_prompt,
                    is_debate,
                    first_prompt,
                    second_prompt,
                    dataloader.data_type,
                    prompt_pool,
                    dataloader.dataset_name,
                )
            )
        else:
            task_id, data_type, dataset_name, question, answer, context1, context2 = (
                dataloader.sample_once()
            )
            if i in record_set:
                continue
            if data_type == "qa":
                the_prompt = prompt
                prompt_pool_path = "/home/test/test04/yuanjiarui/project/src/utils/prompts_diverse.jsonl"
            elif data_type == "math":
                the_prompt = prompt_multi_math
                no_use_prompt_pool = True
                is_debate = True
                first_prompt = prompt_multi_math_first
                second_prompt = prompt_multi_math_second
            elif data_type == "debate":
                the_prompt = prompt_multi_debate
                is_debate = True
                prompt_pool_path = "/home/test/test04/yuanjiarui/project/src/utils/prompts_arc_first.jsonl"
                first_prompt = prompt_multi_arc_first
                second_prompt = prompt_multi_arc_second
            prompt_pool = []
            if not no_use_prompt_pool:
                with open(prompt_pool_path, "r") as fin:
                    for line in fin:
                        prompt_pool.append(
                            parse_prompt_template(json.loads(line)["prompt"])
                        )
            args_list.append(
                (
                    task_id,
                    question,
                    answer,
                    context1,
                    context2,
                    the_prompt,
                    is_debate,
                    first_prompt,
                    second_prompt,
                    data_type,
                    prompt_pool,
                    dataset_name,
                )
            )

    print(f"task count: {len(args_list)} -- {sample_count}")
    port = np.random.choice(ports, len(args_list))
    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        futures = [
            executor.submit(
                vllm_data_generate_once,
                task_id,
                explore_count,
                model_first,
                model_second,
                (
                    f"http://localhost:{port[idx]}/v1/chat/completions"
                    if len(ports) > 0
                    else url_first
                ),
                (
                    f"http://localhost:{port[idx]}/v1/chat/completions"
                    if len(ports) > 0
                    else url_second
                ),
                question_,
                answer_,
                context1_,
                context2_,
                tokenizer_first,
                tokenizer_second,
                output_path,
                prompt_pool_,
                the_prompt_,
                no_use_prompt_pool,
                temperature,
                is_debate_,
                first_prompt_,
                second_prompt_,
                data_type,
                dataset_name_,
            )
            for idx, (
                task_id,
                question_,
                answer_,
                context1_,
                context2_,
                the_prompt_,
                is_debate_,
                first_prompt_,
                second_prompt_,
                data_type,
                prompt_pool_,
                dataset_name_,
            ) in enumerate(args_list)
        ]

        for future in as_completed(futures):
            print(f"----------------")


def random_prompt_template(prompt_file: str):
    key_list = ["prompt", "prompt_json", "prompt_no_polite", "prompt_markdown_table"]
    key = random.choice(key_list)
    with open(prompt_file, "r") as f:
        prompts = json.load(f)
        prompt_list = prompts[key]
        prompt_template = random.choice(prompt_list)
        return prompt_template


def vllm_data_generate_once(
    task_id: int,
    explore_count: int,
    model_first: str,
    model_second: str,
    url_first: str,
    url_second: str,
    question: str,
    answer,
    context1: list,
    context2: list,
    tokenizer_first,
    tokenizer_second,
    output_path: str,
    prompt_pool: list,
    prompt: str,
    no_use_prompt_pool: bool = False,
    temperature: float = 0,
    is_debate: bool = False,
    first_prompt: str = "",
    second_prompt: str = "",
    data_type: str = "qa",
    dataset_name: str = "hotpot_qa",
):
    print(task_id)
    agent_first = VllmAgent(
        url=url_first, my_model_name=model_first, name="Alice", temperature=temperature
    )
    agent_second = VllmAgent(
        url=url_second, my_model_name=model_second, name="Bob", temperature=temperature
    )
    results_list = []
    for j in range(explore_count):
        first_prompt_template = ""
        second_prompt_template = ""
        prompt_template = ""
        if not no_use_prompt_pool:
            prompt_template = random.choice(prompt_pool)
            if is_debate:
                prompt_template = ""
                first_prompt_template = random.choice(prompt_pool)
                second_prompt_template = second_prompt
        else:
            prompt_template = prompt
            if is_debate:
                prompt_template = ""
                first_prompt_template = first_prompt
                second_prompt_template = second_prompt

        agent_first.reset()
        agent_second.reset()
        agent_first.init_system_prompt(
            prompt_template if not is_debate else first_prompt_template,
            {
                "name": agent_first.name,
                "partner": agent_second.name,
                "question": question,
                "information": "\n".join(context1),
            },
        )
        agent_second.init_system_prompt(
            prompt_template if not is_debate else second_prompt_template,
            {
                "name": agent_second.name,
                "partner": agent_first.name,
                "question": question,
                "information": "\n".join(context2),
            },
        )
        agent_list = [agent_first, agent_second]
        print(
            f"...........................{answer}....................................."
        )
        response_list, final_answer = conversation(
            agent_list, tokenizer_first, question, data_type
        )

        score_type = "f1-score"
        if data_type == "debate":
            score_type = "exact-match"
        elif data_type == "math":
            score_type = "exact-match"

        results_list.append(
            {
                "question": question,
                "prompt": prompt_template,
                "first_prompt": first_prompt_template,
                "second_prompt": second_prompt_template,
                "context_first": context1,
                "context_second": context2,
                "conversation": response_list,
                "answer": answer,
                "final_answer": final_answer,
                "token_count": cal_token(
                    response_list, [tokenizer_first, tokenizer_second]
                ),
                "score_type": score_type,
                "data_type": data_type,
                "dataset_name": dataset_name,
            }
        )
    with lock:
        with open(output_path, "a") as f:

            f.write(json.dumps({"task_id": task_id, "results": results_list}) + "\n")


def conversation(agent_list: List[BaseAgent], tokenizer, question, data_type="qa"):
    final_answer = ""
    current_agent_id = 0
    response_list = []
    now_round = 0
    while True:
        if now_round >= max_round:
            break
        agent = agent_list[current_agent_id]

        response = agent.step()
        if len(tokenizer.tokenize(response.content)) >= 2000:
            response_list.append("large")
            break
        content: str = response.content
        if data_type != "qa" and (not (content.strip().startswith(f"{agent.name}"))):
            content = f"{agent.name}:{content}"
        response_list.append(content)
        print(
            f"""
            --------------------------------------------\n
            {content}
            --------------------------------------------\n
            """
        )
        if data_type == "math":
            tmp_answer = math_parser(response)
        else:
            tmp_answer = hotpot_qa_parser(response)
        print(f"tmp_answer: {tmp_answer}")
        if tmp_answer != None and tmp_answer != "":
            if tmp_answer == final_answer:
                break
            else:
                final_answer = tmp_answer
        current_agent_id = (current_agent_id + 1) % 2
        agent_list[current_agent_id].add_memory(response)
        now_round += 1
    return response_list, final_answer


def find_all_linear_names(model):
    pattern = r"\((\w+)\): Linear"
    linear_layers = re.findall(pattern, str(model.modules))
    target_modules = list(set(linear_layers))
    return target_modules
