from typing import List, Dict
from pydantic import BaseModel
from message.message import llmMessage
from model.llm import BaseLLM
from string import Template
import requests


class BaseAgent:

    def no_memory_step(self):
        pass

    def step(self):
        pass

    def init_system_prompt(self, template: str, args: dict):
        pass

    def add_memory(self):
        pass

    def reset(self):
        pass


class Agent(BaseAgent):
    llm: BaseLLM = None
    prompt_template: str = ""
    system_prompt: llmMessage = llmMessage(role="system", content="")
    memory: List[llmMessage] = []
    name: str = ""

    def step(self) -> llmMessage:
        message_input = [
            {"role": message.role, "content": message.content}
            for message in self.memory
        ]
        response = self.llm.generate_response(message_input, self.name)

        self.add_memory(response)

        return response

    def init_system_prompt(self, template: str, args: dict):
        self.system_prompt.content = Template(template).safe_substitute(args)
        self.memory.append(self.system_prompt)

    def add_memory(self, new_memory: llmMessage):
        self.memory.append(new_memory)

    def reset(self):
        self.memory = []
        self.system_prompt = llmMessage(role="system", content=self.prompt_template)


class VllmAgent(BaseAgent):
    """
    The agent class is based on VLLM.
    It handles communication , manages the conversation context (memory),
    and formats the input/output in the required structure.
    """

    def __init__(self, url: str, my_model_name: str, name: str, temperature: float):
        self.url = url
        self.prompt_template = ""
        self.system_prompt: llmMessage = llmMessage(role="system", content="")
        self.memory: List[llmMessage] = []
        self.my_model_name = my_model_name
        self.name = name
        self.temperature = temperature

    def init_system_prompt(self, template: str, args: dict):
        self.system_prompt.content = Template(template).safe_substitute(args)
        self.memory.append(self.system_prompt)

    def add_memory(self, new_memory: llmMessage):
        self.memory.append(new_memory)

    def reset(self):
        self.memory = []
        self.system_prompt = llmMessage(role="system", content=self.prompt_template)

    # step and update memory
    def step(self):
        message_input = [
            {"role": message.role, "content": message.content}
            for message in self.memory
        ]

        headers = {"Content-Type": "application/json"}
        is_iteration_0 = False
        if (
            "You should start your utterance with" in self.system_prompt.content
            or "You must begin your response with" in self.system_prompt.content
        ):
            is_iteration_0 = True
        data_json = {
            "model": self.my_model_name,
            "messages": message_input,
            "temperature": self.temperature,
            "chat_template": """{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}{%- if is_alice %}\n    {{- \'Alice:\' }}\n{%- endif %}\n{%- if is_bob %}\n    {{- \'Bob:\' }}\n{%- endif %}""",
            "chat_template_kwargs": {
                "is_alice": (self.name == "Alice") and (not is_iteration_0),
                "is_bob": (self.name == "Bob") and (not is_iteration_0),
            },
            "max_tokens": 2000,
        }
        response = requests.post(self.url, headers=headers, json=data_json)
        if response.status_code == 400:
            return llmMessage(role="assistant", content="error")
        content: str = response.json()["choices"][0]["message"]["content"]
        if not content.startswith(self.name) and not is_iteration_0:
            content = f"{self.name}:{content}"
        response = llmMessage(
            role="assistant",
            content=content,
        )
        self.add_memory(response)

        return response

    # step but don't update memory
    def no_memory_step(self):
        message_input = [
            {"role": message.role, "content": message.content}
            for message in self.memory
        ]

        headers = {"Content-Type": "application/json"}
        data_json = {
            "model": self.my_model_name,
            "messages": message_input,
            "temperature": self.temperature,
            "max_tokens": 2000,
        }

        response = requests.post(self.url, headers=headers, json=data_json)
        if response.status_code == 400:
            return llmMessage(role="assistant", content="error")
        response = llmMessage(
            role="assistant",
            content=response.json()["choices"][0]["message"]["content"],
        )

        return response
