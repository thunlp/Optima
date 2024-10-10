from typing import List, Any, Set
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.prompt_template import prompt
from string import Template
import requests
from answerParser.parser import hotpot_qa_parser
from reward.reward import get_score
from utils.utils_token import cal_token
from message.message import llmMessage
import threading
import json
import random
import os
import torch
import gc
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.prompt_template import get_prompt_pool

monte_carlo_file_lock = threading.Lock()
monte_carlo_reward_lock = threading.Lock()


class treeNode:

    def __init__(self):
        self.children = []
        self.content: str = ""
        self.value: float = 0
        self.visitTimes: int = 0
        self.parent: Any = None
        self.nodeType: int = 0
        self.min_child: Any = None
        self.max_child: Any = None

    def add_child(self, new_child):
        self.children.append(new_child)
        new_child.parent = self
        new_child.nodeType = (self.nodeType + 1) % 2

    def update_min_max(self, update_child):
        if self.min_child is None or update_child.value < self.min_child.value:
            self.min_child = update_child
        if self.max_child is None or update_child.value > self.max_child.value:
            self.max_child = update_child
