from typing import List, Dict, Union, Any
import torch
from pydantic import BaseModel
from abc import abstractmethod
from message.message import llmMessage
from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


class BaseLLM(BaseModel):
    device: str = "cuda:1"
    model: Any = None
    tokenizer: Any = None
    max_new_tokens: int = 1000
    do_sample: bool = True
    temperature: float = 0

    @abstractmethod
    def generate_response(self, messages: List[Dict]) -> dict:
        pass


class Llama3(BaseLLM):
    def generate_response(self, messages: List[Dict], name: str = "") -> dict:
        messages_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if name != "":
            messages_input += name + ":"
        messages_input = self.tokenizer(messages_input, return_tensors="pt")[
            "input_ids"
        ].to(self.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        with torch.inference_mode():
            response = self.model.generate(
                messages_input,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                eos_token_id=terminators,
            )
        response = self.tokenizer.batch_decode(response)
        response = (
            response[0].split("<|end_header_id|>")[-1].strip().strip("<|eot_id|>")
        )
        return llmMessage(
            role="assistant",
            content=response,
        )
