import numpy as np
from trl import DataCollatorForCompletionOnlyLM
from transformers import DataCollatorForLanguageModeling
import warnings
from typing import Union, List, Optional, Dict, Any

# mask token
you_are_Alice_token = [128006, 9125, 128007, 271, 2675, 527, 30505]
you_are_Bob_token = [128006, 9125, 128007, 271, 2675, 527, 14596]
Alice_token = [128006, 78191, 128007, 271, 62786, 25]
Bob_token = [128006, 78191, 128007, 271, 33488, 25]


class Mask(DataCollatorForLanguageModeling):
    """
    Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only
    calculated on the completion made by the assistant.

    Args:
        response_template (`Union[str, List[int]]`): the template form that indicates the start of the response, typically something like
            '### Response:\n'. It can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.
        instruction_template (`Union[str, List[int]]`): the template form that indicates the start of the human instruction, typically something like
            '### Human:\n'. Useful for assistant-style conversation datasets. It can also be passed as tokenized ids.
        mlm (`bool`, *optional*, defaults to `False`): Whether or not to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
    """

    def __init__(
        self,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.ignore_index = ignore_index

    def torch_call(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        batch = super().torch_call(examples)
        for i in range(len(examples)):
            response_token_ids_idxs = []
            human_token_ids_idxs = []

            # import pdb
            # pdb.set_trace()

            for assistant_idx in np.where(batch["labels"][i] == you_are_Alice_token[0])[
                0
            ]:
                if (
                    you_are_Alice_token
                    == batch["labels"][i][
                        assistant_idx : assistant_idx + len(you_are_Alice_token)
                    ].tolist()
                ):
                    self.response_token_ids = Alice_token
                    self.instruction_token_ids = Bob_token
            for assistant_idx in np.where(batch["labels"][i] == you_are_Bob_token[0])[
                0
            ]:
                if (
                    you_are_Bob_token
                    == batch["labels"][i][
                        assistant_idx : assistant_idx + len(you_are_Bob_token)
                    ].tolist()
                ):
                    self.response_token_ids = Bob_token
                    self.instruction_token_ids = Alice_token

            for assistant_idx in np.where(
                batch["labels"][i] == self.response_token_ids[0]
            )[0]:
                # find the indexes of the start of a response.
                if (
                    self.response_token_ids
                    == batch["labels"][i][
                        assistant_idx : assistant_idx + len(self.response_token_ids)
                    ].tolist()
                ):
                    response_token_ids_idxs.append(
                        assistant_idx + len(self.response_token_ids)
                    )

            human_token_ids = self.instruction_token_ids
            for human_idx in np.where(batch["labels"][i] == human_token_ids[0])[0]:
                # find the indexes of the start of a human answer.
                if (
                    human_token_ids
                    == batch["labels"][i][
                        human_idx : human_idx + len(human_token_ids)
                    ].tolist()
                ):
                    human_token_ids_idxs.append(human_idx)

            if (
                len(human_token_ids_idxs) > 0
                and len(response_token_ids_idxs) > 0
                and human_token_ids_idxs[0] > response_token_ids_idxs[0]
            ):
                human_token_ids_idxs = [0] + human_token_ids_idxs

            for idx, (start, end) in enumerate(
                zip(human_token_ids_idxs, response_token_ids_idxs)
            ):
                # Make pytorch loss function ignore all non response tokens
                if idx != 0:
                    batch["labels"][i, start:end] = self.ignore_index
                else:
                    batch["labels"][i, :end] = self.ignore_index

            if len(response_token_ids_idxs) < len(human_token_ids_idxs):
                batch["labels"][i, human_token_ids_idxs[-1] :] = self.ignore_index

        return batch
