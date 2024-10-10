from typing import List


def cal_token(conversation: List, tokenizer_list):
    token_count = 0
    for i, sentence in enumerate(conversation):
        tokens = tokenizer_list[i % 2].tokenize(sentence)
        token_count += len(tokens)
    return token_count
