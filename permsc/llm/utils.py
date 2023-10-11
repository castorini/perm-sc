from typing import List

import tiktoken

from ..data import Message
from ..types import MaybeList, maybe_list_to_list


def max_tokens(model: str) -> int:
    if 'gpt-4' in model:
        return 8192
    elif 'turbo-16k' in model:
        return 16384
    else:
        return 4096


def num_tokens_from_messages(messages: MaybeList[Message], model: str = 'gpt-3.5-turbo') -> int:
    """Returns the number of tokens used by a list of messages."""
    match model:
        case 'gpt-3.5-turbo' | 'gpt-3.5-turbo-16k':
            model = 'gpt-3.5-turbo-0301'
            tokens_per_message = 4
            tokens_per_name = -1  # if there's a name, the role is omitted
        case 'gpt-4':
            model = 'gpt-4-0314'
            tokens_per_message = 3
            tokens_per_name = 1
        case _:
            tokens_per_message, tokens_per_name = 0, 0

    try:
        encoding = tiktoken.get_encoding(model)
    except:
        encoding = tiktoken.get_encoding('cl100k_base')

    num_tokens = 0
    messages, _ = maybe_list_to_list(messages)

    for message in messages:
        num_tokens += tokens_per_message

        for key, value in message.items():
            num_tokens += len(encoding.encode(value))

            if key == 'name':
                num_tokens += tokens_per_name

    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>

    return num_tokens
