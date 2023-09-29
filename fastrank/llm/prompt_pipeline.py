__all__ = ['OpenAIPromptPipeline']

import numpy as np

from .prompt_builder import RankingPromptBuilder
from .openai_pool import ChatCompletionPool
from ..data import RankingExample, Message
from ..types import MaybeList, maybe_list_to_list
from .utils import num_tokens_from_messages, max_tokens


class OpenAIPromptPipeline:
    def __init__(self, prompt_builder: RankingPromptBuilder, pool: ChatCompletionPool):
        self.builder = prompt_builder
        self.pool = pool

    def run(self, examples: MaybeList[RankingExample], **kwargs) -> MaybeList[np.ndarray]:
        """Produces an array of preferences over the hits in `example`."""
        examples, converted = maybe_list_to_list(examples)
        prompts_list = []

        for example in examples:
            prompts = [x.model_dump() for x in self.builder.make_prompt(example)]

            if num_tokens_from_messages(prompts) > max_tokens(self.pool.model_name) - 200:
                self.builder.max_item_length -= 1
                continue

            self.builder.max_item_length = 300
            prompts_list.append(prompts)

        responses = self.pool.create_batch(messages=prompts_list, **kwargs)
        num_items_list = [len(example.hits) for example in examples]
        results = []

        for response, num_items in zip(responses, num_items_list):
            message = Message(**response['choices'][0]['message'])
            results.append(self.builder.extract_preferences(message.content, num_items=num_items))

        return results[0] if converted else results
