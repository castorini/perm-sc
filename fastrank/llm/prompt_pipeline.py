__all__ = ['OpenAIPromptPipeline']

import numpy as np

from .prompt_builder import RankingPromptBuilder
from .openai_pool import ChatCompletionPool
from ..data import RankingExample, Message
from ..types import MaybeList, maybe_list_to_list


class OpenAIPromptPipeline:
    def __init__(self, prompt_builder: RankingPromptBuilder, pool: ChatCompletionPool):
        self.builder = prompt_builder
        self.pool = pool

    def run(self, examples: MaybeList[RankingExample]) -> MaybeList[np.ndarray]:
        """Produces an array of preferences over the hits in `example`."""
        examples, converted = maybe_list_to_list(examples)
        prompts = [[x.model_dump() for x in self.builder.make_prompt(example)] for example in examples]
        num_items_list = [len(example.hits) for example in examples]

        responses = self.pool.create_batch(messages=prompts)
        results = []

        for response, num_items in zip(responses, num_items_list):
            message = Message(**response['choices'][0]['message'])
            results.append(self.builder.extract_preferences(message.content, num_items=num_items))

        return results[0] if converted else results
