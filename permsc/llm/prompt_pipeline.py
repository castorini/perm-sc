__all__ = ['OpenAIPromptPipeline', 'FastRankAPIPromptPipeline']

from typing import Dict, Any, List
import urllib.parse

import numpy as np
import pydantic
import requests
from tqdm import tqdm

from .prompt_builder import RankingPromptBuilder
from .openai_pool import ChatCompletionPool
from ..data import RankingExample, Message
from ..types import MaybeList, maybe_list_to_list
from .utils import num_tokens_from_messages, max_tokens


class PromptPipeline:
    def __init__(self, prompt_builder: RankingPromptBuilder, model_name: str):
        self.builder = prompt_builder
        self.model_name = model_name

    def produce_responses(self, prompts_list: List[List[Dict[str, Any]]], **kwargs) -> List[Any]:
        raise NotImplementedError

    def run(self, examples: MaybeList[RankingExample], **kwargs) -> MaybeList[np.ndarray]:
        """Produces an array of preferences over the hits in `example`."""
        examples, converted = maybe_list_to_list(examples)
        prompts_list = []

        for example in examples:
            prompts = [x.dict() for x in self.builder.make_prompt(example)]

            if num_tokens_from_messages(prompts) > max_tokens(self.model_name) - 200:
                self.builder.max_item_length -= 1
                continue

            self.builder.max_item_length = 300
            prompts_list.append(prompts)

        responses = self.produce_responses(prompts_list, **kwargs)
        results = []

        for response, example in zip(responses, examples):
            try:
                message = Message(**response)
            except pydantic.ValidationError:
                message = Message(role='assistant', content='')

            results.append(self.builder.extract_preferences(example, message.content))

        return results[0] if converted else results


class OpenAIPromptPipeline(PromptPipeline):
    def __init__(self, prompt_builder: RankingPromptBuilder, pool: ChatCompletionPool):
        super().__init__(prompt_builder, pool.model_name)
        self.pool = pool

    def produce_responses(self, prompts_list: List[List[Dict[str, Any]]], **kwargs) -> List[Any]:
        return [response['choices'][0]['message'] for response in self.pool.create_batch(messages=prompts_list, **kwargs)]


class FastRankAPIPromptPipeline(PromptPipeline):
    def __init__(self, prompt_builder: RankingPromptBuilder, endpoint: str = 'http://falcon:8008', model_name: str = 'llama-2-13b'):
        super().__init__(prompt_builder, model_name)
        self.endpoint = urllib.parse.urljoin(endpoint, f'/model/{model_name}/generate/')

    def produce_responses(self, prompts_list: List[List[Dict[str, Any]]], **kwargs) -> List[Any]:
        responses = []

        for prompts in tqdm(prompts_list):
            response = requests.post(self.endpoint, json=dict(messages=prompts)).json()
            responses.append(dict(role='assistant', content=response['response']))

        return responses
