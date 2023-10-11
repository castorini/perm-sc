__all__ = ['RankingPromptBuilder', 'RelevanceRankingPromptBuilder', 'AnswerRankingPromptBuilder',
           'SummarizationRankingPromptBuilder', 'MathSortRankingPromptBuilder', 'FastMathSortRankingPromptBuilder',
           'SentenceSortRankingPromptBuilder', 'WordSortRankingPromptBuilder']

from copy import deepcopy
import re
from typing import List

import editdistance
import numpy as np

from ..data import Message, Item, RankingExample


class RankingPromptBuilder:
    def __init__(self, max_item_length: int = 300):
        self.max_item_length = max_item_length
        self.rank_offset = 1

    def make_prefix_prompt(self, example: RankingExample) -> List[Message]:
        return []

    def make_post_prompt(self, example: RankingExample) -> List[Message]:
        return []

    def make_body_prompt(self, item: Item, rank: int) -> List[Message]:
        return []

    def make_all_body_prompt(self, example: RankingExample) -> List[Message]:
        return []

    def process_item(self, item: Item) -> Item:
        max_length = self.max_item_length
        content = item.content
        content = content.replace('Title: Content: ', '')
        content = content.strip()
        # For Japanese should cut by character: content = content[:int(max_length)]
        content = ' '.join(content.split()[:int(max_length)])
        item.content = content

        return item

    def extract_preferences(self, example: RankingExample, response: str) -> np.ndarray:
        """Extract a preference array from the response."""
        ranking = {}
        num_items = len(example)
        orig_rankings = set(range(num_items))

        for m in re.finditer(r'\d+', response):
            rank = min(num_items - 1, int(m.group(0)) - self.rank_offset)

            try:
                orig_rankings.remove(rank)
            except KeyError:
                pass

            ranking[rank] = None

        return np.array(list(ranking.keys()) + [-1] * len(orig_rankings))

    def make_prompt(self, example: RankingExample) -> List[Message]:
        messages = self.make_prefix_prompt(example)
        messages += self.make_all_body_prompt(example)

        for rank, item in enumerate(example.hits):
            rank += 1
            item = self.process_item(deepcopy(item))
            messages += self.make_body_prompt(item, rank)

        messages += self.make_post_prompt(example)

        return messages


class RelevanceRankingPromptBuilder(RankingPromptBuilder):
    """The same prompt builder used in RankGPT."""
    def make_prefix_prompt(self, example: RankingExample) -> List[Message]:
        num = len(example.hits)
        query = example.query.content

        return [Message(role='system', content='You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query.'),
                Message(role='user', content=f'I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}.'),
                Message(role='assistant', content='Okay, please provide the passages.')]

    def make_post_prompt(self, example: RankingExample) -> List[Message]:
        query = example.query.content
        num = len(example.hits)

        text = f'Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages ' \
        f'should be listed in descending order using identifiers. The most relevant passages should be listed first. ' \
        f'The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain.'

        return [Message(role='user', content=text)]

    def make_body_prompt(self, item: Item, rank: int) -> List[Message]:
        messages = []
        messages.append(Message(role='user', content=f'[{rank}] {item.content}'))
        messages.append(Message(role='assistant', content=f'Received passage [{rank}].'))

        return messages


class AnswerRankingPromptBuilder(RankingPromptBuilder):
    def make_prefix_prompt(self, example: RankingExample) -> List[Message]:
        num = len(example.hits)
        query = example.query.content

        return [Message(role='system', content='You are RankGPT, an intelligent assistant that can rank answer passages based on many upvotes they will get.'),
                Message(role='user', content=f'I will provide you with {num} answer passages, each indicated by number identifier []. \nRank the passages based on how many upvotes they will get for the question passage: {query}.'),
                Message(role='assistant', content='Okay, please provide the answer passages.')]

    def make_post_prompt(self, example: RankingExample) -> List[Message]:
        query = example.query.content
        num = len(example.hits)

        text = f'Question passage: {query}. \n\nRank the {num} answer passages above based on how many upvotes they will get for answering the question passage. The answers ' \
        f'should be listed in descending order using identifiers. The most upvoted answer passages should be listed first. ' \
        f'The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain.'

        return [Message(role='user', content=text)]

    def make_body_prompt(self, item: Item, rank: int) -> List[Message]:
        messages = []
        messages.append(Message(role='user', content=f'[{rank}] {item.content}'))
        messages.append(Message(role='assistant', content=f'Received passage [{rank}].'))

        return messages


class SummarizationRankingPromptBuilder(RankingPromptBuilder):
    def make_prefix_prompt(self, example: RankingExample) -> List[Message]:
        num = len(example.hits)
        query = example.query.content

        return [Message(role='system', content='You are an intelligent assistant that can rank summaries based on their accuracy of the given passage.'),
                Message(role='user', content=f'I will provide you with {num} summaries, each indicated by number identifier []. \nRank the summaries based on their accuracy for summarizing the passage: {query}.'),
                Message(role='assistant', content='Okay, please provide the summaries.')]

    def make_post_prompt(self, example: RankingExample) -> List[Message]:
        query = example.query.content
        num = len(example.hits)

        text = f'Summarize this passage: {query}. \nRank the {num} summaries above based on the sumarization accuracy for the given passage. The summaries ' \
        f'should be listed in descending order using identifiers. The most accurate summaries should be listed first. ' \
        f'The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain.'

        return [Message(role='user', content=text)]

    def make_body_prompt(self, item: Item, rank: int) -> List[Message]:
        messages = []
        messages.append(Message(role='user', content=f'[{rank}] {item.content}'))
        messages.append(Message(role='assistant', content=f'Received summary [{rank}].'))

        return messages


class MathSortRankingPromptBuilder(RankingPromptBuilder):
    def make_prefix_prompt(self, example: RankingExample) -> List[Message]:
        num = len(example.hits)
        query = example.query.content

        return [Message(role='system', content='You are an intelligent assistant that can evaluate and sort math expressions based on their value.'),
                Message(role='user', content=f'I will provide you with {num} math expressions, each indicated by number identifier []. \nRank them based on their value.'),
                Message(role='assistant', content='Okay, please provide the expressions.')]

    def make_post_prompt(self, example: RankingExample) -> List[Message]:
        query = example.query.content
        num = len(example.hits)

        text = f'Sort the {num} expressions above in increasing order. The math expressions ' \
        f'should be listed in order from smallest to biggest using identifiers. The smallest expressions should be listed first. ' \
        f'The output format should be [] < [], e.g., [2] < [1]. Only response the ranking results, do not say any word or explain.'

        return [Message(role='user', content=text)]

    def make_body_prompt(self, item: Item, rank: int) -> List[Message]:
        messages = []
        messages.append(Message(role='user', content=f'[{rank}] {item.content}'))
        messages.append(Message(role='assistant', content=f'Received expression [{rank}].'))

        return messages


class FastMathSortRankingPromptBuilder(RankingPromptBuilder):
    def extract_preferences(self, example: RankingExample, response: str) -> np.ndarray:
        """Extract a preference array from the response."""
        item_map = {hit.content: idx for idx, hit in enumerate(example.hits)}
        num_items = len(example)

        ranking = {}
        orig_rankings = set(range(num_items))
        response = response.split('\n')[-1]
        clean_patt = re.compile(r'[^\d\+\-\*\/\s]')

        for item in response.split(','):
            item = clean_patt.sub('', item).strip()

            try:
                rank = min(num_items - 1, item_map[item])
            except KeyError:
                continue

            try:
                orig_rankings.remove(rank)
            except KeyError:
                pass

            ranking[rank] = None

        return np.array(list(ranking.keys()) + [-1] * len(orig_rankings))

    def make_prefix_prompt(self, example: RankingExample) -> List[Message]:
        return [Message(role='system', content='You are an intelligent assistant that can evaluate and sort math expressions based on their value.')]

    def make_all_body_prompt(self, example: RankingExample) -> List[Message]:
        expressions = ', '.join(item.content for item in example.hits)
        messages = [
            Message(role='user', content=f'Sort the following expressions from smallest to largest: {expressions}.\nThe output format should be a comma-separated list containing the exact expressions; do not reduce them. Only respond with the results; do not say any word or explain.'),
        ]

        return messages


class SentenceSortRankingPromptBuilder(RankingPromptBuilder):
    def extract_preferences(self, example: RankingExample, response: str) -> np.ndarray:
        """Extract a preference array from the response."""
        gt_items = [hit.content for hit in example.hits]
        num_items = len(example)

        ranking = {}
        orig_rankings = set(range(num_items))
        response = max(response.split('\n\n'), key=len)  # For LLaMA-2, which doesn't listen
        pred_items = response.split('\n')[-num_items:]

        for item in pred_items:
            try:
                rank = gt_items.index(item)
            except ValueError:
                rank = min(enumerate(gt_items), key=lambda x: editdistance.eval(x[1], item))[0]

            try:
                orig_rankings.remove(rank)
            except KeyError:
                pass

            ranking[rank] = None

        return np.array(list(ranking.keys()) + [-1] * len(orig_rankings))

    def make_prefix_prompt(self, example: RankingExample) -> List[Message]:
        return [Message(role='system', content='You are an intelligent assistant that can order sentences.')]

    def make_all_body_prompt(self, example: RankingExample) -> List[Message]:
        input_ = '\n- '.join(item.content for item in example.hits)
        input_ = input_.strip()

        messages = [
            Message(role='user', content=f'Order the scrambled sentences logically:\n{input_}\nThe output format should have each sentence on a new line. Only respond with the results; do not say any word or explain.'),
        ]

        return messages


class WordSortRankingPromptBuilder(SentenceSortRankingPromptBuilder):
    def make_prefix_prompt(self, example: RankingExample) -> List[Message]:
        return [Message(role='system', content='You are an intelligent assistant that can order words alphabetically.')]

    def make_all_body_prompt(self, example: RankingExample) -> List[Message]:
        input_ = '\n- '.join(item.content for item in example.hits)
        input_ = input_.strip()

        messages = [
            Message(role='user', content=f'Order these words alphabetically:\n{input_}\nThe output format should have each word on a new line. Only respond with the results; do not say any word or explain.'),
        ]

        return messages
