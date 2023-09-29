__all__ = ['Item', 'RankingExample', 'Message', 'StackExchangeDataset', 'RankingDataset', 'OpenAISummarizationDataset',
           'MathSortDataset']

from copy import deepcopy
from typing import List
import unicodedata

import datasets
import markdownify
import numpy as np
import pandas as pd
from pydantic import BaseModel


class Item(BaseModel):
    content: str
    score: float = 0
    id: str = ''
    metadata: dict = {}


class RankingExample(BaseModel):
    hits: List[Item]
    query: Item = None
    metadata: dict = {}

    def __getitem__(self, key) -> 'RankingExample':
        if isinstance(key, int):
            key = slice(key, key + 1)

        assert isinstance(key, slice), 'RankingExample can only be sliced with int or slice'
        assert key.step is None or key.step == 1, 'Slicing with step is not supported'

        metadata = self.metadata.copy()
        metadata['orig_indices'] = key

        return RankingExample(hits=deepcopy(self.hits[key]), query=deepcopy(self.query), metadata=metadata)

    def randomize_order(self) -> np.ndarray:
        perm_mask = np.random.permutation(len(self.hits))
        self.metadata['current_permutation'] = self.metadata.get('current_permutation', np.arange(len(self.hits)))[perm_mask]
        self.hits = np.array(self.hits, dtype=object)[perm_mask].tolist()

        return self.metadata['current_permutation']

    def restore_order(self):
        perm_mask = self.metadata.get('current_permutation', np.arange(len(self.hits)))
        self.hits = np.array(self.hits, dtype=object)[np.argsort(perm_mask)].tolist()
        del self.metadata['current_permutation']

    def permuted_preferences_to_original_order(self, preferences: np.ndarray) -> np.ndarray:
        """Converts preference arrays over the permuted items to preference arrays over the original order."""
        perm_mask = self.metadata.get('current_permutation', np.arange(len(self.hits)))
        pref_restore_map = dict(zip(range(len(perm_mask)), perm_mask))
        pref_restore_map[-1] = -1

        return np.array([pref_restore_map[x] for x in preferences])

    def merge(self, other_example: 'RankingExample'):
        if other_example.metadata.get('orig_indices'):  # perform non-appending merge
            a, b = other_example.metadata['orig_indices']
            self.hits = self.hits[:a] + other_example.hits + self.hits[b:]
        else:
            self.hits += other_example.hits


class RankingDataset:
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx: int | slice) -> RankingExample:
        converted = False

        if isinstance(idx, int):
            idx = slice(idx, idx + 1)
            converted = True

        examples = [self.load_example(i) for i in range(idx.start or 0, idx.stop or len(self), idx.step or 1)]

        return examples[0] if converted else examples

    def load_example(self, idx: int) -> RankingExample:
        raise NotImplementedError

    def __iter__(self):
        return (self[i] for i in range(len(self)))


class StackExchangeDataset(RankingDataset):
    def __init__(self, path: str):
        self.path = path
        self.dataset = datasets.load_from_disk(path)

    def _clean(self, text: str) -> str:
        md = markdownify.markdownify(text)
        md = unicodedata.normalize('NFKC', md)

        return md

    def load_example(self, idx: int) -> RankingExample:
        ex = self.dataset[idx]
        hits = [Item(content=self._clean(answer['text']), id=str(answer['answer_id']), score=answer['pm_score']) for answer in ex['answers']]
        hits.sort(key=lambda x: x.score, reverse=True)
        query = Item(content=self._clean(ex['question']))

        return RankingExample(hits=hits, query=query)

    def __len__(self):
        return len(self.dataset)


class OpenAISummarizationDataset(RankingDataset):
    def __init__(self, path: str):
        self.path = path
        ds = datasets.load_from_disk(path)
        df = ds.to_pandas()
        df['post_id'] = [x['info']['id'] for _, x in df.iterrows()]
        self.dfs = [grouped_df for _, grouped_df in df.groupby('post_id')]

    def _clean(self, text: str) -> str:
        return unicodedata.normalize('NFKC', text)

    def load_example(self, idx: int) -> RankingExample:
        ex = self.dfs[idx]
        answers = ex['summary']
        hits = [Item(content=self._clean(x['text']), score=x['axes']['accuracy']) for x in answers]
        hits.sort(key=lambda x: x.score, reverse=True)
        query = Item(content=self._clean(ex['info'].iloc[0]['post']))

        return RankingExample(hits=hits, query=query)

    def __len__(self):
        return len(self.dfs)


class MathSortDataset(RankingDataset):
    def __init__(self, path: str):
        df = pd.read_csv(path, sep='\t', quoting=3, escapechar='\\')
        self.dfs = [df for _, df in df.groupby('group')]

    def __len__(self):
        return len(self.dfs)

    def load_example(self, idx: int) -> RankingExample:
        df = self.dfs[idx]
        exprs = []

        for _, row in df.iterrows():
            exprs.append((row['expression'], row['answer']))

        exprs.sort(key=lambda x: x[1])
        hits = [Item(content=expr, score=1 / (idx + 1)) for idx, (expr, _) in enumerate(exprs)]

        return RankingExample(hits=hits, query=Item(content='0'))


class Message(BaseModel):
    role: str
    content: str
