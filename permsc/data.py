__all__ = ['Item', 'RankingExample', 'Message', 'RankingDataset', 'MathSortDataset', 'GSM8KSortDataset',
           'WordSortDataset', 'CountrySortDataset']

from copy import deepcopy
import json
from pathlib import Path
from typing import List, Any, Dict, Iterable

import nltk
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

        # Update current permutation
        new_perm = np.empty(len(self.hits[key]), dtype=int)
        sort_idx = np.argsort(metadata['current_permutation'][key])
        new_perm[sort_idx] = np.arange(key.stop - key.start)
        metadata['current_permutation'] = new_perm

        return RankingExample(hits=deepcopy(self.hits[key]), query=deepcopy(self.query), metadata=metadata)

    @property
    def current_permutation(self) -> np.ndarray:
        return self.metadata.get('current_permutation', np.arange(len(self.hits)))

    def to_pyserini_dict(self) -> Dict[str, Any]:
        return {
            'query': self.query.content,
            'hits': [{'content': hit.content, 'score': hit.score, 'rank': hit.metadata.get('rank', 0)} for hit in self.hits]
        }

    @classmethod
    def from_pyserini_dict(cls, d: Dict[str, Any]) -> 'RankingExample':
        hits = [Item(content=h['content'], score=h['score'], metadata={'rank': h['rank']}) for h in d['hits']]
        query = Item(content=d['query'])

        return cls(hits=hits, query=query)

    def sort_by(self, key, standardize: bool = False, reverse: bool = False) -> np.ndarray:
        perm_mask = np.argsort([key(hit) for hit in self.hits])

        if reverse:
            perm_mask = perm_mask[::-1]

        return self.randomize_order(perm_mask=perm_mask, standardize=standardize)

    def randomize_order(self, standardize: bool = False, perm_mask: np.ndarray = None) -> np.ndarray:
        perm_mask = np.random.permutation(len(self.hits)) if perm_mask is None else perm_mask
        self.metadata['current_permutation'] = self.metadata.get('current_permutation', np.arange(len(self.hits)))[perm_mask]
        self.hits = np.array(self.hits, dtype=object)[perm_mask].tolist()
        perm_mask = self.metadata['current_permutation']

        if standardize and 'current_permutation' in self.metadata:
            self.metadata['current_permutation'] = np.arange(len(self.hits))

        return perm_mask

    def split(self, split_size: int) -> Iterable['RankingExample']:
        for i in range(0, len(self), split_size):
            try:
                yield self[i:i + split_size]
            except ValueError:
                break

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

    def __len__(self) -> int:
        return len(self.hits)


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

        return RankingExample(hits=hits)


class GSM8KSortDataset(RankingDataset):
    def __init__(self, path: str):
        self.question_sents_list = []

        for line in Path(path).read_text().splitlines():
            data = json.loads(line)
            sentences = nltk.tokenize.sent_tokenize(data['question'])

            if len(sentences) < 5 or len(sentences) > 10:
                continue

            self.question_sents_list.append(sentences)

    def __len__(self):
        return len(self.question_sents_list)

    def load_example(self, idx: int) -> RankingExample:
        sentences = self.question_sents_list[idx]
        hits = [Item(content=sent, score=1 / (idx + 1)) for idx, sent in enumerate(sentences)]

        return RankingExample(hits=hits)


class WordSortDataset(RankingDataset):
    def __init__(self, path: str):
        df = pd.read_csv(path, sep='\t', quoting=3, escapechar='\\')
        df['word_samples'] = df.word_samples.apply(lambda x: json.loads(x))
        self.df = df

    def __len__(self):
        return len(self.df)

    def load_example(self, idx: int) -> RankingExample:
        words = sorted(self.df.iloc[idx].word_samples)
        hits = [Item(content=word, score=1 / (idx + 1)) for idx, word in enumerate(words)]

        return RankingExample(hits=hits)


class CountrySortDataset(RankingDataset):
    def __init__(self, path: str):
        df = pd.read_csv(path, sep='\t', quoting=3, escapechar='\\')
        self.dfs = [x[1] for x in df.groupby('key')]

    def __len__(self):
        return len(self.dfs)

    def load_example(self, idx: int) -> RankingExample:
        df = self.dfs[idx]
        hits = []
        keys = ['percentage', 'number', 'year', 'rate']

        value = json.loads(df['value'].iloc[0])
        rel_key = None

        for key in keys:
            if key in value:
                rel_key = key
                break

        assert rel_key is not None, f'No relevant key found in {value}'

        for _, row in df.iterrows():
            value = json.loads(row['value'])
            hits.append(Item(content=row['country'], score=value[rel_key], metadata=dict(prompt=row['prompt'])))

        return RankingExample(hits=sorted(hits, key=lambda x: x.score, reverse=True))


class Message(BaseModel):
    role: str
    content: str
