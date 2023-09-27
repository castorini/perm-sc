__all__ = ['Item', 'RankingExample', 'Message']

from copy import deepcopy
from typing import List

from pydantic import BaseModel


class Item(BaseModel):
    content: str
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

    def merge(self, other_example: 'RankingExample'):
        if other_example.metadata.get('orig_indices'):  # perform non-appending merge
            a, b = other_example.metadata['orig_indices']
            self.hits = self.hits[:a] + other_example.hits + self.hits[b:]
        else:
            self.hits += other_example.hits


class Message(BaseModel):
    role: str
    content: str
