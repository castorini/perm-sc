# Permutation Self-Consistency

This is the official Python library for the permutation self-consistency method.
Described in [the paper](), our self-consistency-inspired approach improves
listwise ranking in black-box large language models (LLMs) at inference time.
Like the original [self-consistency decoding strategy](), our core algorithm
comprises two main stages:
1. **Sample**: we collect a sample of LLM output rankings by randomly shuffling the input list in the prompt. 
2. **Aggregate**: we combine these rankings into the one that minimizes the Kendall tau distance to all rankings.

## Getting Started

### Installation
If you only need the library, run
```
pip install permsc
```

If you want to develop the code or need the sorting datasets (see `./data/`) from the paper, clone the repository:
```
git clone ssh://git@github.com/castorini/perm-sc
```

### Using the Library

## Paper Replication

## Citation
```

```
