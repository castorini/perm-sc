__all__ = ['maybe_list_to_list']

from typing import TypeVar, List, Tuple

T = TypeVar('T')
MaybeList = List[T] | T


def maybe_list_to_list(x: MaybeList[T]) -> Tuple[List[T], bool]:
    """
    If `x` is a single element, it becomes a single-element list. Otherwise, the function is a no-op.

    Returns:
         A tuple whose first element is the potentially converted list and the second a flag denoting conversion.
    """
    if not isinstance(x, list):
        return [x], True

    return x, False
