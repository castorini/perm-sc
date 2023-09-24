from ._version import *
from .aggregator import *
from .utils import *

try:
    import beir  # Try to import BEIR
    from .llm import *
except ImportError:
    import logging
    logging.warning('fastrank[llm] dependencies not installed. Some auxiliary modules will not be available.')
