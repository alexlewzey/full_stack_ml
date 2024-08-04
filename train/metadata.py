"""Cats vs Dogs metadata."""
from types import MappingProxyType

ENCODING = MappingProxyType(
    {
        "dog": 0,
        "cat": 1,
    }
)
DECODING = tuple(ENCODING.keys())
