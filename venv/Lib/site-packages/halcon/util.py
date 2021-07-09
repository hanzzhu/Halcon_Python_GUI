"""
************************************************************
util.py - general purpose utility
************************************************************

Project: HALCON/Python

Description:
Utility not specific to the HALCON domain.

************************************************************

(c) 1996-2020 by MVTec Software GmbH

Software by: MVTec Software GmbH, www.mvtec.com
"""


from typing import List, Union, Sequence, TypeVar


__all__ = [
    'slice_to_index_list',
    'MaybeSequence',
]


_T1 = TypeVar('_T1')

MaybeSequence = Union[_T1, Sequence[_T1]]


def slice_to_index_list(index: slice, length: int) -> List[int]:
    """Return list of indices with same semantics as list slicing."""

    # ['a', 'b', 'c', 'd', 'e']
    #   0    1    2    3    4
    #  -4   -3   -2   -1    0

    return list(range(length)[index])
