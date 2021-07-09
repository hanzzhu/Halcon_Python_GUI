"""
************************************************************
mixin.py - interface specific functionality
************************************************************

Project: HALCON/Python

************************************************************

(c) 1996-2020 by MVTec Software GmbH

Software by: MVTec Software GmbH, www.mvtec.com
"""


__all__ = [
    'to_handle',
    'upcast_obj_list'
]

import ctypes

from typing import Sequence

# Full path import because another module split would be inherently circular.
import halcon


# --- Exported Functions ---


def to_handle(out_handle_ptr: ctypes.c_void_p) -> 'halcon.HHandle':
    return halcon.HHandle(out_handle_ptr)


def upcast_obj_list(iconic_objects: Sequence['halcon.ffi.HObjectBase']) -> None:
    # TODO doc string.
    # TODO single ffi call.
    assert len(iconic_objects) >= 1

    result_obj = iconic_objects[0]
    for iconic_object in iconic_objects[1:]:
        result_obj = halcon.concat_obj(result_obj, iconic_object)

    return result_obj
