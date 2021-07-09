"""
************************************************************
numpy_interop.py - HALCON <-> Python inter operation
************************************************************

Project: HALCON/Python

Description:
Inter operation functionality between HALCON and Python constructs.

************************************************************

(c) 1996-2020 by MVTec Software GmbH

Software by: MVTec Software GmbH, www.mvtec.com
"""

from typing import cast, Any, Callable, Dict, Union, Optional, Sequence

from .ffi import HTupleType, HObjectBase, HHandleBase, get_sem_type
from .hhandle import HHandle
from .hobject import HObject
# Explicit look ups to avoid full look up all the time.
from .operator_set import (
    create_dict,
    get_dict_object,
    get_dict_param_s,
    get_dict_param,
    get_dict_tuple_s,
    get_dict_tuple,
    set_dict_object,
    set_dict_tuple,
)


__all__ = [
    'as_python_dict_s',
    'as_python_dict',
    'from_python_dict',
    'HDictElementType',
    'HDictKeyType',
    'HDictSingleElementType',
    'HDictType',
]


# In the absence of recursive types this is a best effort.
HDictKeyType = Union[int, str]
HDictSingleElementType = Union[HTupleType, HObject]
HDictElementType = Union[HDictSingleElementType, Dict[HDictKeyType, Any]]
HDictType = Dict[HDictKeyType, HDictElementType]


# --- Exported Functions ---


def from_python_dict(
    dict_val: HDictType,
    unknown_hook: Optional[Callable[[Any], HDictElementType]] = None
) -> HHandle:
    """
    Converts Python dict to a HALCON dict (HDict).

    Parameters
    ----------

    dict_val: HDictType,
              Python dict that should be converted.

    unknown_hook : Optional[Callable[[Any], HDictType]]
                   Optional function that will be called with the corresponding
                   value if an unknown type is encountered inside dict_val.
                   Only works for values, NOT keys.
                   Value Suggestion: himage_from_numpy_array

    Returns
    -------

    hhandle : HHandle
              Converted control HALCON dict.

    Notes
    -----
    HALCON does NOT support all python constructs. Avoid these patterns:
        - lists containing lists {'key': [0, 1, ['a', 'b']]}
        - lists with a mix of iconic and tuple types {'key': [0, 1, img]}
        - lists of iconic objects {'key': [img1, img2]}

    The following use-cases are supported:
        - dicts containing dicts {'key': {'child_key': 'child_val'}}
        - lists containing dicts {'key': [0, 1, 2, {'key': 'value'}]}
        - iconic objects {'key': img}
        - int and str key types {1: 'val', 'str': 'val'}
        - baisc value types {'k1': 1, 'k2': 2.2, 'k3': 'str', 'k4': handle}
        - arbitary nesting {1: {2: {3: {4: {5: 6}}}}}

    Where:
        img = ha.read_image('patras')
        handle = ha.create_camera_setup_model(num_cameras=4)

    Does not modify input object, unless unknown_hook does so.

    Raises HError if conversion cannot be performed.
    """
    # By default Python doesn't do tail call optimization, so this uses a
    # concrete DFS instead.

    # TODO single FFI call flat unwrap.

    top_level_hhandle = create_dict()

    if len(dict_val) == 0:
        return top_level_hhandle

    dict_stack = [(top_level_hhandle, dict_val)]

    def register_dict(val):
        if isinstance(val, dict):
            child_hhandle = create_dict()
            dict_stack.append((child_hhandle, val))
            return child_hhandle

        return val

    if unknown_hook is None:
        def hook_process(val):
            return val
    else:
        def hook_process(val):
            # Work with fixed list of known types, otherwise unknown_hook
            # would have to fulfill additional requirements.
            #
            # This is asymmetric to the way other functionality is compatible
            # with anything that implements requirements like __iter__.
            val_type = type(val)
            is_known = (
                val_type == float or val_type == int or val_type == str or
                isinstance(val, (list, set, dict, HHandleBase, HObjectBase))
            )

            if is_known:
                return val

            return unknown_hook(val)

    while len(dict_stack) != 0:
        hhandle, dic = dict_stack.pop()

        for key, value in dic.items():
            # Hook the value before checking if it's a single entry or not.
            hooked_val = hook_process(value)

            if (not hasattr(hooked_val, '__iter__') or type(hooked_val) == str
                    or isinstance(hooked_val, (dict, HObjectBase))):
                insert_val = register_dict(hooked_val)

                if isinstance(insert_val, HObject):
                    set_dict_object(insert_val, hhandle, key)
                else:
                    # set_dict_tuple type checks insert_val for compatibility.
                    set_dict_tuple(hhandle, key, insert_val)
            else:
                # This is not optimal, but alternatives are way more complex
                # and the optimization amied for should be single call FFI,
                # which brings other type check implication anyway.
                insert_list = [
                    register_dict(hook_process(e)) for e in hooked_val
                ]

                # Only valid configuration left.
                # List of objects or list not possible anymore.
                set_dict_tuple(hhandle, key, insert_list)

    return top_level_hhandle


def as_python_dict(
    hhandle: HHandle,
    hobject_hook: Optional[Callable[[HObject], Any]] = None
) -> HDictType:
    """
    Converts HALCON dict (HDict) to Python dict.

    Parameters
    ----------

    hhandle: HHandle,
             HALCON dict that should be converted.

    hobject_hook : Optional[Callable[[HObject], Any]]
                   Optional function that will be called for all hobject values
                   that are encountered inside the hdict.
                   Value Suggestion: himage_as_numpy_array

    Returns
    -------

    dict_val : HDictType
               Converted Python dict.

    Notes
    -----
    The conversion from a HDict to Python dict should always succeed because
    the Python dict and list data structures are a super-set of HDict.

    HALCON tuples are represented as Python lists, with 0, 1 or more elements.
    """
    return _as_python_dict_impl(hhandle, True, hobject_hook)


def as_python_dict_s(
    hhandle: HHandle,
    hobject_hook: Optional[Callable[[HObject], Any]] = None
) -> HDictType:
    """
    Converts HALCON dict (HDict) to Python dict.

    Parameters
    ----------

    hhandle: HHandle,
             HALCON dict that should be converted.

    hobject_hook : Optional[Callable[[HObject], Any]]
                   Optional function that will be called for all hobject values
                   that are encountered inside the hdict.
                   Value Suggestion: himage_as_numpy_array

    Returns
    -------

    dict_val : HDictType
               Converted Python dict.

    Notes
    -----
    The conversion from a HDict to Python dict should always succeed because
    the Python dict and list data structures are a super-set of HDict.

    Tuples are represented as single Python values.

    Raises HError if any HALCON tuple has not exactly 1 value.
    """
    return _as_python_dict_impl(hhandle, False, hobject_hook)


# --- Private Implementation Details ---


def _as_python_dict_impl(
    hhandle: HHandle,
    as_list: bool,
    hobject_hook: Optional[Callable[[HObject], Any]] = None
) -> HDictType:
    """Converts HALCON dict (HDict) to Python dict."""
    # By default Python doesn't do tail call optimization, so this uses a
    # concrete DFS instead.

    # TODO single FFI call flat unwrap.

    result_dict: HDictType = {}

    dict_stack = [(hhandle, result_dict)]

    def map_value_no_hook(val):
        if (isinstance(val, HHandleBase) and
                get_sem_type(val._handle_ptr) == 'dict'):
            dict_stack.append((val, {}))
            # Returns refrence to object which will later be populated.
            return dict_stack[len(dict_stack) - 1][1]

        return val

    if hobject_hook is None:
        map_value = map_value_no_hook
    else:
        def map_value_with_hook(val):
            if isinstance(val, HObject):
                return hobject_hook(val)

            return map_value_no_hook(val)

        map_value = map_value_with_hook

    while len(dict_stack) != 0:
        dict_handle, dic = dict_stack.pop()

        keys = cast(Sequence[str], get_dict_param(dict_handle, 'keys', []))

        for key in keys:
            val_type = get_dict_param_s(dict_handle, 'key_data_type', key)
            if val_type == 'tuple':
                if as_list:
                    dic[key] = [
                        map_value(e) for e in get_dict_tuple(dict_handle, key)
                    ]
                else:
                    dic[key] = map_value(get_dict_tuple_s(dict_handle, key))
            else:
                dic[key] = map_value(get_dict_object(dict_handle, key))

    return result_dict
