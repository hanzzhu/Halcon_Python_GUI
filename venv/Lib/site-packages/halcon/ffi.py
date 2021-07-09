"""
************************************************************
ffi.py - wrapper for HALCON LI functions
************************************************************

Project: HALCON/Python

Description:
Handles ALL FFI and related error handling.

HALCON API
  ^
  | FFI (Foreign Function Interface)
  v
 HLI (Halcon Language Interface C Library)
  ^
  | C calls
  v
HALCON (Halcon C Library)

HLI and HALCON are both inside the halcon shared library.

************************************************************

(c) 1996-2020 by MVTec Software GmbH

Software by: MVTec Software GmbH, www.mvtec.com
"""

import ctypes
import sys

from typing import Sequence, Final, Union, Optional, Dict, cast

from . import mixin

from .util import MaybeSequence
from .meta import halcon_native_lib_names


__all__ = [
    'get_sem_type',
    'HalconOperator',
    'HCkP',
    'Herror',
    'HError',
    'HHandleBase',
    'Hkey',
    'HNull',
    'HObjectBase',
    'HOperatorError',
    'HTupleConversionError',
    'HTupleElementType',
    'HTupleType',
]


# --- Exported Classes ---


class HError(Exception):
    """HALCON base exception."""

    def __init__(self, message: str):
        self.message = message

    def __str__(self) -> str:
        return self.message


class HOperatorError(HError):
    """HALCON operator exception."""

    def __init__(self, error_code: int):
        self.error_code = error_code
        self.message = _get_error_text(error_code)


class HTupleConversionError(HError):
    """HALCON tuple conversion exception."""
    pass


class HHandleBase(object):
    """
    HALCON handle base class.

    Interface only creates derived object instances.
    """

    def __init__(self, handle_ptr: ctypes.c_void_p):
        """Initialize HALCON handle from valid pointer."""
        self._handle_ptr = ctypes.c_void_p()
        HCkP(_halcon_lib.HLICopyHandle(
            handle_ptr,
            ctypes.byref(self._handle_ptr)
        ))

    def __del__(self) -> None:
        HCkP(_halcon_lib.HLIClearHandle(self._handle_ptr))


# TODO what should what shouldn't be valid with HNull?
class HNull(HHandleBase):
    """HALCON HNULL handle class."""
    pass


# --- Type Definitions ---

# Defines type aliases for HALCON types used internally and in the interface.

HTupleElementType = Union[int, float, str, HHandleBase]
HTupleType = Union[HTupleElementType, Sequence[HTupleElementType]]

# Keep in sync with C definitions.
Herror = ctypes.c_uint32
# Technically long* on C side. But since we never intend to look at or modify
# the value on Python side. c_void_p is simpler than byref(c_long).
Hkey = ctypes.c_void_p


class HObjectBase(object):
    """
    HALCON iconic object base class.

    Interface only creates derived object instances.
    """

    def __init__(
        self,
        iconic_objects: Union[Hkey, MaybeSequence['HObjectBase']]
    ):
        """
        Construct based on key or concatenate list of iconic objects.

        Notes
        -----
        This constructor serves dual purpose, one internal for raw keys

        And one public one, for making owning copies or iconic objects,
        optionally concatenating a list of iconic objects.

        This probably does not perform pixel level copies, the implementation
        uses ref counting and other techniques to avoid expensive copies.

        Examples
        --------
        img_copy = HObject(img1)
        img_seq = HObject([img1, img2, img3])
        """
        if isinstance(iconic_objects, Hkey):
            self._key = iconic_objects
        else:
            if isinstance(iconic_objects, (list, set)):
                new_obj = mixin.upcast_obj_list(iconic_objects)
            else:
                new_obj = iconic_objects

            self._key = self._copy_key(new_obj._key)

    def __del__(self) -> None:
        """
        Cleans up associated native resources.

        Notes
        -----
        This might be a simple ref count decrease or some more expensive
        operation.
        """
        HCkP(_halcon_lib.HLIClearObject(self._key))

    @staticmethod
    def _copy_key(from_key: Hkey) -> Hkey:
        """Ref count increasing copy of iconic object key."""
        cloned_key = Hkey()
        HCkP(_halcon_lib.HLICopyObject(
            from_key,
            ctypes.byref(cloned_key)
        ))
        return cloned_key


class HalconOperator(object):
    """
    HALCON operator.

    Facilitates setting up, and calling operators.
    Manages proc_handle lifetime via context manager.

    Typical call order:
    __init__
    __enter__
    set_input_object x N
    set_input_tuple x N
    init_oct x N
    execute
    get_output_object_key x N
    get_output_tuple(_s/_m) x N
    __exit__
    """
    def __init__(self, operator_id: int):
        """Create procedure handle based on operator id."""
        assert _is_valid_i32(operator_id)
        self.proc_handle = ctypes.c_ssize_t()

        # Does not call destroy when create fails.
        HCkP(_halcon_lib.HLICreateProcedure(
            ctypes.c_int32(operator_id),
            ctypes.byref(self.proc_handle)
        ))

    def set_input_object(
        self,
        par_index: int,
        input_object: HObjectBase
    ) -> None:
        """Set procedure iconic input object."""
        # Assumes input_object private _key field was not modified by user.
        HCkP(_halcon_lib.HLISetInputObject(
            self.proc_handle,
            ctypes.c_int32(par_index),
            input_object._key
        ))

    def set_input_tuple(
        self,
        par_index: int,
        value: HTupleType
    ) -> None:
        """Set procedure control input tuple."""
        tuple_ptr = ctypes.c_ssize_t()
        HCkP(_halcon_lib.HLIGetInputTuple(
            self.proc_handle,
            ctypes.c_int32(par_index),
            ctypes.byref(tuple_ptr)
        ))
        _python_to_htuple(tuple_ptr, value)

    def init_oct(self, par_index: int) -> None:
        """Initialize output control tuple."""
        # TODO extend HLI to allow init multiple at once, to avoid FFI roundtrips.
        HCkP(_halcon_lib.HLIInitOCT(
            self.proc_handle,
            ctypes.c_int32(par_index)
        ))

    def execute(self) -> None:
        """Execute procedure."""
        HCkP(_halcon_lib.HLICallProcedure(self.proc_handle))

    def get_output_object_key(self, par_index: int) -> Hkey:
        """Return procedure iconic output object key."""
        output_key = Hkey()
        HCkP(_halcon_lib.HLIGetOutputObject(
            self.proc_handle,
            ctypes.c_int32(par_index),
            ctypes.byref(output_key)
        ))
        return output_key

    def get_output_tuple_s(
        self,
        par_index: int
    ) -> HTupleElementType:
        """Return procedure control output tuple, as single value."""
        output_tuple_ptr = self._get_output_tuple_ptr(par_index)
        if output_tuple_ptr.value == 0:
            raise HTupleConversionError(
                'Expected exactly one tuple value, got 0.'
            )

        # Type checking disabled because we know it's a single value,
        # based on as_list False.
        return _htuple_to_python(output_tuple_ptr, False)  # type: ignore

    def get_output_tuple_m(
        self,
        par_index: int
    ) -> Sequence[HTupleElementType]:
        """Return procedure control output tuple, as list."""
        output_tuple_ptr = self._get_output_tuple_ptr(par_index)
        if output_tuple_ptr.value == 0:
            return []

        # Type checking disabled because we know it's a list,
        # based on as_list True.
        return _htuple_to_python(output_tuple_ptr, True)  # type: ignore

    def _get_output_tuple_ptr(self, par_index: int) -> ctypes.c_ssize_t:
        output_tuple_ptr = ctypes.c_ssize_t()
        HCkP(_halcon_lib.HLIGetOutputTuple(
            self.proc_handle,
            ctypes.c_int32(par_index),
            True,  # handle_type True == not legacy
            ctypes.byref(output_tuple_ptr)
        ))

        # assert output_tuple_ptr.value != 0
        return output_tuple_ptr

    def __enter__(self) -> 'HalconOperator':
        return self

    # Type checking disabled because the parameters are not user provided.
    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore
        """Destroy procedure handle, later use is UB."""
        # HDeleteHProc on the C side is severely underspecified, it's not
        # entirely clear what should happen if something goes wrong before the
        # procedure was called. The C++ interface passes the received Herror
        # into HDeleteHProc, if for example a tuple allocation fails. But what
        # if a python value could not be converted to a HALCON tuple? Then no
        # Herror exists, but the procedure still needs to be cleaned up.
        # A coarse analysis of HDeleteHProc and it's transitive function shows
        # that if the value is set to H_MSG_OK, associated functionality gets
        # mostly ignored. And it should be fine.

        proc_result = exc_value.error_code \
            if exc_type == HOperatorError else 2  # C H_MSG_OK

        HCkP(_halcon_lib.HLIClearAllIOCT(self.proc_handle))
        # TODO maybe only if already called and has proc_result.
        HCkP(_halcon_lib.HLIDestroyProcedure(
            self.proc_handle,
            # Either execute result or error from setup calls, or H_MSG_OK,
            # if something else went wrong, like tuple convert error.
            ctypes.c_int32(proc_result)
        ))


# --- Exported Functions ---


def HCkP(error_code: int) -> None:
    """Check the error number and raise exception if no ok."""
    if error_code != 2:  # C constant H_MSG_OK
        raise HOperatorError(error_code)


# Expects caller to call with valid handle.
def get_sem_type(handle_ptr: ctypes.c_void_p) -> str:
    out_string = ctypes.c_char_p()
    HCkP(_halcon_lib.HLIGetHandleSemType(handle_ptr, ctypes.byref(out_string)))
    if out_string.value is None:
        return 'hnull'

    return out_string.value.decode('utf-8')


# Load dynamic HALCON library. The name of the library to load depends on the
# operating system. Open with RTLD_GLOBAL on Linux and macOS so that other
# HALCON components like acquisition interfaces can find the HALCON symbols.
def load_native_lib(names: Dict[str, str]) -> ctypes.CDLL:
    # See sys.startswith docs for explanation about startswith pattern.
    if sys.platform.startswith('linux'):
        name = names['linux']
    elif sys.platform.startswith('darwin'):
        name = names['darwin']
    elif sys.platform == 'win32' or sys.plaform == 'cygwin':
        name = names['win']
    else:
        raise HError(f'Unsupported operating system {sys.platform}')

    # Python 3.8 added the winmode parameter. If it is not specified, Python
    # will attempt to open the DLL using LoadLibraryEx's
    # LOAD_LIBRARY_SEARCH_DEFAULT_DIRS mode, which does not take the PATH
    # environment variable into account. Since we expect the user to specify
    # the location of the HALCON libraries to load via the PATH environment
    # variable, that is not what we want, and we must specify winmode=0. Note
    # the Python 3.8 documentation is incorrect and claims winmode=0 is the
    # default; it is not.
    lib = ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL, winmode=0)

    if lib is None:
        raise HError(f'Failed to load {name} library')

    return lib

# --- Private Implementation Details ---


# Load dynamic HALCON library.
_halcon_lib: Final[ctypes.CDLL] = load_native_lib(halcon_native_lib_names)


class _HTuple(object):
    """
    Internal convenience HALCON HTuple RAII wrapper.

    Notes
    -----
    HALCON/Python does NOT have a HTuple class a user is ever supposed to use.
    HTuple is represented as Python list.

    This is ONLY meant for internal use.
    """

    def __init__(self, value: Optional[HTupleType]):
        """Construct HTuple with value."""
        self._tuple_ptr = ctypes.c_ssize_t()
        HCkP(_halcon_lib.HLICreateTuple(ctypes.byref(self._tuple_ptr)))

        if value is not None:
            _python_to_htuple(self._tuple_ptr, value)

    def as_python(self, as_list: bool) -> HTupleType:
        """Convert native HTuple to python object."""
        return _htuple_to_python(self._tuple_ptr, as_list)

    def __enter__(self) -> '_HTuple':
        """Do nothing on enter."""
        return self

    # Type checking disabled because the parameters are not user provided.
    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore
        """Destroy tuple, later use is UB."""
        HCkP(_halcon_lib.HLIDestroyTuple(self._tuple_ptr))


def _get_error_text(error_code: int) -> str:
    """Return the description associated to the error id."""
    # Keep in sync with #define MAX_STRING 1024
    out_string = ctypes.create_string_buffer(1024)
    # Returns void should not fail.
    # TODO extended error messages.
    _halcon_lib.HLIGetErrorMessage(error_code, out_string)
    return out_string.value.decode('utf-8')


def _is_valid_i32(number: int) -> bool:
    return number < 2_147_483_648 and number >= -2_147_483_648


def _is_valid_ssize_t(number: int) -> bool:
    return number <= sys.maxsize and number >= -sys.maxsize


# HTuple conversions are carefully optimized because it is used ubiquitously.

def _set_htuple_value(
    tuple_ptr: ctypes.c_ssize_t,
    index: ctypes.c_int32,
    value: HTupleElementType
) -> None:
    # Only lookup type once, no need for inheritance behavior of isinstance.
    value_type = type(value)

    # Order sorted by most common as optimization.
    if value_type == float:
        c_double_value = ctypes.c_double(cast(float, value))
        HCkP(_halcon_lib.HLISetD(tuple_ptr, index, c_double_value))
    elif value_type == int:
        int_value = cast(int, value)
        # NOTE: while the interface takes an int64 it treats it as ssize_t.
        assert _is_valid_ssize_t(int_value)
        HCkP(_halcon_lib.HLISetL(tuple_ptr, index, ctypes.c_int64(int_value)))
    elif isinstance(value, HHandleBase):
        if isinstance(value, HNull):
            HCkP(_halcon_lib.HLISetH(tuple_ptr, index, 0))
        else:
            HCkP(_halcon_lib.HLISetH(tuple_ptr, index, value._handle_ptr))
    elif value_type == str:
        c_string = ctypes.c_char_p(cast(str, value).encode('utf-8'))
        HCkP(_halcon_lib.HLISetS(tuple_ptr, index, c_string))
    else:
        raise HTupleConversionError(
            f'Type is not str, int, float or HHandleBase: {value_type}'
        )


def _python_to_htuple(
    tuple_ptr: ctypes.c_ssize_t,
    value: HTupleType
) -> None:
    """
    Convert python value to htuple.

    Parameters
    ----------

    tuple_ptr : ctypes.c_ssize_t
                Number representing pointer of successfully created htuple.

    value : HTupleType
            Python value that will be converted.

    Notes
    -----
    The specific FFI functions called depend on the run time layout of the
    Python value, and are subject to change while preserving the overall
    existing semantics.

    All invalid Python values will raise an HTupleConversionError.
    If an exception happens, the htuple is left in a valid but unspecified
    state.
    """
    value_type = type(value)

    if not hasattr(value, '__iter__') or value_type == str:
        # TODO leverage Hctuple's SBO capabilites.
        _halcon_lib.HLICreateElements(tuple_ptr, ctypes.c_int32(1))
        _set_htuple_value(
            tuple_ptr,
            ctypes.c_int32(0),
            cast(HTupleElementType, value)
        )
        return

    if hasattr(value, '__len__'):
        # Type checking disabled because with hasattr we know it has len.
        tuple_length = len(value)  # type: ignore

        assert _is_valid_i32(tuple_length)

        # Reserve elements of known size if possible as optimization.
        _halcon_lib.HLICreateElements(tuple_ptr, ctypes.c_int32(tuple_length))

        # TODO try out homogenous check and single call array copy.

        # Type checking disabled because with hasattr we know it has len,
        # and the only types in the union are sequence which have iter.
        for i, element in enumerate(value):  # type: ignore
            _set_htuple_value(tuple_ptr, ctypes.c_int32(i), element)
    else:
        raise HTupleConversionError(
            'Type is not str, int, float, HHandleBase or Sequence of those:'
            f' {value_type}'
        )


def _htuple_element_to_python(
    tuple_ptr: ctypes.c_ssize_t,
    py_index: int,
    element_type: int
) -> HTupleElementType:
    """Return native HTuple value and convert it into Python representation."""

    # No need to check for 32 bit range, becuase output of c call.
    index = ctypes.c_int32(py_index)

    # Order sorted by most common as optimization.
    # Direct values because used once and constant propagation in Python :|

    if element_type == 2:  # C constant DOUBLE_PAR
        out_c_double = ctypes.c_double()
        HCkP(_halcon_lib.HLIGetD(tuple_ptr, index, ctypes.byref(out_c_double)))
        return out_c_double.value

    if element_type == 1:  # C constant LONG_PAR
        out_c_int64 = ctypes.c_int64()
        HCkP(_halcon_lib.HLIGetL(tuple_ptr, index, ctypes.byref(out_c_int64)))
        return out_c_int64.value

    if element_type == 16:  # C constant HANDLE_PAR
        out_handle_ptr = ctypes.c_void_p()
        HCkP(_halcon_lib.HLIGetH(
            tuple_ptr,
            index,
            ctypes.byref(out_handle_ptr)
        ))

        if out_handle_ptr.value is None:
            return HNull(ctypes.c_void_p())

        return mixin.to_handle(out_handle_ptr)

    if element_type == 4:  # C constant STRING_PAR
        out_c_str = ctypes.c_char_p()
        HCkP(_halcon_lib.HLIGetS(tuple_ptr, index, ctypes.byref(out_c_str)))
        c_str_bytes = out_c_str.value
        if c_str_bytes is None:
            raise Exception('TODO What should be done here?')

        return c_str_bytes.decode('utf-8')

    raise HTupleConversionError(
        f'Unknown HTuple element type {element_type}'
    )


def _htuple_to_python(
    tuple_ptr: ctypes.c_ssize_t,
    as_list: bool
) -> HTupleType:
    """
    Convert htuple to python value.

    Parameters
    ----------

    tuple_ptr : ctypes.c_ssize_t
                Number representing pointer of successfully created htuple.

    as_list : bool
              Should the output always be a list.

    Returns
    -------

    value : Optional[HTupleType]
            Python value representation of htuple.

    Notes
    -----
    The specific FFI functions called depend on the run time layout of the
    Python value, and are subject to change while preserving the overall
    existing semantics.

    If the htuple cannot be represented as Python value, raises
    HTupleConversionError.

    If as_list is set to False, but the tuple has not exactly 1 value,
    raises HTupleConversionError.

    Does not modify the htuple, regardless of execution path.
    """
    # TODO SBO store type in pointer without alloc if len == 1.
    # and SBO value in additional parameter.
    # TODO check for homogenous type and pull out via one GatArr call.

    # Guessing that allocation and complexity cost of pulling out all
    # elements of a hetergenous tuple in a single call won't be worth it.

    c_tuple_len = ctypes.c_int32()
    types = ctypes.POINTER(ctypes.c_int32)()

    try:
        HCkP(_halcon_lib.HLIGetElementTypes(
            tuple_ptr,
            ctypes.byref(c_tuple_len),
            ctypes.byref(types)
        ))

        tuple_len = c_tuple_len.value

        if as_list:
            return [
                _htuple_element_to_python(tuple_ptr, i, cast(int, types[i]))
                for i in range(tuple_len)
            ]
        elif tuple_len != 1:
            raise HTupleConversionError(
                f'Invalid tuple length: {tuple_len}, expected single value.'
            )

        return _htuple_element_to_python(tuple_ptr, 0, cast(int, types[0]))

    finally:
        HCkP(_halcon_lib.HLIDestroyTupleTypes(types))

# TODO test fail import with path set to only pipenv virtualenv.
