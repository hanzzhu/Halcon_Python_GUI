"""
************************************************************
numpy_interop.py - HALCON <-> numpy inter operation
************************************************************

Project: HALCON/Python

Description:
The goal for these optional third party integration modules, is as follows.
- be optional
- work directly if the third party module is available,
without further user actions required
- provide conversion and other interop functionality
- allow interop module import even if third party module is not available
- provide actionable error message should an interop function be called,
when the third party module is not available.
- provide type annotations

The strategy for achieving these goals is:
- recovering from import errors
- gate public functionality by a bail out based on whether the third party
module was imported successfully

************************************************************

(c) 1996-2020 by MVTec Software GmbH

Software by: MVTec Software GmbH, www.mvtec.com
"""

try:
    # Type checking disabled because numpy by default does not have
    # type annotations.
    import numpy  # type: ignore
    _numpy_imported = True
except ImportError:
    _numpy_imported = False


from .hobject import HObject
from .interop import HInteropError
# Explicit look ups to avoid full look up all the time.
from .operator_set import (
    access_channel,
    channels_to_image,
    count_channels_s,
    gen_image1,
    get_image_pointer1_s,
)


__all__ = [
    'himage_from_numpy_array',
    'himage_as_numpy_array',
]


# --- Exported Functions ---


def himage_from_numpy_array(arr: 'numpy.ndarray') -> HObject:
    """
    Converts numpy array to HALCON object.

    Parameters
    ----------

    arr : numpy.ndarray
          Numpy array that should be converted.


    Returns
    -------

    obj : HObject
          Converted iconic HALCON image.

    Notes
    -----
    Has to perform a deep copy of all image data. This might be expensive.

    The numpy array has to be either 2 or 3 dimensional,
    other shapes are NOT supported.

    Raises HInteropError if conversion cannot be performed,
    because the numpy array has certain expected incompatible properties,
    more fundamental exceptions otherwise.
    """
    _numpy_import_bailout()

    obj_shape_len = len(arr.shape)

    if obj_shape_len == 2:
        # Single channel images (width x height).
        return _hobject_from_numpy_channel(arr)
    elif obj_shape_len == 3:
        # Treat 3 dimensional numpy arrays as multi channel images.
        # (width x height x N) where N is the number of channels.
        num_channels = arr.shape[2]

        if num_channels == 1:
            # Optimize the shape (width x height x 1).
            return _hobject_from_numpy_channel(arr)

        return channels_to_image(HObject([
            _hobject_from_numpy_channel(channel)
            for channel in numpy.dsplit(arr, num_channels)
        ]))

    raise HInteropError(
        f'numpy array shape not supported: {arr.shape}'
    )


def himage_as_numpy_array(obj: HObject) -> 'numpy.ndarray':
    """
    Converts single HALCON image to numpy array.

    Parameters
    ----------

    obj : HObject
          Single iconic HALCON image that should be converted.

    Returns
    -------

    arr : numpy.ndarray
          Converted numpy array.

    Notes
    -----
    The conversion from HALCON image to numpy array should always succeed
    because the numpy array is a super-set of iconic HALCON images.

    Has to perform a deep copy of all image data. This might be expensive.

    Represents single channel images as 2D numpy arrays, and multi channel
    images as 3D numpy arrays.

    Raises HTupleConversionError if obj is not a single HALCON image.
    """
    _numpy_import_bailout()

    num_channels = count_channels_s(obj)
    if num_channels == 1:
        return numpy.copy(_as_numpy_array_interface(obj))

    # TODO pull out channel image pointers in single call.
    return numpy.dstack([
        _as_numpy_array_interface(access_channel(obj, i))
        for i in range(1, num_channels + 1)
    ])


# --- Private Implementation Details ---


if _numpy_imported:
    # Numpy dtypes are kind of funky numpy.uint8 != numpy.dtype('uint8').
    _numpy_image_type_to_halcon = {
        numpy.dtype('int16'): 'int2',
        numpy.dtype('int32'): 'int4',
        numpy.dtype('int8'): 'int1',
        numpy.dtype('single'): 'real',
        numpy.dtype('uint16'): 'uint2',
        numpy.dtype('uint8'): 'byte',
    }

    # = is native byte order.
    _halcon_image_type_to_numpy = {
        'byte': '=u1',
        'int1': '=i1',
        'int2': '=i2',
        'int4': '=i4',
        'real': '=f4',
        'uint2': '=u2',
    }


class _NumpyArrayInterfaceWrapper(object):
    """Numpy compatible array interface."""

    def __init__(self, height: int, width: int, np_type_str: str, ptr: int):
        self.__array_interface__ = {
            'version': 3,
            'shape': (height, width),
            'typestr': np_type_str,
            'data': (ptr, True),
        }


def _numpy_import_bailout() -> None:
    if not _numpy_imported:
        raise HInteropError(
            'This functionality is only available if numpy is available.'
        )


def _as_numpy_array_interface(obj: HObject) -> _NumpyArrayInterfaceWrapper:
    """
    Return numpy array interface.

    Raises HInteropError if image has not exaclty 1 channel.

    Raises HInteropError if image type cannot be mapped to numpy
    image type.
    """
    num_channels = count_channels_s(obj)
    if num_channels != 1:
        raise HInteropError(
            f'Expected single channel object found: {num_channels}'
        )

    ptr, type_str, width, height = get_image_pointer1_s(obj)
    np_type_str = _halcon_image_type_to_numpy.get(type_str)
    if np_type_str is None:
        raise HInteropError(
            f'No matching numpy type string for: {type_str}'
        )

    return _NumpyArrayInterfaceWrapper(height, width, np_type_str, ptr)


def _hobject_from_numpy_channel(obj: 'numpy.ndarray') -> HObject:
    """Converts single numpy channel to iconic HALCON object."""
    # Flatten array to be sure, that the underlying data doesn't use
    # strides. Ravel only copies, when necessary.
    flat_obj = obj.ravel()
    img_ptr = flat_obj.ctypes.data
    type_str = _numpy_image_type_to_halcon.get(flat_obj.dtype)
    if type_str is None:
        raise HInteropError(
            f'No matching halcon type string for: {flat_obj.dtype}'
        )

    # Copy image data to HALCON memory.
    return gen_image1(type_str, obj.shape[1], obj.shape[0], img_ptr)
