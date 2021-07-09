"""
************************************************************
hobject.py - HALCON object
************************************************************

Project: HALCON/Python

Description:
HALCON objects model images, regions and XLDs.
The type hierarchy for objects looks like this:

         object
           |
  |--------|--------|
  |        |        |
  v        v        v
 image   region    xld
                    |
                    |----------|----------|----------|----------|
                    |          |          |          |          |
                    v          v          v          v          v
                xld_cont   xld_poly  xld_para  xld_mod_para  xld_ext_para

NOTE: The Python interface represents all these variants as HObject.

************************************************************

(c) 1996-2020 by MVTec Software GmbH

Software by: MVTec Software GmbH, www.mvtec.com
"""

from typing import Sequence, Union, Final, Dict, Callable, Generator

from .ffi import HObjectBase, HError, Hkey
from .util import slice_to_index_list

import halcon


class HObject(HObjectBase):
    """
    HALCON HObject class.

    NOTE: HObject and derived classes manage a sequence of values internally,
    for example a slice of HObject is still a single HObject.
    """
    def __getitem__(self, index: Union[int, slice]) -> 'HObject':
        """
        Select element or slice of elements.

        Notes:
        ------
        Same index semantics as Python list.
        That includes zero based indicies. Converts internally to HALCON object
        indicies starting at one.

        Raises IndexError if index is out of range.
        """
        length = len(self)

        if (isinstance(index, slice)):
            idx_list = [i + 1 for i in slice_to_index_list(index, length)]
            return halcon.select_obj(self, idx_list)
        elif index < 0:
            effective_idx = length + index + 1
        else:
            effective_idx = index + 1

        if effective_idx > length:
            raise IndexError(
                f'HObject index out of range, idx: {effective_idx - 1}'
                f' but length is: {length}'
            )

        return halcon.select_obj(self, effective_idx)

    def __len__(self) -> int:
        """Return length of elements."""
        return halcon.count_obj(self)

    def __iter__(self) -> Generator['HObject', None, None]:
        """Return iterator over elements."""
        length = len(self)
        for i in range(length):
            yield self[i]

    def __array__(self) -> None:
        raise HError('Use himage_as_numpy_array to convert to numpy')

    def __repr__(self) -> str:
        """
        Generate debug representation.

        Notes:
        ------
        The output is meant for console output, not for deserailization
        and provides zero version stablity gurantees.
        """
        obj_types = halcon.get_obj_class(self)
        hid = hex(id(self))
        obj_length = len(obj_types)

        if obj_length == 1:
            typ = obj_types[0]
            type_repr = _obj_repr.get(typ, _generic_repr)(self, typ)
            return f'HObject{{len: {obj_length}, id: {hid}, {type_repr}}}'

        nl = ',\n    '

        if obj_length <= 8:
            tr = nl.join(self._map_obj_repr(obj_types))
            return f'HObject{{\n    len: {obj_length}{nl}id: {hid}{nl}{tr}\n}}'

        types_begin_repr = self._map_obj_repr(obj_types[:4])
        types_end_repr = self._map_obj_repr(obj_types[-4:], obj_length - 4)

        return f'HObject{{\n    len: {obj_length}{nl}id: {hid}{nl}' \
            f'{nl.join(types_begin_repr)}{nl}' \
            f'[...]\n' \
            f'    {nl.join(types_end_repr)}\n}}'

    def _map_obj_repr(
        self,
        obj_types: Sequence[str],
        offset: int = 0
    ) -> Sequence[str]:
        return [
            f'{i}: {_obj_repr.get(typ, _generic_repr)(self[i], typ)}'
            for i, typ in enumerate(obj_types, offset)
        ]

    @staticmethod
    def _copy_from_key(from_key: Hkey) -> 'HObject':
        """Ref count increasing copy of iconic object, based on key."""
        return HObject(HObjectBase._copy_key(from_key))


# --- Private Implementation Details ---


def _image_repr(image: HObject, obj_type: str) -> str:
    """Generates repr string for HALCON image."""
    width, height = halcon.get_image_size_s(image)
    # Channel indexing starts at 1 here.
    channel_ids = range(1, halcon.count_channels_s(image) + 1)
    channels = ', '.join([
        typ for typ in halcon.get_channel_info(image, 'type', channel_ids)
    ])

    return f'type: {obj_type}, size: {width}x{height}, channels: [{channels}]'


def _region_repr(region: HObject, obj_type: str) -> str:
    """Generates repr string for HALCON region."""
    area, row, column = halcon.area_center_s(region)
    return f'type: {obj_type}, area: {area}, center: ({row:.2f}, {column:.2f})'


# TODO xld


def _generic_repr(obj: HObject, obj_type: str) -> str:
    """Generates repr string for generic HALCON object."""
    return f'type: {obj_type}'


_obj_repr: Final[Dict[str, Callable[[HObject], str]]] = {
    'image': _image_repr,
    'region': _region_repr,
}
