"""
************************************************************
hhandle.py - HALCON handle
************************************************************

Project: HALCON/Python

Description:
HALCON Handles model complex control data structures.

************************************************************

(c) 1996-2020 by MVTec Software GmbH

Software by: MVTec Software GmbH, www.mvtec.com
"""

from .ffi import get_sem_type, HHandleBase


class HHandle(HHandleBase):
    """HALCON HHandle class."""

    def __eq__(self, other: 'HHandle') -> bool:
        """
        Return true if both handles represent the same HALCON handle.

        Notes:
        ------
        Implementation for built in rich equality comparison.
        """
        return self._handle_ptr.value == other._handle_ptr.value

    def __hash__(self) -> int:
        """
        Return hash of object.

        Notes:
        ------
        Implementation for built in hash().

        Python requires that the object is immutable if you implement __hash__.
        This means the value of _handle_ptr should never change over the
        lifetime of the object.
        This should be a given as _handle_ptr is not passed by ref to the C,
        and _handle_ptr is a private implementation detail the user should
        not touch.
        """
        return self._handle_ptr.value

    def __repr__(self) -> str:
        """
        Generate debug representation.

        Notes:
        ------
        The output is meant for console output, not for deserailization
        and provides zero version stablity gurantees.
        """
        sem_type = get_sem_type(self._handle_ptr)
        hid = hex(id(self))
        return f'HHandle{{type: {sem_type}, id: {hid}}}'
