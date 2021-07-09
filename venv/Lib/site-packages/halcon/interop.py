"""
************************************************************
numpy_interop.py - inter operation utility
************************************************************

Project: HALCON/Python

************************************************************

(c) 1996-2020 by MVTec Software GmbH

Software by: MVTec Software GmbH, www.mvtec.com
"""

from .ffi import HError


__all__ = [
    'HInteropError',
]


# --- Exported Classes ---


class HInteropError(HError):
    """HALCON interop exception."""
    pass
