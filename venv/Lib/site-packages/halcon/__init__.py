"""
************************************************************
__init__.py - module entry point
************************************************************

Project: HALCON/Python

************************************************************

(c) 1996-2020 by MVTec Software GmbH

Software by: MVTec Software GmbH, www.mvtec.com
"""

# Make everything from operator_set visible when importing halcon.
from .operator_set import *

from .hobject import HObject
from .hhandle import HHandle
from .ffi import HError, HOperatorError, HTupleConversionError
from .interop import HInteropError
from .hdevengine import (
    HDevEngineError,
    HDevEngine,
    HDevProcedure,
    HDevProcedureCall,
    HDevProgram,
    HDevProgramCall,
)


# Import all interop to facilitate ha.* style namespace resolution,
# in an ergonomic, consistent and readable way.
from .python_interop import *
from .numpy_interop import *
