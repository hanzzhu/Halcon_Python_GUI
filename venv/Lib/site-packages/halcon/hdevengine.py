"""
************************************************************
hdevengine.py - wrapper for HDevEngine functions
************************************************************

Project: HALCON/Python

Description:
Implements native Python bindings for HDevEngine.

Similar to ffi.py performs FFI calls into a native shared library,
in this case hdevenginecpp.

************************************************************

(c) 1996-2020 by MVTec Software GmbH

Software by: MVTec Software GmbH, www.mvtec.com
"""


import ctypes

from typing import Final, List, Union, cast

from .ffi import (
    _htuple_to_python,
    _HTuple,
    _is_valid_i32,
    HError,
    Hkey,
    HTupleType,
    load_native_lib,
)
from .hobject import HObject
from .meta import hdevengine_native_lib_names


__all__ = [
    'HCkE',
    'HDevEngineError',
    'HDevEngine',
    'HDevProcedure',
    'HDevProcedureCall',
    'HDevProgram',
    'HDevProgramCall',
]


# --- Exported Classes ---


class HDevEngineError(HError):
    """HALCON HDevEngine exception."""
    def __init__(
        self,
        error_code: int,
        category: int,
        message: str,
        procedure_name: str,
        line_text: str,
        line_number: int,
        user_data: HTupleType
    ):
        # The HALCON error code also shows up in the message,
        # can be queried programmatically with this field
        self.error_code = error_code
        self.message = message

        # These values also show up in the error message,
        # they can be queried programmatically with these fields.
        self.procedure_name = procedure_name
        self.line_text = line_text
        self.line_number = line_number

        # These fields hold situational information.
        self.category = category
        self.user_data = user_data

    def __str__(self) -> str:
        # 10000 == C constant H_ERR_START_EXT
        is_hlib_error = self.error_code > 1000 and self.error_code <= 10000
        err_src = 'HALCON' if is_hlib_error else 'HDevEngine'

        return f'{err_src} error #{self.error_code}: {self.message}'


class HDevEngine(object):
    """
    HALCON HDevEngine class.

    Executes HDevelop programs and procedures at run time.

    Warning
    -------
    HDevEngine is a mutable singleton.
    """
    def __init__(self) -> None:
        """
        Initialize HDevEngine.

        Notes
        -----
        While you can have multiple Python instances of HDevEngine,
        they all share the same implementation as a mutable singleton.

        This mostly affects configuration, specific procedure call instances
        are more or less independent from each other.
        """
        self._engine_ptr = ctypes.c_ssize_t()
        HCkE(_hdevengine_lib.HCenCreateEngine(ctypes.byref(self._engine_ptr)))

    def __del__(self) -> None:
        HCkE(_hdevengine_lib.HCenDestroyEngine(self._engine_ptr))

    def set_attribute(self, name: str, value: HTupleType) -> None:
        """
        Changes a global setting of the engine.

        Parameters
        ----------

        name : str
               Name of the attribute, e.g. 'ignore_invalid_lines'
               or 'debug_port'.

        value : HTupleType
                New value of the attribute.
                Note that for boolean attributes like 'ignore_invalid_lines',
                the string 'false' or 'true' are expected.
        """
        with _HTuple(value) as htuple:
            HCkE(_hdevengine_lib.HCenSetEngineAttribute(
                self._engine_ptr,
                ctypes.c_char_p(name.encode('utf8')),
                htuple._tuple_ptr
            ))

    def get_attribute(self, name: str) -> HTupleType:
        """
        Queries a global setting of the engine

        Parameters
        ----------

        name : str
               Name of the attribute, e.g. 'ignore_invalid_lines'
               or 'debug_port'.

        Returns
        -------

        value : HTupleType
                Current value of the attribute.
                Note that for boolean attributes like 'ignore_invalid_lines',
                the value will be an int of value 0 or 1 respectively.
        """
        with _HTuple(value=None) as htuple:
            HCkE(_hdevengine_lib.HCenGetEngineAttribute(
                self._engine_ptr,
                ctypes.c_char_p(name.encode('utf8')),
                htuple._tuple_ptr
            ))
            return htuple.as_python(as_list=False)

    def start_debug_server(self) -> None:
        """
        Starts the debug server that allows to attach HDevelop as debugger
        to step through engine code.

        Notes
        -----
        With default settings the server listens on port 57786 and engine runs
        normally until HDevelop is connected and F9 is pressed to stop
        execution.

        Use this in conjunction with wait_for_debug_connection.

        You can control the port via the attribute 'debug_port'.
        """
        HCkE(_hdevengine_lib.HCenStartDebugServer(self._engine_ptr))

    def stop_debug_server(self) -> None:
        """Stops the debug server, resuming execution if stopped."""
        HCkE(_hdevengine_lib.HCenStopDebugServer(self._engine_ptr))

    def set_procedure_path(self, path: str) -> None:
        """
        Sets the search path for loading external procedures.

        Parameters
        ----------

        path : str
               Path in format of the operating system.
        """
        HCkE(_hdevengine_lib.HCenSetProcedurePath(
            self._engine_ptr,
            ctypes.c_char_p(path.encode('utf8'))
        ))

    def add_procedure_path(self, path: str) -> None:
        """
        Appends path to search paths used for loading external procedures.

        Parameters
        ----------

        path : str
               Path in format of the operating system.
        """
        HCkE(_hdevengine_lib.HCenAddProcedurePath(
            self._engine_ptr,
            ctypes.c_char_p(path.encode('utf8'))
        ))

    def get_procedure_names(self) -> List[str]:
        """
        Returns the names of available procedures.

        Returns
        -------

        procedure_names : List[str]
                          Available procedure names.
        """
        with _HTuple(value=None) as htuple:
            HCkE(_hdevengine_lib.HCenGetProcedureNames(
                self._engine_ptr,
                htuple._tuple_ptr
            ))
            return htuple.as_python(as_list=True)  # type: ignore

    def get_loaded_procedure_names(self) -> List[str]:
        """
        Returns the names of loaded procedures.

        Returns
        -------

        procedure_names : List[str]
                          Loaded procedure names.
        """
        with _HTuple(value=None) as htuple:
            HCkE(_hdevengine_lib.HCenGetLoadedProcedureNames(
                self._engine_ptr,
                htuple._tuple_ptr
            ))
            return htuple.as_python(as_list=True)  # type: ignore

    def unload_procedure(self, name: str) -> None:
        """
        Unloads a previously loaded procedure.

        Parameters
        ----------

        name : str
               Name of the procedure to unload.
        """
        HCkE(_hdevengine_lib.HCenUnloadProcedure(
            self._engine_ptr,
            ctypes.c_char_p(name.encode('utf8'))
        ))

    def unload_all_procedures(self) -> None:
        """Unloads all previously loaded procedures."""
        HCkE(_hdevengine_lib.HCenUnloadAllProcedures(self._engine_ptr))

    def get_global_iconic_var_names(self) -> List[str]:
        """
        Returns the names of all global iconic variables.

        Returns
        -------

        names : List[str]
                Names of all global iconic variables.
        """
        with _HTuple(value=None) as htuple:
            HCkE(_hdevengine_lib.HCenGetGlobalIconicVarNames(
                self._engine_ptr,
                htuple._tuple_ptr
            ))
            return htuple.as_python(as_list=True)  # type: ignore

    def get_global_control_var_names(self) -> List[str]:
        """
        Returns the names of all global control variables.

        Returns
        -------

        names : List[str]
                Names of all global control variables.
        """
        with _HTuple(value=None) as htuple:
            HCkE(_hdevengine_lib.HCenGetGlobalCtrlVarNames(
                self._engine_ptr,
                htuple._tuple_ptr
            ))
            return htuple.as_python(as_list=True)  # type: ignore

    def get_global_iconic_var_dimension(self, name: str) -> int:
        """
        Returns the dimension of a global iconic variable.

        Parameters
        ----------

        name : str
               Name of the global iconic variable.

        Returns
        -------

        dimension : int
                    Dimension of the global iconic variable.
        """
        dimension_ptr = ctypes.c_int()
        HCkE(_hdevengine_lib.HCenGetGlobalIconicVarDimension(
            self._engine_ptr,
            ctypes.c_char_p(name.encode('utf8')),
            ctypes.byref(dimension_ptr)
        ))
        return dimension_ptr.value

    def get_global_control_var_dimension(self, name: str) -> int:
        """
        Returns the dimension of a global control variable.

        Parameters
        ----------

        name : str
               Name of the global control variable.

        Returns
        -------

        dimension : int
                    Dimension of the global control variable.
        """
        dimension_ptr = ctypes.c_int()
        HCkE(_hdevengine_lib.HCenGetGlobalCtrlVarDimension(
            self._engine_ptr,
            ctypes.c_char_p(name.encode('utf8')),
            ctypes.byref(dimension_ptr)
        ))
        return dimension_ptr.value

    def set_global_iconic_var(self, name: str, value: HObject) -> None:
        """
        Sets the value of a global iconic variable.

        Parameters
        ----------

        name : str
               Name of the global iconic variable.

        value : HObject
                Name of the global iconic variable.
        """
        HCkE(_hdevengine_lib.HCenSetGlobalIconicVarObject(
            self._engine_ptr,
            ctypes.c_char_p(name.encode('utf8')),
            value._key
        ))

    def set_global_control_var(self, name: str, value: HTupleType) -> None:
        """
        Sets the value of a global control variable.

        Parameters
        ----------

        name : str
               Name of the global control variable.

        value : HObject
                Name of the global control variable.
        """
        with _HTuple(value) as htuple:
            HCkE(_hdevengine_lib.HCenSetGlobalCtrlVarTuple(
                self._engine_ptr,
                ctypes.c_char_p(name.encode('utf8')),
                htuple._tuple_ptr
            ))

    def get_global_iconic_var(self, name: str) -> HObject:
        """
        Returns the value of a global iconic variable.

        Parameters
        ----------

        name : str
               Name of the global iconic variable.

        Returns
        -------

        value : HTupleType
                Value of the global iconic variable.
        """
        hkey = Hkey()
        HCkE(_hdevengine_lib.HCenGetGlobalIconicVarObject(
            self._engine_ptr,
            ctypes.c_char_p(name.encode('utf8')),
            ctypes.byref(hkey)
        ))
        return HObject._copy_from_key(hkey)

    def get_global_control_var(self, name: str) -> HTupleType:
        """
        Returns the value of a global control variable.

        Parameters
        ----------

        name : str
               Name of the global control variable.

        Returns
        -------

        value : HTupleType
                Value of the global control variable.
        """
        with _HTuple(value=None) as htuple:
            HCkE(_hdevengine_lib.HCenGetGlobalCtrlVarTuple(
                self._engine_ptr,
                ctypes.c_char_p(name.encode('utf8')),
                htuple._tuple_ptr
            ))
            return htuple.as_python(as_list=True)


class HDevProgram(object):
    """
    HALCON HDevProgram class.

    Facilitates calling HDevelop programs.

    Notes
    -----
    Access program metadata via these read only member variables:
    - name : str
    - loaded : bool
    - inconic_var_names : List[str]
    - control_var_names : List[str]
    - inconic_var_dimensions : List[int]
    - control_var_dimensions : List[int]
    """
    def __init__(self, name: str) -> None:
        """
        Loads a HDevelop program.

        Parameters
        ----------

        name : str
               Full path to HDevelop program, in format of the
               operating system, including file name.
        """
        self._program_ptr = ctypes.c_ssize_t()

        HCkE(_hdevengine_lib.HCenCreateProgram(
            ctypes.byref(self._program_ptr)
        ))

        HCkE(_hdevengine_lib.HCenLoadProgram(
            self._program_ptr,
            ctypes.c_char_p(name.encode('utf8'))
        ))

        self._setup_program_info()

    def __del__(self) -> None:
        HCkE(_hdevengine_lib.HCenDestroyProgram(self._program_ptr))

    def _setup_program_info(self) -> None:
        """Sets program info member variables."""
        name_ptr = ctypes.c_char_p()
        loaded_ptr = ctypes.c_int()

        with _HTuple(value=None) as inconic_var_names, \
                _HTuple(value=None) as control_var_names, \
                _HTuple(value=None) as inconic_var_dimensions, \
                _HTuple(value=None) as control_var_dimensions:

            HCkE(_hdevengine_lib.HCenGetProgramInfo(
                self._program_ptr,
                ctypes.byref(name_ptr),
                ctypes.byref(loaded_ptr),
                inconic_var_names._tuple_ptr,
                control_var_names._tuple_ptr,
                inconic_var_dimensions._tuple_ptr,
                control_var_dimensions._tuple_ptr
            ))

            # If value is None, that exception is as useful as anything.
            self.name = name_ptr.value.decode('utf-8')  # type: ignore

            self.loaded = loaded_ptr.value == 1

            self.inconic_var_names = cast(
                List[str],
                inconic_var_names.as_python(as_list=True)
            )
            self.control_var_names = cast(
                List[str],
                control_var_names.as_python(as_list=True)
            )
            self.inconic_var_dimensions = cast(
                List[int],
                inconic_var_dimensions.as_python(as_list=True)
            )
            self.control_var_dimensions = cast(
                List[int],
                control_var_dimensions.as_python(as_list=True)
            )

    def get_used_procedure_names(self) -> List[str]:
        """
        Returns the names of used local and external procedures.

        Returns
        -------

        names : List[str]
                Names of used local and external procedures.
        """
        with _HTuple(value=None) as htuple:
            HCkE(_hdevengine_lib.HCenProgGetUsedProcedureNames(
                self._program_ptr,
                htuple._tuple_ptr,
            ))
            return htuple.as_python(as_list=True)  # type: ignore

    def get_local_procedure_names(self) -> List[str]:
        """
        Returns the names of all local procedures.

        Returns
        -------

        names : List[str]
                Names of all local procedures.
        """
        with _HTuple(value=None) as htuple:
            HCkE(_hdevengine_lib.HCenProgGetLocalProcedureNames(
                self._program_ptr,
                htuple._tuple_ptr
            ))
            return htuple.as_python(as_list=True)  # type: ignore

    def compile_used_procedures(self) -> bool:
        """
        JIT compile used procedures.

        Returns
        -------

        all_compiled : bool
                       Whether all used procedures were JIT compiled.

        Notes
        -----
        Compile all procedures that are used by the program and that
        can be compiled with a just-in-time compiler.
        Procedures that could not be compiled are called by the HDevEngine
        interpreter in the usual way.
        To check which procedure could not be compiled and what the
        reason is for that, start HDevelop and check the compilation states
        there.
        """
        all_compiled_ptr = ctypes.c_int()
        HCkE(_hdevengine_lib.HCenProgCompileUsedProcedures(
            self._program_ptr,
            ctypes.byref(all_compiled_ptr)
        ))
        return all_compiled_ptr.value == 1


class HDevProgramCall(object):
    """
    HALCON HDevProgramCall class.

    Specific instance of a HDevelop program call.

    Typical call order:
    execute
    get_control_param_by_name x N
    get_iconic_param_by_name x N
    """
    def __init__(self, program: HDevProgram):
        self._program_call_ptr = ctypes.c_ssize_t()
        HCkE(_hdevengine_lib.HCenCreateProgramCall(
            program._program_ptr,
            ctypes.byref(self._program_call_ptr)
        ))

    def __del__(self) -> None:
        HCkE(_hdevengine_lib.HCenDestroyProgramCall(self._program_call_ptr))

    def execute(self) -> None:
        """
        Executes the program.

        Notes
        -----
        Raises HError if execute is called more than once on a
        HDevProgramCall instance, without reseting in between.
        """
        HCkE(_hdevengine_lib.HCenExecuteProgramCall(self._program_call_ptr))

    def wait_for_debug_connection(self) -> None:
        """
        Stops execution on first line of program.

        Notes
        -----
        This is intended for debugging purposes when you wish to step
        through a specific program call. It only has an effect when a
        debug server is running and it will only stop once.

        Use this in conjunction with start_debug_server.
        """
        HCkE(_hdevengine_lib.HCenSetWaitForDebugConnectionProgramCall(
            self._program_call_ptr,
            ctypes.c_bool(True)
        ))

    def reset(self) -> None:
        """
        Resets the program execution.

        Notes
        -----
        This is mainly for situations when you want to abort execution from
        another thread or possibly free native resources even while some
        instances are still alive.
        """
        HCkE(_hdevengine_lib.HCenResetProgramCall(self._program_call_ptr))

    def get_control_var_by_index(self, idx: int) -> HTupleType:
        """
        Gets the value of a control variable in main.

        Parameters
        ----------

        idx : int
              Index of the control variable.

        Returns
        -------

        value : HTupleType
                Value of the control variable.

        Notes
        -----
        Indices here start at 1 instead of 0.
        """
        assert _is_valid_i32(idx)

        # Call creates a tuple for us, cleanup happens on program cleanup.
        out_tuple_ptr = ctypes.c_ssize_t()
        HCkE(_hdevengine_lib.HCenGetCtrlVarTupleIndex(
            self._program_call_ptr,
            ctypes.c_int(idx),
            ctypes.byref(out_tuple_ptr)
        ))
        return _htuple_to_python(out_tuple_ptr, as_list=True)

    def get_control_var_by_name(self, name: str) -> HTupleType:
        """
        Gets the value of a control variable in main.

        Parameters
        ----------

        name : str
               Name of the control variable.

        Returns
        -------

        value : HTupleType
                Value of the control variable.
        """
        # Call creates a tuple for us, cleanup happens on program cleanup.
        out_tuple = ctypes.c_ssize_t()
        HCkE(_hdevengine_lib.HCenGetCtrlVarTupleName(
            self._program_call_ptr,
            ctypes.c_char_p(name.encode('utf8')),
            ctypes.byref(out_tuple)
        ))
        return _htuple_to_python(out_tuple, as_list=True)

    def get_iconic_var_by_index(self, idx: int) -> HObject:
        """
        Gets the value of an iconic variable in main.

        Parameters
        ----------

        idx : int
              Index of the iconic variable.

        Returns
        -------

        value : HObject
                Value of the iconic variable.

        Notes
        -----
        Indices here start at 1 instead of 0.
        """
        assert _is_valid_i32(idx)

        hkey = Hkey()
        HCkE(_hdevengine_lib.HCenGetIconicVarObjectIndex(
            self._program_call_ptr,
            ctypes.c_int(idx),
            ctypes.byref(hkey)
        ))
        return HObject._copy_from_key(hkey)

    def get_iconic_var_by_name(self, name: str) -> HObject:
        """
        Gets the value of an iconic variable in main.

        Parameters
        ----------

        name : str
               Name of the control variable.

        Returns
        -------

        value : HObject
                Value of the iconic variable.
        """
        hkey = Hkey()
        HCkE(_hdevengine_lib.HCenGetIconicVarObjectName(
            self._program_call_ptr,
            ctypes.c_char_p(name.encode('utf8')),
            ctypes.byref(hkey)
        ))
        return HObject._copy_from_key(hkey)


class HDevProcedure(object):
    """
    HALCON HDevProcedure class.

    Facilitates calling HDevelop procedures.

    Notes
    -----
    Access procedure metadata via these read only member variables:
    - name : str
    - short_description : str
    - loaded : bool
    - input_iconic_param_names : List[str]
    - output_iconic_param_names : List[str]
    - input_control_param_names : List[str]
    - output_control_param_names : List[str]
    - input_iconic_param_dimensions : List[int]
    - output_iconic_param_dimensions : List[int]
    - input_control_param_dimensions : List[int]
    - output_control_param_dimensions : List[int]
    """
    def __init__(self) -> None:
        """
        Initialize new procedure instance.

        Notes
        -----
        This is mostly an implementation detail.

        Use load_external and load_local to acquire useful instances of
        HDevProcedure, eg:
        proc = ha.HDevProcedure.load_external('count_nuts')

        This is left as __init__, so that accidentally creating an instance
        directly via construction does not lead to calling HCenDestroyProcedure
        in the finalizer with invalid _procedure_ptr.
        """
        self._procedure_ptr = ctypes.c_ssize_t()
        HCkE(_hdevengine_lib.HCenCreateProcedure(
            ctypes.byref(self._procedure_ptr)
        ))

    def __del__(self) -> None:
        HCkE(_hdevengine_lib.HCenDestroyProcedure(self._procedure_ptr))

    def _setup_procedure_info(self) -> None:
        """Sets procedure info member variables."""
        name_ptr = ctypes.c_char_p()
        short_des_ptr = ctypes.c_char_p()
        loaded_ptr = ctypes.c_int()

        with _HTuple(value=None) as input_iconic_param_names, \
                _HTuple(value=None) as output_iconic_param_names, \
                _HTuple(value=None) as input_control_param_names, \
                _HTuple(value=None) as output_control_param_names, \
                _HTuple(value=None) as input_iconic_param_dimensions, \
                _HTuple(value=None) as output_iconic_param_dimensions, \
                _HTuple(value=None) as input_control_param_dimensions, \
                _HTuple(value=None) as output_control_param_dimensions:

            HCkE(_hdevengine_lib.HCenGetProcedureInfo(
                self._procedure_ptr,
                ctypes.byref(name_ptr),
                ctypes.byref(short_des_ptr),
                ctypes.byref(loaded_ptr),
                input_iconic_param_names._tuple_ptr,
                output_iconic_param_names._tuple_ptr,
                input_control_param_names._tuple_ptr,
                output_control_param_names._tuple_ptr,
                input_iconic_param_dimensions._tuple_ptr,
                output_iconic_param_dimensions._tuple_ptr,
                input_control_param_dimensions._tuple_ptr,
                output_control_param_dimensions._tuple_ptr
            ))

            # If value is None, that exception is as useful as anything.
            self.name = name_ptr.value.decode('utf-8')  # type: ignore
            short_des = short_des_ptr.value.decode('utf-8')  # type: ignore
            self.short_description = short_des

            self.loaded = loaded_ptr.value == 1

            self.input_iconic_param_names = cast(
                List[str],
                input_iconic_param_names.as_python(as_list=True)
            )
            self.output_iconic_param_names = cast(
                List[str],
                output_iconic_param_names.as_python(as_list=True)
            )
            self.input_control_param_names = cast(
                List[str],
                input_control_param_names.as_python(as_list=True)
            )
            self.output_control_param_names = cast(
                List[str],
                output_control_param_names.as_python(as_list=True)
            )

            self.input_iconic_param_dimensions = cast(
                List[int],
                input_iconic_param_dimensions.as_python(as_list=True)
            )
            self.output_iconic_param_dimensions = cast(
                List[int],
                output_iconic_param_dimensions.as_python(as_list=True)
            )
            self.input_control_param_dimensions = cast(
                List[int],
                input_control_param_dimensions.as_python(as_list=True)
            )
            self.output_control_param_dimensions = cast(
                List[int],
                output_control_param_dimensions.as_python(as_list=True)
            )

    @staticmethod
    def load_external(name: str) -> 'HDevProcedure':
        """
        Loads an external procedure.

        Parameters
        ----------

        name : str
               Name of the procedure to load.

        Returns
        -------

        procedure : HDevProcedure
                    New procedure instance with the external procedure loaded.
        """
        procedure = HDevProcedure()
        HCkE(_hdevengine_lib.HCenLoadProcedure(
            procedure._procedure_ptr,
            ctypes.c_char_p(name.encode('utf8'))
        ))

        procedure._setup_procedure_info()
        return procedure

    @staticmethod
    def load_local(
        program: Union[HDevProgram, str],
        name: str
    ) -> 'HDevProcedure':
        """
        Loads a local procedure.

        Parameters
        ----------

        program : Union[HDevProgram, str]
                  Either an instance of HDevProgram or,
                  the file name of a program.

        name : str
               Name of the procedure to load.

        Returns
        -------

        procedure : HDevProcedure
                    New procedure instance with the local procedure loaded.
        """
        procedure = HDevProcedure()

        if isinstance(program, str):
            HCkE(_hdevengine_lib.HCenLoadProcedureProgramName(
                procedure._procedure_ptr,
                ctypes.c_char_p(program.encode('utf8')),
                ctypes.c_char_p(name.encode('utf8'))
            ))
        else:
            # Duck type program.
            HCkE(_hdevengine_lib.HCenLoadProcedureProgram(
                procedure._procedure_ptr,
                program._program_ptr,
                ctypes.c_char_p(name.encode('utf8'))
            ))

        procedure._setup_procedure_info()
        return procedure

    def get_used_procedure_names(self) -> List[str]:
        """
        Returns the names of all used procedures.

        Returns
        -------

        procedure_names : List[str]
                          Names of used procedures.
        """
        with _HTuple(value=None) as htuple:
            HCkE(_hdevengine_lib.HCenProcGetUsedProcedureNames(
                self._procedure_ptr,
                htuple._tuple_ptr
            ))
            return htuple.as_python(as_list=True)  # type: ignore

    def compile_used_procedures(self) -> bool:
        """
        JIT compile used procedures.

        Returns
        -------

        all_compiled : bool
                       Whether all used procedures were JIT compiled.

        Notes
        -----
        Compile all procedures that are used by the procedure and that
        can be compiled with a just-in-time compiler.
        Procedures that could not be compiled are called by the HDevEngine
        interpreter in the usual way.
        To check which procedure could not be compiled and what the
        reason is for that, start HDevelop and check the compilation states
        there.
        """
        all_compiled_ptr = ctypes.c_int()
        HCkE(_hdevengine_lib.HCenProcCompileUsedProcedures(
            self._procedure_ptr,
            ctypes.byref(all_compiled_ptr)
        ))
        return all_compiled_ptr.value == 1

    def get_info(self, slot: str) -> HTupleType:
        """
        Returns the info of the referenced procedure documentation slot.

        Parameters
        ----------

        slot : str
               Procedure documentation slot, eg. 'short'.

        Returns
        -------

        info : HTupleType
               The documentation info.
        """
        with _HTuple(value=None) as htuple:
            HCkE(_hdevengine_lib.HCenProcGetInfo(
                self._procedure_ptr,
                ctypes.c_char_p(slot.encode('utf8')),
                htuple._tuple_ptr
            ))
            return htuple.as_python(as_list=True)

    def get_param_info(self, name: str, slot: str) -> HTupleType:
        """
        Returns documentation info about parameter.

        Parameters
        ----------

        name : str
               Parameter name.

        slot : str
               Parameter documentation slot, eg. 'short'.

        Returns
        -------

        info : HTupleType
               The documentation info.
        """
        with _HTuple(value=None) as htuple:
            HCkE(_hdevengine_lib.HCenProcGetParamInfo(
                self._procedure_ptr,
                ctypes.c_char_p(name.encode('utf8')),
                ctypes.c_char_p(slot.encode('utf8')),
                htuple._tuple_ptr
            ))
            return htuple.as_python(as_list=True)

    def get_input_iconic_param_info(self, idx: int, slot: str) -> HTupleType:
        """
        Returns documentation info about iconic input parameter.

        Parameters
        ----------

        idx : int
              Index of the iconic input parameter.

        slot : str
               Parameter documentation slot, eg. 'short'.

        Returns
        -------

        info : HTupleType
               The documentation info.
        Notes
        -----
        Indices here start at 1 instead of 0.
        """
        with _HTuple(value=None) as htuple:
            HCkE(_hdevengine_lib.HCenProcGetInputIconicParamInfo(
                self._procedure_ptr,
                idx,
                ctypes.c_char_p(slot.encode('utf8')),
                htuple._tuple_ptr
            ))
            return htuple.as_python(as_list=True)

    def get_output_iconic_param_info(self, idx: int, slot: str) -> HTupleType:
        """
        Returns documentation info about iconic output parameter.

        Parameters
        ----------

        idx : int
              Index of the iconic output parameter.

        slot : str
               Parameter documentation slot, eg. 'short'.

        Returns
        -------

        info : HTupleType
               The documentation info.
        Notes
        -----
        Indices here start at 1 instead of 0.
        """
        with _HTuple(value=0) as htuple:
            HCkE(_hdevengine_lib.HCenProcGetOutputIconicParamInfo(
                self._procedure_ptr,
                idx,
                ctypes.c_char_p(slot.encode('utf8')),
                htuple._tuple_ptr
            ))
            return htuple.as_python(as_list=True)

    def get_input_control_param_info(self, idx: int, slot: str) -> HTupleType:
        """
        Returns documentation info about control input parameter.

        Parameters
        ----------

        idx : int
              Index of the control input parameter.

        slot : str
               Parameter documentation slot, eg. 'short'.

        Returns
        -------

        info : HTupleType
               The documentation info.
        Notes
        -----
        Indices here start at 1 instead of 0.
        """
        with _HTuple(value=None) as htuple:
            HCkE(_hdevengine_lib.HCenProcGetInputCtrlParamInfo(
                self._procedure_ptr,
                idx,
                ctypes.c_char_p(slot.encode('utf8')),
                htuple._tuple_ptr
            ))
            return htuple.as_python(as_list=True)

    def get_output_control_param_info(self, idx: int, slot: str) -> HTupleType:
        """
        Returns documentation info about control output parameter.

        Parameters
        ----------

        idx: int
             Index of the control output parameter.

        slot : str
               Parameter documentation slot, eg. 'short'.

        Returns
        -------

        info : HTupleType
               The documentation info.
        Notes
        -----
        Indices here start at 1 instead of 0.
        """
        with _HTuple(value=None) as htuple:
            HCkE(_hdevengine_lib.HCenProcGetOutputCtrlParamInfo(
                self._procedure_ptr,
                idx,
                ctypes.c_char_p(slot.encode('utf8')),
                htuple._tuple_ptr
            ))
            return htuple.as_python(as_list=True)

    def query_slots(self) -> List[str]:
        """
        Returns list of all possible procedure documentation slots.

        Returns
        -------

        slots : List[str]
                List of all possible procedure documentation slots.
        """
        with _HTuple(value=None) as htuple:
            HCkE(_hdevengine_lib.HCenProcQueryInfo(
                self._procedure_ptr,
                htuple._tuple_ptr
            ))
            return htuple.as_python(as_list=True)  # type: ignore

    def query_param_slots(self) -> List[str]:
        """
        Returns list of all possible parameter documentation slots.

        Returns
        -------

        slots : List[str]
                List of all possible parameter documentation slots.
        """
        with _HTuple(value=None) as htuple:
            HCkE(_hdevengine_lib.HCenProcQueryParamInfo(
                self._procedure_ptr,
                htuple._tuple_ptr
            ))
            return htuple.as_python(as_list=True)  # type: ignore


class HDevProcedureCall(object):
    """
    HALCON HDevProgramCall class.

    Specific instance of a HDevelop program call.

    Typical call order:
    set_input_iconic_param_by_name x N
    set_input_control_param_by_name x N
    execute
    get_output_iconic_param_by_name x N
    get_output_control_param_by_name x N
    """
    def __init__(self, procedure: HDevProcedure):
        self._procedur_call_ptr = ctypes.c_ssize_t()
        HCkE(_hdevengine_lib.HCenCreateProcedureCall(
            procedure._procedure_ptr,
            ctypes.byref(self._procedur_call_ptr)
        ))

    def __del__(self) -> None:
        HCkE(_hdevengine_lib.HCenDestroyProcedureCall(self._procedur_call_ptr))

    def execute(self) -> None:
        """
        Executes the program.

        Notes
        -----
        Raises HDevEngineError if execute fails for some reason.
        """
        HCkE(_hdevengine_lib.HCenExecuteProcedureCall(self._procedur_call_ptr))

    def wait_for_debug_connection(self) -> None:
        """
        Stops execution on first line of procedure.

        Notes
        -----
        This is intended for debugging purposes when you wish to step
        through a specific procedure call. It only has an effect when a
        debug server is running and it will only stop once.

        Use this in conjunction with start_debug_server.
        """
        HCkE(_hdevengine_lib.HCenSetWaitForDebugConnectionProcedureCall(
            self._procedur_call_ptr,
            True
        ))

    def reset(self) -> None:
        """
        Resets the program execution.

        Notes
        -----
        This is mainly for situations when you want to abort execution from
        another thread or possibly free native resources even while some
        instances are still alive.
        """
        HCkE(_hdevengine_lib.HCenResetProcedureCall(self._procedur_call_ptr))

    def set_input_control_param_by_index(
        self,
        idx: int,
        value: HTupleType
    ) -> None:
        """
        Sets control input parameter.

        Parameters
        ----------

        idx : int
              Index of the control input parameter.

        value : HTupleType
                Value of the control input parameter.

        Notes
        -----
        Indices here start at 1 instead of 0.
        """
        assert _is_valid_i32(idx)

        with _HTuple(value) as htuple:
            HCkE(_hdevengine_lib.HCenSetInputCtrlParamTupleIndex(
                self._procedur_call_ptr,
                ctypes.c_int(idx),
                htuple._tuple_ptr
            ))

    def set_input_control_param_by_name(
        self,
        name: str,
        value: HTupleType
    ) -> None:
        """
        Sets control input parameter.

        Parameters
        ----------

        name : str
               Name of the control input parameter.

        value : HTupleType
                Value of the control input parameter.
        """
        with _HTuple(value) as htuple:
            HCkE(_hdevengine_lib.HCenSetInputCtrlParamTupleName(
                self._procedur_call_ptr,
                ctypes.c_char_p(name.encode('utf8')),
                htuple._tuple_ptr
            ))

    def set_input_iconic_param_by_index(
        self,
        idx: int,
        value: HObject
    ) -> None:
        """
        Sets iconic input parameter.

        Parameters
        ----------

        idx : int
              Index of the iconic input parameter.

        value : HObject
                Value of the iconic input parameter.
        Notes
        -----
        Indices here start at 1 instead of 0.
        """
        assert _is_valid_i32(idx)

        HCkE(_hdevengine_lib.HCenSetInputIconicParamObjectIndex(
            self._procedur_call_ptr,
            ctypes.c_int(idx),
            value._key
        ))

    def set_input_iconic_param_by_name(
        self,
        name: str,
        value: HObject
    ) -> None:
        """
        Sets iconic input parameter.

        Parameters
        ----------

        idx : str
              Name of the iconic input parameter.

        value : HObject
                Value of the iconic input parameter.
        """
        HCkE(_hdevengine_lib.HCenSetInputIconicParamObjectName(
            self._procedur_call_ptr,
            ctypes.c_char_p(name.encode('utf8')),
            value._key
        ))

    def get_output_control_param_by_index(self, idx: int) -> HTupleType:
        """
        Gets value of a control output variable.

        Parameters
        ----------

        idx : int
              Index of the control output parameter.

        Returns
        -------

        value : HTupleType
                Value of the control output parameter.

        Notes
        -----
        Indices here start at 1 instead of 0.
        """
        assert _is_valid_i32(idx)

        # Call creates a tuple for us, cleanup happens on procedure cleanup.
        out_tuple_ptr = ctypes.c_ssize_t()
        HCkE(_hdevengine_lib.HCenGetOutputCtrlParamTupleIndex(
            self._procedur_call_ptr,
            ctypes.c_int(idx),
            ctypes.byref(out_tuple_ptr)
        ))
        return _htuple_to_python(out_tuple_ptr, as_list=True)

    def get_output_control_param_by_name(self, name: str) -> HTupleType:
        """
        Gets value of a control output variable.

        Parameters
        ----------

        name : str
               Name of the control output parameter.

        Returns
        -------

        value : HTupleType
                Value of the control output parameter.
        """
        # Call creates a tuple for us, cleanup happens on procedure cleanup.
        out_tuple_ptr = ctypes.c_ssize_t()
        HCkE(_hdevengine_lib.HCenGetOutputCtrlParamTupleName(
            self._procedur_call_ptr,
            ctypes.c_char_p(name.encode('utf8')),
            ctypes.byref(out_tuple_ptr)
        ))
        return _htuple_to_python(out_tuple_ptr, as_list=True)

    def get_output_iconic_param_by_index(self, idx: int) -> HObject:
        """
        Gets value of a iconic output variable.

        Parameters
        ----------

        idx : int
              Index of the iconic output parameter.

        Returns
        -------

        value : HObject
                Value of the iconic output parameter.

        Notes
        -----
        Indices here start at 1 instead of 0.
        """
        assert _is_valid_i32(idx)

        hkey = Hkey()
        HCkE(_hdevengine_lib.HCenGetOutputIconicParamObjectIndex(
            self._procedur_call_ptr,
            ctypes.c_int(idx),
            ctypes.byref(hkey)
        ))
        return HObject._copy_from_key(hkey)

    def get_output_iconic_param_by_name(self, name: str) -> HObject:
        """
        Gets value of a iconic output variable.

        Parameters
        ----------

        name : str
               Name of the control output parameter.

        Returns
        -------

        value : HObject
                Value of the iconic output parameter.

        Notes
        -----
        Indices here start at 1 instead of 0.
        """
        hkey = Hkey()
        HCkE(_hdevengine_lib.HCenGetOutputIconicParamObjectName(
            self._procedur_call_ptr,
            ctypes.c_char_p(name.encode('utf8')),
            ctypes.byref(hkey)
        ))
        return HObject._copy_from_key(hkey)


# --- Exported Functions ---


def HCkE(inital_error_code: int) -> None:
    """Check the error number and raise exception if no ok."""
    if inital_error_code == 2:  # C constant H_MSG_OK
        return

    try:
        category_ptr = ctypes.c_int()
        message_ptr = ctypes.c_char_p()
        proc_name_ptr = ctypes.c_char_p()
        line_text_ptr = ctypes.c_char_p()
        line_number_ptr = ctypes.c_int()
        user_data_handle_ptr = ctypes.c_ssize_t()

        error_code = _hdevengine_lib.HCenGetLastException(
            ctypes.byref(category_ptr),
            ctypes.byref(message_ptr),
            ctypes.byref(proc_name_ptr),
            ctypes.byref(line_text_ptr),
            ctypes.byref(line_number_ptr),
            ctypes.byref(user_data_handle_ptr)
        )

        # If a HDevEngine specific error occured inital_error_code is -1,
        # and error_code > 20000.

        category = category_ptr.value

        # If value is None, that exception is as useful as anything.
        message = message_ptr.value.decode('utf-8')  # type: ignore
        procedure_name = proc_name_ptr.value.decode('utf-8')  # type: ignore
        line_text = line_text_ptr.value.decode('utf-8')  # type: ignore

        line_number = line_number_ptr.value
        user_data = _htuple_to_python(user_data_handle_ptr, as_list=True)

    except Exception as exc:
        raise HDevEngineError(
            error_code=0,
            category=0,
            message=f'Error handling exception: \'{exc}\'',
            procedure_name='',
            line_text='',
            line_number=0,
            user_data=[]
        )
    finally:
        _hdevengine_lib.HCenReleaseLastException()

    raise HDevEngineError(
        error_code,
        category,
        message,
        procedure_name,
        line_text,
        line_number,
        user_data
    )


# --- Private Implementation Details ---


# Load dynamic HDevEngine library.
_hdevengine_lib: Final[ctypes.CDLL] = \
    load_native_lib(hdevengine_native_lib_names)
