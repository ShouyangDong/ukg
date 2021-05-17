"""Statement AST Node IN UKG.

Each statement node have subfields that can be visited from python side.
"""
import tvm
import tvm._ffi
from tvm import _ffi_api
from tvm.stmt import Stmt 

@tvm._ffi.register_object("tir.Break")
class BreakStmt(Stmt):
    """Break node.

    Parameters
    ----------
    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, span=None):
        self.__init__handle__by_constructor__(
            _ffi_api.Break,
            span
        )

@tvm._ffi.register_object("tir.Return")
class ReturnStmt(Stmt):
    """ReturnStmt node.
    Parameters
    ----------
    value : PrimExpr
        The return value.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, value, span=None):
        self.__init__handle__by_constructor__(
            _ffi_api.Return,
            value,
            span
        )

@tvm._ffi.register_object("tir.KernelStmt")
class KernelStmt(Stmt):
    """KernelStmt node.

    Parameters
    ----------
    args : Array[PrimExpr]
        The input argments of kernel function.
    
    name : Str
        The name of def function.
    
    annotate_keys : Array[PrimExpr]
        The annotate keys of kernel def function.
    
    annotate_values : Array[PrimExpr]
        The annotate value of kernel def function.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, args=None, name=None, annotate_keys=None, annotate_values=None, span=None):
        self.__init__handle__by_constructor__(
            _ffi_api.KernelStmt,
            args,
            name,
            annotate_keys,
            annotate_values,
            span
        )

@tvm._ffi.register_object("tir.KernelDef")
class KernelDef(Stmt):
    """KernelDef node.

    Parameters
    ----------
    args : Array[PrimExpr]
        The input argments of kernel function.

    arg_shapes : 2D array
        The shape of input argments.
    
    arg_types : Array<PrimExpr>
        The type of input argments.
    
    arg_buffers : Array[Buffer]
        The Buffer of input argments.
    
    body : Stmt
        The buffer of kernel function.
    
    ret_void : PrimExpr
        The return Expr

    ret_type : Type
        The return type.
    
    name : str
        The name of kernel function.

    attributes : 2D array
        The attribute of kernel funciton.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, args, arg_shapes, args_types, arg_buffers, body, ret_void, ret_type, attributes, span=None):
        self.__init__handle__by_constructor__(
            _ffi_api.KernelDef,
            args, 
            arg_shapes, 
            args_types, 
            arg_buffers, 
            body, 
            ret_void, 
            ret_type, 
            attributes, 
            span
        )