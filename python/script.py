import inspect

def script(script_in):
    """Decorate a python function or class as ukg script.

    The tvm function or parsing support parsing to the internal TIR.

    Returns
    -------
    output : Union[Function, Module]
        The Function or Module in IR.
    """

    if inspect.isfunction(script_in):
        result = from_source(script_in)
    else:
        raise TypeError("Only function and class definitions are supported.")
    result.__name__ = script_in.__name__
    result.__qualname__ = script_in.__qualname__
    return result

def from_source(src):
    """Parse function or string into TIR.

    If possible, pass the ukg script in as a function so that line numbers and
    filename will be accurate.

    Parameters
    ----------
    src : [str, function, class]
        Pruned source of original script

    Returns
    -------
    functions : PrimFunc or IRModule
        The PrimFunc or IRModule in IR.
    """
    if isinstance(src, str):
        start_line = 0
    else:
        _, start_line = inspect.getsourcelines(src)
    parser = TVMScriptParser(start_line)
    return to_ast(src, TVMDiagnosticCtx(), parser)