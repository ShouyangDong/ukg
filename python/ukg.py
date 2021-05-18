from tvm import tir 
from stmt import KStmt
from tvm.tir import stmt as _stmt
from tvm.tir import expr as _expr
from tvm.tir.ir_builder import WithScope

def and_(*args, span=None):
    """Create a new expression of the intersection of all conditions in the
      arguments

    Parameters
    ----------
    args : list
        List of symbolic boolean expressions

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    expr: Expr
        Expression
    """
    return tir.all(args, span)

def or_(*args, span):
    """Create a new experssion of the union of all conditions in the arguments

    Parameters
    ----------
    args : list
        List of symbolic boolean expressions

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    expr: Expr
        Expression
    """
    return tir.any(args, span)

def if_(cond):
    """Construct an IF branch.

    The usage is the same as Python `if` statement. Namely, a single `if`
    statement without the `else` branch if allowed. In addition, we cannot 
    use `else` and `elif` without an `if` statement. Finally, an `else` 
    statement must be preceded by either an `if` or `elif` statement.

    Parameters
    ----------
    cond : Expr
        The condition of the `if` statement.

    Returns
    -------
    None

    Examples
    --------
    .. code-block:: python

        def example(x):
            with ukg.if_(A[x] < 3):
                # do something

            with ukg.elif_(A[x] < 6):
                # do something

            with ukg.else_():
                # do something
    """
    ukg_stmt = KStmt.get_current()
    ukg_stmt.stmt_stack.append([])
    def _exit_cb():
        stmt = ukg_stmt.pop_stmt()
        ukg_stmt.has_break = False
        ukg_stmt.emit(_stmt.IfThenElse(cond, stmt, None))
    return WithScope(None, _exit_cb)

def else_():
    """Construct an ELSE branch.

    Parameters
    ----------

    Returns
    -------
    None

    See Also
    --------
    if_
    """
    ukg_stmt = KStmt.get_current()
    if not ukg_stmt.stmt_stack[-1]:
        raise RuntimeError("else_ can only follow an if_")
    prev = ukg_stmt.stmt_stack[-1][-1]
    if not isinstance(prev, _stmt.IfThenElse) or prev.else_case:
        raise RuntimeError("else_ can only follow an if_")
    ukg_stmt.stmt_stack[-1].pop()
    ukg_stmt.stmt_stack.append([])
    def _exit_cb():
        stmt = ukg_stmt.pop_stmt()
        ukg_stmt.has_break = False
        ukg_stmt.emit(ukg_stmt.replace_else(prev, stmt))
    return WithScope(None, _exit_cb)

def elif_(cond):
    """Construct an ELIF branch.

    Parameters
    ----------
    cond : Expr
        The condition of the elif branch.

    Returns
    -------
    None 

    See Also
    --------
    if_
    """
    ukg_stmt = KStmt.get_current()
    if not ukg_stmt.stmt_stack[-1]:
        raise RuntimeError("elif_ can only follow an if_")
    prev = ukg_stmt.stmt_stack[-1][-1]
    if not isinstance(prev, _stmt.IfThenElse):
        raise RuntimeError("elif_ can only follow an if_")

    ukg_stmt.stmt_stack[-1].pop()
    ukg_stmt.stmt_stack.append([])
    def _exit_cb():
        stmt = ukg_stmt.pop_stmt()
        ukg_stmt.has_break = False
        ukg_stmt.emit(ukg_stmt.replace_else(prev, _stmt.IfThenElse(cond, stmt, None)))
    return WithScope(None, _exit_cb)

def for_(begin, end, name="i", dtype="int32", kind="serial"):
    """Create a for iteration scope.

    Parameters
    ----------
    begin : Expr
        The min iteration scope.

    end : Expr
        The end iteration scope

    name : str, optional
        The name of iteration variable, if no input names,
        using typical index names i, j, k, then i_nidx

    dtype : str, optional
        The data type of iteration variable.

    kind : str, optional
        The special tag on the for loop.

    Returns
    -------
    loop_scope : With.Scope of Var
        The for scope, when enters returns loop_var

    Examples
    --------
    .. code-block:: python

        # example 1 - basic usage
        with ukg.for_(0, 5) as i:
            # i = [0, 1, 2, 3, 4]
        
        # example 2 - negative step
        with ukg.for_(5, 0, -1) as i:
            # i = [5, 4, 3, 2, 1]

        # example 3 - larger step
        with ukg.for_(0, 5, 2) as i:
            # i = [0, 2, 4]

        # example 4 - arbitrary bound
        with ukg.for_(-4, -8, -2) as i:
            # i = [-4, -6]
    """
    ukg_stmt = KStmt.get_current()
    if name == "i":
        name = chr(ord(name) + ukg_stmt.nidx) if ukg_stmt.nidx < 3 else name + "_" + str(ukg_stmt.nidx - 3)
        ukg_stmt.nidx += 1
    ukg_stmt.stmt_stack.append([])
    loop_var = _expr.Var(name, dtype=dtype)
    extent = end if begin == 0 else (end - begin)
    KStmt.for_level += 1
    def _exit_cb():
        if kind == "serial":
            kind_id = _stmt.ForKind.SERIAL
        elif kind == "parallel":
            kind_id = _stmt.ForKind.PARALLEL
        elif kind == "vectorize":
            kind_id = _stmt.ForKind.VECTORIZED
        elif kind == "unroll":
            kind_id = _stmt.ForKind.UNROLLED
        else:
            raise ValueError("Unknown kind")
        KStmt.has_break = False
        KStmt.for_level += 1
        ukg_stmt.emit(_stmt.For(loop_var, begin, extent, kind_id, ukg_stmt))

    return WithScope(loop_var, _exit_cb)

def while_(condition):
    """Create a while loop scope.

    Parameters
    ----------
    condition : Expr
        The termination condition.

    Returns
    -------
    loop_scope : With.Scope of Var
        The while scope.

    Examples
    --------
    .. code-block:: python

        with ukg.while_(A[x] > 5):
            # do something
    """
    ukg_stmt = KStmt.get_current()
    ukg_stmt.stmt_stack.append([])
    ukg_stmt.for_level += 1

    def _exit_cb():
        stmt = ukg_stmt.pop_stmt()
        ukg_stmt.has_break = False
        ukg_stmt.for_level -= 1
        ukg_stmt.emit(_stmt.While(condition, stmt))
    return WithScope(None, _exit_cb)

def break_():
    """
    Construct a BREAK statement.
    This DSL can only be used inside a `while` loop or a `for loop`. Moreover,
    it is not allowed to have tracing statements after the `break`.
    Parameters
    ----------
    Returns
    -------
    None
    Examples
    --------
    .. code-block:: python

        # example 1 - inside a for loop
        with ukg.for_(0, 5) as i:
            with ukg.if_(A[i] > 5):
                ukg.break_()

        # example 2 - inside a while loop
        with ukg.while_(A[i] > 5):
            with ukg.if_(A[i] > 10):
                ukg.break_()
    """
    if not KStmt.get_current().for_level:
        raise RuntimeError("break_ must be used inside a for/while loop")
    KStmt.get_current().emit(_stmt.Break())
    KStmt.get_current().has_break = True

def def_(shapes, dtypes=None, ret_dtype=None, name=None, arg_names=None):
    """
    Define a UKG function from a Python function.
    This TCP is used as a Python decorator. The function defined with UKG
    is not inlined by default. Users need to provide the shapes of the
    arguments, while the data types of the arguments and the returned data
    type are optional. This DSL helps make the algorithm more organized and
    could potentially reduce the memory usage by reusing the same
    functionality. Users can later on use compute primitives to decide whether
    to inline these functions or not.
    After specifying a Python function to be a UKG function, users can
    use the function just like using a Python function. We can also apply
    optimization primitives.

    Parameters
    ----------
    shapes : list of tuple
        The shapes of the arguments

    dtypes : list of Type, optional
        The data types of the argument

    ret_dtype : Type, optional
        The data type of the returned value

    name : str, optional
        The name of the function. By default, it is the same as the Python
        function

    Returns
    -------
    None

    Examples
    --------
    .. code-block:: python

        # example 1 - no return
        A = ukg.placeholder((10,))
        B = ukg.placeholder((10,))
        x = ukg.placeholder(())

        @ukg.def_([A.shape, B.shape, x.shape])
        def update_B(A, B, x):
            with ukg.for_(0, 10) as i:
                B[i] = A[i] + x

        # directly call the function
        update_B(A, B, x)

        # example 2 - with return value
        @ukg.def_([(10,), (10,), ()])
        def ret_add(A, B, x):
            ukg.return_(A[x] + B[x])
        # use inside a compute API
        A = ukg.placeholder((10,))
        B = ukg.placeholder((10,))
        C = ukg.compute((10,), lambda x: ret_add(A, B, x))
        D = ukg.compute((10,), lambda x: ret_add(A, C, x))
    """
    def decorator(fmodule, shapes=shapes, dtypes=dtypes, ret_dtype=ret_dtype, name=name, arg_names=arg_names):
        name = name if name is not None else fmodule.__name__
        code = fmodule.__code__
        names = code.co_varnames
        if arg_names is not None:
          names = list(names)
          for i in range(len(arg_names)):
            names[i] = arg_names[i]
          names = tuple(names)
        nargs = code.co_argcount

        with KStmt(name) as s:
            # prepare names
            new_names = [s.name_with_prefix + "." + name_ for name_ in names]
            # prepare dtypes
            ukg_dtypes = []
            if dtypes is None:
                dtypes = []
                for name_ in new_names:
                    dtypes.append(util.get_tvm_dtype(None, name_))
                    ukg_dtypes.append(util.get_dtype(None, name_))
            elif isinstance(dtypes, list):
                if len(dtypes) != nargs:
                    raise RuntimeError("The number of data types does not match the of arguments")
                for (name_, dtype_) in zip(new_names, dtypes):
                    dtypes.append(util.get_tvm_dtype(dtype_, name_))
                    ukg_dtypes.append(util.get_dtype(dtype_, name_))
                dtypes = dtypes[int(len(dtypes)/2):]
            else:
                dtype = util.get_tvm_dtype(dtypes)
                dtypes = []
                for name_ in new_names:
                    dtypes.append(util.get_tvm_dtype(dtype, name_))
            ret_dtype = util.get_tvm_dtype(ret_dtype, s.name_with_prefix)
            # prepare inputs for IR generation
            inputs = []
            inputs_tvm = []
            arg_shapes, arg_dtypes, arg_tensors = [], [], []
            for shape, name_, dtype, htype in zip(shapes, new_names, dtypes, ukg_dtypes):
                if shape == ():
                    var_ = placeholder((), name_, dtype)
                    inputs.append(var_)
                    inputs_tvm.append(var_.var)
                    arg_shapes.append([1])
                    arg_dtypes.append(dtype)
                else: # tensor inputs (new bufs)
                    placeholder_ = placeholder(shape, name_, htype)
                    inputs.append(placeholder_)
                    inputs_tvm.append(placeholder_.buf.data)
                    arg_shapes.append(list(shape))
                    arg_dtypes.append(dtype)
                    arg_tensors.append(placeholder_.op)

            s.ret_dtype = ret_dtype
            s._module = True
            s._inputs = inputs
            fmodule(*inputs)
            lhs = []
            for tensor in s.lhs_tensors:
                try:
                    lhs.append(inputs.index(tensor))
                except ValueError:
                    pass
            ret_void = _expr.UIntImm("uint1", 0) if s.has_return else _expr.UIntImm("uint1", 1)
            body = s.pop_stmt()

            s.stmt_stack.append([])
            s.emit(_stmt.KernelDef(inputs_tvm, arg_shapes, arg_dtypes, arg_tensors,
                                   body, ret_void, ret_dtype, name, []))
            for name_, i in zip(names, inputs):
                s.var_dict[name_] = i
            s.input_ukg_stmts.clear()

        return Module(shapes, names, name, not s.has_return, lhs, ret_dtype)
    return decorator

def return_(val):
    """Return an expression within a function.
    This DSL should be used within a function definition. The return type can
    only be an expression.

    Parameters
    ----------
    val : Expr
        The returned expression

    Returns
    -------
    None

    See Also
    --------
    UKG.compute, def_

    Examples
    --------

    .. code-block:: python
        # example 1 - using with a compute API
        A = ukg.placeholder((10,))
        def compute_out(x):
            with ukg.if_(A[x]>0):
                ukg.return_(1)
            with ukg.else_():
                ukg.return_(0)
        B = ukg.compute(A.shape, compute_out)

        # example 2 - using with a UKG function
        A = ukg.placeholder((10,))
        @ukg.def_([A.shape, ()])
        def compute_out(A, x):
            with ukg.if_(A[x]>0):
                ukg.return_(1)
            with ukg.else_():
                ukg.return_(0)
        B = ukg.compute(A.shape, lambda x: compute_out(A, x))
    """
    if not KStmt.get_len():
        raise RuntimeError("Imperative DSL must be used with other compute APIs")
    ukg_stmt = KStmt.get_current()
    dtype = util.get_tvm_dtype(ukg_stmt.ret_dtype)
    ukg_stmt.emit(_stmt.Return(_expr.Cast(dtype, val)))
    ukg_stmt.has_return = True

