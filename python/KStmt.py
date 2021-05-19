
from tvm.tir import stmt as _stmt
from tvm.tir import expr as _expr

class KStmt(object):
    """Create a KStmt in the algorithm.
    KStmt is needed when an imperative DSL block is not used within any other
    compute APIs. We can further use the created KStmt to help us schedule
    the imperative components within it. It can also be used to describe a
    higher level of computation hierarchy. For example, we can wrap several
    compute APIs into a single stmt.
    Parameters
    ----------
    name : str, optional
        The name of the KStmt

    Attributes
    ----------
    stmt_stack : list of list of Stmt
        Store all statements. There are two levels. The outer level is
        for different scopes of statement. The inner level is for
        different statements

    var_dict : dict(str, _Var)
        A dictionary whose key is the name of the variable
        and the value is the variable itself. This enables users to
        access a variable inside a KStmt via a Python attribute

    axis_list : list of IterVar
        A list of axes appeared in this KStmt

    has_break : bool
        Set to `True` if there is a `break` statement within the KStmt

    has_return : bool
        Set to `True` if there is a `return` statement within the KStmt

    ret_dtype : Type
        The returned data type. Only exists for `heterocl.compute`

    for_level : int
        The level of a loop nest where the current statement is.

    for_id : int
        An index used to label the unnamed axes

    input_KStmts : set of KStmt
        A set of KStmts that are the input to the KStmt

    lhs_tensors : set of Tensor
        The tensors that are updated at the left-hand side

    last_subKStmts : set of KStmt
        A set of sub-KStmts that are last used in the current KStmt

    name_with_prefix : str
        The full name of the KStmt. This is used when two KStmts at different
        levels share the same name

    Examples
    --------
    .. code-block:: python
        A = ukg.placeholder((10,))
        with ukg.KStmt():
            A[0] = 5
            with ukg.for_(1, 10) as i:
                A[i] = A[i-1] * 2
    """
    _current = []
    """Store all living `KStmt`. The newest is at the end."""

    def __init__(self, name=None, dtype=None, shape=()):
        # # Attributes related to a single KStmt
        # self.name = util.get_name("KStmt", name)
        # # Create non-duplicateing KStmt names
        # while self.name in Schedule.KStmt_names:
        #     self.name += "_"
        # Schedule.KStmt_names.add(self.name)

        self.stmt_stack = [[]]
        self.var_dict = {}
        self.axis_list = []
        self.has_break = False
        self.has_return = False
        self.ret_dtype = None
        self.for_level = 0
        self.for_ID = 0
        self.subKStmts = []
        # Attributes for ExternModule
        self.ext_ip_name = None
        self.inputs = []
        self.port_types = []
        self.source = []
        self.command  = []
        # Attributes for cross-KStmt relation
        self.input_KStmts = set([])
        self.lhs_tensors = set([])
        self.last_subKStmts = set([])
        self.name_with_prefix = self.name if KStmt.get_len() == 0 \
                                    else KStmt.get_current().name_with_prefix + "." + self.name
        # Attribute for constant tensor
        self.init_values = None
        self.is_const = False
        # Private attributes for building a KStmt
        self._op = None
        self._ukg_dtype = dtype
        self._dtype = dtype
        self._buf = tvm_api.decl_buffer(shape, self._dtype, self.name)
        self._shape = self._buf.shape

    def __enter__(self):
        KStmt._current.append(self)
        return self

    def __exit__(self, ptype, value, trace):
        # update input_KStmts: the union of the last subKStmts and original input KStmts
        # collected in the KStmt
        self.input_KStmts = self.last_subKStmts.union(self.input_KStmts)
        # create the output operation
        input_ops = [i._op for i in self.input_KStmts]
        input_bufs = [i._buf for i in self.input_KStmts]
        output_bufs = [self._buf]
        body = self.pop_stmt()
        KStmt._current.pop()
        if self.init_values is not None:
            op = _ExternOp(self.name, "", self.axis_list, input_ops,
                           input_bufs, output_bufs, body,
                           self.init_values, self.is_const)
        else:
            op = _ExternOp(self.name, "", self.axis_list, input_ops,
                           input_bufs, output_bufs, body)
        self._op = op.output(0)
        # update last_update KStmts
        # if this KStmt is a subKStmt of other KStmts
        if KStmt._current:
            superKStmt = KStmt._current[-1]
            # add attribute statement for later KStmt insertion
            superKStmt.emit(
                lambda x: _stmt.AttrStmt(self._buf, "attach_scope",
                                         _stmt.StringImm(superKStmt.name), x))
            # update the input KStmts of the superKStmt:
            # input_KStmts = original input KStmts + current input KStmts - last subKStmts
            superKStmt.input_KStmts = superKStmt.input_KStmts.union(self.input_KStmts)
            superKStmt.input_KStmts.difference_update(superKStmt.last_subKStmts)
            # update the last subKStmts of the superKStmt:
            # last_subKStmts = original subKStmts + current KStmt - inputs of current KStmt
            superKStmt.last_subKStmts.add(self)
            superKStmt.last_subKStmts.difference_update(self.input_KStmts)
            # update lhs_tensors:
            # lhs_tensors = original tensors + lhs tensors of current KStmt
            superKStmt.lhs_tensors.update(self.lhs_tensors)
            # update var_dict
            superKStmt.var_dict[self.name] = self
            # update prefix
            self.name_with_prefix = superKStmt.name_with_prefix + "." + self.name
            # update superKStmt's subKStmts
            superKStmt.subKStmts.append(self)
        # Otherwise update the list of KStmts globally
        else:
            Schedule.KStmt_ops.append(self)
            Schedule.last_KStmts.add(self)
            Schedule.last_KStmts -= self.input_KStmts

    def __repr__(self):
        return self.name

    def __getattr__(self, name):
        try:
            if name in self.var_dict:
                return self.var_dict[name]
            else:
                # return KStmt and target tensor op
                for tensor in self.lhs_tensors:
                    if tensor.name == name:
                        return (self, tensor._tensor)
                # check tensors in input KStmts
                for KStmt in self.input_KStmts:
                    if KStmt.name == name:
                        return (self, KStmt._op)
                # check tensors in input_KStmt.lhs
                for KStmt in self.input_KStmts:
                    lhs = KStmt.lhs_tensors
                    for tensor in lhs:
                        if tensor.name == name:
                            return (self, tensor._tensor)
                raise ValueError("Member " + name + \
                    " not found in " + str(self.lhs_tensors) + " or " + \
                    str(self.input_KStmts))
        except KeyError:
            raise ValueError("Uknown member " + name + " of " + self.name)

    def emit(self, stmt):
        """Insert statements to the current KStmt."""
        if self.has_break:
            raise RuntimeError("Cannot write statements after break")
        self.stmt_stack[-1].append(stmt)

    def replace_else(self, if_stmt, else_stmt):
        """Add an ELSE or ELIF branch to an existing IF or ELIF branch."""
        assert isinstance(if_stmt, _stmt.IfThenElse), "Wrong if statement"
        if isinstance(if_stmt.else_case, _stmt.IfThenElse):
            return _stmt.IfThenElse(if_stmt.condition, if_stmt.then_case,
                                    self.replace_else(if_stmt.else_case, else_stmt))
        return _stmt.IfThenElse(if_stmt.condition, if_stmt.then_case, else_stmt)

    def pop_stmt(self):
        """Create a statement from the statements within current KStmt."""
        stmts = self.stmt_stack.pop()
        if not stmts or callable(stmts[-1]):
            stmts.append(_stmt.Evaluate(0))
        stmt = stmts[-1]
        for s in reversed(stmts[:-1]):
            if callable(s):
                stmt = s(stmt)
            else:
                assert isinstance(s, _stmt.Stmt)
                stmt = _stmt.Block(s, stmt)
        return stmt

    @staticmethod
    def get_current():
        """Get the current KStmt."""
        return KStmt._current[-1]

    @staticmethod
    def get_len():
        """Get the level of KStmts."""
        return len(KStmt._current)

    @property
    def axis(self):
        """Get the axes of the KStmt."""
        return self._op.op.axis




