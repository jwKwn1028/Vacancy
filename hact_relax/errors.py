class _ContextError(RuntimeError):
    def __init__(self, message: str, **context: object):
        super().__init__(message)
        self.context = dict(context)


class SCFNotConverged(_ContextError):
    pass


class BranchFlip(_ContextError):
    pass


class SubspaceContinuityError(_ContextError):
    pass


class GeometryOptimizationNotConverged(_ContextError):
    pass
