from contextvars import ContextVar

Device: ContextVar[str] = ContextVar("device", default="cpu")
