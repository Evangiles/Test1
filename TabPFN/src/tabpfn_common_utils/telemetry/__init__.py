# Telemetry stubs for TabPFN

def track_model_call(**kwargs):
    def decorator(func):
        def wrapper(*args, **fkwargs):
            return func(*args, **fkwargs)
        return wrapper
    return decorator


def set_model_config(*args, **kwargs):
    """Stub for set_model_config - does nothing."""
    pass

