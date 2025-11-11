# Telemetry stubs for TabPFN

def track_model_call(*call_args, **call_kwargs):
    """Decorator factory stub for track_model_call."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def set_model_config(*args, **kwargs):
    """Stub for set_model_config - does nothing."""
    pass

