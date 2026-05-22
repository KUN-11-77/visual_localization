import time
from functools import wraps
import numpy as np

_timing_log = {}


def timed(stage_name: str):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            _timing_log.setdefault(stage_name, []).append(elapsed_ms)
            return result
        return wrapper
    return decorator


def dump_timing(output_path: str):
    import json
    summary = {k: {"mean_ms": float(np.mean(v)), "std_ms": float(np.std(v))}
                for k, v in _timing_log.items()}
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)