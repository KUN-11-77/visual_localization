from typing import Type, Dict

_RETRIEVAL_REGISTRY: Dict[str, Type] = {}
_DETECTOR_REGISTRY: Dict[str, Type] = {}
_MATCHER_REGISTRY: Dict[str, Type] = {}


def register_retrieval(name: str):
    def decorator(cls):
        _RETRIEVAL_REGISTRY[name] = cls
        return cls
    return decorator


def register_detector(name: str):
    def decorator(cls):
        _DETECTOR_REGISTRY[name] = cls
        return cls
    return decorator


def register_matcher(name: str):
    def decorator(cls):
        _MATCHER_REGISTRY[name] = cls
        return cls
    return decorator


def build_retrieval(config: dict):
    cls = _RETRIEVAL_REGISTRY[config["method"]]
    return cls(config.get("params", {}))


def build_detector(config: dict):
    cls = _DETECTOR_REGISTRY[config["method"]]
    return cls(config.get("params", {}))


def build_matcher(config: dict):
    cls = _MATCHER_REGISTRY[config["method"]]
    return cls(config.get("params", {}))


# Import implementations to trigger @register_* decorators
import extensions.retrievals.netvlad      # noqa: E402
import extensions.retrievals.eigenplaces  # noqa: E402
import extensions.retrievals.cricavpr     # noqa: E402
import extensions.detectors.sift          # noqa: E402
import extensions.detectors.superpoint    # noqa: E402
import extensions.detectors.aliked        # noqa: E402
import extensions.matchers.nn             # noqa: E402
import extensions.matchers.superglue      # noqa: E402
import extensions.matchers.lightglue      # noqa: E402
