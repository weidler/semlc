import functools
from typing import Union, Callable, Tuple

from layers import SemLC, AdaptiveSemLC, ParametricSemLC, SingleShotSemLC, LRN, CMapLRN, GaussianSemLC
from networks.base import BaseNetwork


def prepare_lc_builder(setting: str, widths: Tuple[float, float], ratio: float, damping: float) -> Union[
    None, Callable[..., BaseNetwork]]:
    """Return a partial function of the semantic lateral connectivity layer requested by name."""

    if setting is None or setting == "none":
        return None

    setting = setting.strip().lower()

    # SEMLC
    if setting in ["semlc"]:
        return functools.partial(SemLC, widths=widths, ratio=ratio, damping=damping)
    elif setting in ["adaptive-semlc", "adaptivesemlc"]:
        return functools.partial(AdaptiveSemLC, widths=widths, ratio=ratio, damping=damping)
    elif setting in ["parametric-semlc", "parametricsemlc"]:
        return functools.partial(ParametricSemLC, widths=widths, ratio=ratio, damping=damping)
    elif setting in ["singleshot-semlc", "singleshotsemlc"]:
        return functools.partial(SingleShotSemLC, widths=widths, ratio=ratio, damping=damping)

    # COMPETITORS
    elif setting in ["lrn"]:
        return functools.partial(LRN)
    elif setting in ["cmap-lrn", "cmaplrn"]:
        return functools.partial(CMapLRN)
    elif setting in ["gaussian-semlc", "gaussiansemlc"]:
        return functools.partial(GaussianSemLC, widths=widths, ratio=ratio, damping=damping)
    else:
        raise NotImplementedError(f"LC layer construction for given layer setting {setting} not implemented.")