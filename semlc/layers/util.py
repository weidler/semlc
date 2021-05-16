import functools
from typing import Union, Callable

from layers import SemLC, AdaptiveSemLC, ParametricSemLC, SingleShotSemLC, LRN, CMapLRN, GaussianSemLC
from networks.base import BaseNetwork


def prepare_lc_builder(setting: str, ricker_width: float, ricker_damp: float) -> Union[
    None, Callable[..., BaseNetwork]]:
    """Return a partial function of the semantic lateral connectivity layer requested by name."""

    if setting is None or setting == "none":
        return None

    setting = setting.strip().lower()

    # SEMLC
    if setting in ["semlc"]:
        return functools.partial(SemLC, ricker_width=ricker_width, ricker_damp=ricker_damp)
    elif setting in ["adaptive-semlc", "adaptivesemlc"]:
        return functools.partial(AdaptiveSemLC, ricker_width=ricker_width, ricker_damp=ricker_damp)
    elif setting in ["parametric-semlc", "parametricsemlc"]:
        return functools.partial(ParametricSemLC, ricker_width=ricker_width, ricker_damp=ricker_damp)
    elif setting in ["singleshot-semlc", "singleshotsemlc"]:
        return functools.partial(SingleShotSemLC, ricker_width=ricker_width, ricker_damp=ricker_damp)

    # COMPETITORS
    elif setting in ["lrn"]:
        return functools.partial(LRN)
    elif setting in ["cmap-lrn", "cmaplrn"]:
        return functools.partial(CMapLRN)
    elif setting in ["gaussian-semlc", "gaussiansemlc"]:
        return functools.partial(GaussianSemLC, ricker_width=ricker_width, ricker_damp=ricker_damp)
    else:
        raise NotImplementedError(f"LC layer construction for given layer setting {setting} not implemented.")