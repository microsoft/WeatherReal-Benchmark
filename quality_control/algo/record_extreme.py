import numpy as np
from .utils import get_config, intra_station_check, quality_control_statistics


CONFIG = get_config()


def _record_extreme_main(ts, upper, lower):
    """
    Flag outliers as erroneous based on the record extremes.

    Parameters:
    -----------
    ts : np.ndarray
        1D time series to be checked
    upper : float
        Upper bound of the record extreme
    lower : float
        Lower bound of the record extreme

    Returns:
    --------
    flag : np.ndarray
        1D array with the same length as ts, containing flags
    """
    flag_upper = ts > upper
    flag_lower = ts < lower
    flag = np.full(ts.shape, CONFIG["flag_normal"], dtype=np.int8)
    flag[np.logical_or(flag_upper, flag_lower)] = CONFIG["flag_error"]
    flag[np.isnan(ts)] = CONFIG["flag_missing"]
    return flag


def run(da, varname):
    flag = intra_station_check(
        da,
        qc_func=_record_extreme_main,
        kwargs=CONFIG["record"][varname],
    )
    quality_control_statistics(da, flag)
    return flag.rename("record_extreme")
