import numpy as np
import xarray as xr
from .utils import CONFIG


def supersaturation(t, td):
    flag = xr.where(t < td, CONFIG["flag_error"], CONFIG["flag_normal"])
    isnan = np.isnan(t) | np.isnan(td)
    flag = flag.where(~isnan, CONFIG["flag_missing"])
    return flag.astype(np.int8)


def wind_consistency(ws, wd):
    zero_ws = (ws == 0)
    zero_wd = (wd == 0)
    inconsistent = (zero_ws != zero_wd)
    flag = xr.where(inconsistent, CONFIG["flag_error"], CONFIG["flag_normal"])
    isnan = np.isnan(ws) | np.isnan(wd)
    flag = flag.where(~isnan, CONFIG["flag_missing"])
    return flag.astype(np.int8)


def ra_consistency(ds):
    """
    Precipitation in different period length is cross validated
    If there is a conflict (ra1 at 18:00 is 5mm, while ra3 at 19:00 is 0mm),
    both of them are flagged
    Parameters:
    -----------
    ds: xarray dataset with precipitation data including ra1, ra3, ra6, ra12 and ra24
    Returns:
    --------
    flag : numpy.ndarray
        1D array with the same length as ts, containing flags for each value.
    """
    flag = xr.full_like(ds, 0, dtype=np.int8)
    periods = [3, 6, 12, 24]
    for period in periods:
        da_longer = ds[f"ra{period}"]
        for shift in range(1, period):
            # Shift the precipitation in the longer period to align with the shorter period
            shifted = da_longer.roll(time=-shift)
            shifted[:, -shift:] = np.nan
            for target_period in [1, 3, 6, 12, 24]:
                # Check if the two periods are overlapping
                if target_period >= period or target_period + shift > period:
                    continue
                # If the precipitation in the shorter period is larger than the longer period, flag both
                flag_indices = np.where(ds[f"ra{target_period}"].values - shifted.values > 0.101)
                if len(flag_indices[0]) == 0:
                    continue
                flag[f"ra{target_period}"].values[flag_indices] = 1
                flag[f"ra{period}"].values[(flag_indices[0], flag_indices[1]+shift)] = 1
    flag = flag.where(ds.notnull(), -1)
    return flag
