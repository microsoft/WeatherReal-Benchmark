import numpy as np
import pandas as pd
import xarray as xr
import bottleneck as bn
from suntime import Sun, SunTimeException
from .utils import CONFIG, intra_station_check


def _diurnal_cycle_check_daily(ts, lat, lon, date, max_bias):
    """
    Fit the daily temperature time series with a sine curve
    The amplitude is estimated by the daily range of the time series while the phase is estimated by the sunrise time
    If the time series is well fitted by the sine curve, it is considered as normal
    Returns:
    --------
        flag: int, a single flag indicating the result of the daily series
    """
    if ts.size != 24:
        raise ValueError("The time series should have 24 values")
    # Only check stations between 60S and 60N with significant diurnal cycles
    if abs(lat) > 60:
        return CONFIG["flag_missing"]
    # Only check samples with at least 2 valid data points in each quartile of the day
    if (np.isfinite(ts.reshape(-1, 6)).sum(axis=1) < 2).any():
        return CONFIG["flag_missing"]

    maxv = bn.nanmax(ts)
    minv = bn.nanmin(ts)
    amplitude = (maxv - minv) / 2

    # Only check samples with significant amplitude
    if amplitude < 5:
        return CONFIG["flag_missing"]

    timestep = pd.Timestamp(date)
    try:
        sunrise = Sun(float(lat), float(lon)).get_sunrise_time(timestep)
    except SunTimeException:
        return CONFIG["flag_missing"]

    # Normalize the time series by the max and min values
    normed = (ts - maxv + amplitude) / amplitude

    # Assume the diurnal cycle is a sine curve and the valley is 1H before the sunrise time
    shift = (sunrise.hour + sunrise.minute / 60) / 24 * 2 * np.pi - 20 / 12 * np.pi - timestep.hour / 12 * np.pi
    # A tolerance of 3 hours is allowed
    tolerance_values = np.arange(-np.pi / 4, np.pi / 4 + 1e-5, np.pi / 12)
    for tolerance in tolerance_values:
        # Try to find a best fitted phase of the sine curve
        sine_curve = np.sin((2 * np.pi / 24) * np.arange(24) - shift - tolerance)
        if bn.nanmax(np.abs(normed - sine_curve)) < max_bias:
            return CONFIG["flag_normal"]

    return CONFIG["flag_suspect"]


def _diurnal_cycle_check(ts, flagged, lat, lon, dates, max_bias):
    """
    Check the diurnal cycle of the temperature time series only for short period flagged by `flagged`
    Parameters:
    -----------
        ts: 1D np.array, the daily temperature time series
        flagged: 1D np.array, the flag array from other algorithms
        lat: float, the latitude of the station
        lon: float, the longitude of the station
        date: 1D np.array of numpy.datetime64, the date of the time series
        max_bias: float, the maximum bias allowed for the sine curve fitting (suggested: 0.5-1)
    Returns:
    --------
        new_flag: 1D np.array, only the checked days are flagged as either normal or suspect
    """
    new_flag = np.full_like(flagged, CONFIG["flag_missing"])
    length = len(flagged)
    error_flags = np.argwhere((flagged == CONFIG["flag_error"]) | (flagged == CONFIG["flag_suspect"])).flatten()
    end_idx = 0
    for idx, start_idx in enumerate(error_flags):
        if start_idx <= end_idx:
            continue
        # Combine these short erroneous/suspect periods into a longer one
        end_idx = start_idx
        for next_idx in error_flags[idx+1:]:
            if (
                (flagged[idx+1: next_idx] == CONFIG["flag_normal"]).any() or
                (flagged[idx+1: next_idx] == CONFIG["flag_suspect"]).any()
            ):
                break
            else:
                end_idx = next_idx
        period_length = end_idx - start_idx + 1
        if period_length > 12:
            continue
        # Select the daily series centered at the short erroneous/suspect period
        if length % 2 == 1:
            num_left, num_right = (24-period_length) // 2, (24-period_length) // 2 + 1
        else:
            num_left, num_right = (24-period_length) // 2, (24-period_length) // 2
        if (flagged[start_idx-num_left: start_idx] != CONFIG["flag_normal"]).all():
            continue
        if (flagged[end_idx+1: end_idx+1+num_right] != CONFIG["flag_normal"]).all():
            continue

        daily_start_idx = start_idx - num_left
        if daily_start_idx < 0:
            daily_start_idx = 0
        elif daily_start_idx > length - 24:
            daily_start_idx = length - 24
        daily_ts = ts[daily_start_idx: daily_start_idx + 24]
        daily_flag = _diurnal_cycle_check_daily(daily_ts, lat, lon, dates[daily_start_idx], max_bias=max_bias)
        new_flag[start_idx: end_idx+1] = daily_flag

    new_flag[np.isnan(ts)] = CONFIG["flag_missing"]
    return new_flag


def adjust_by_diurnal(target_flag, diurnal_flag):
    """
    Diurnal cycle check can be used for refine flags from other algorithms
    """
    normal_diurnal = diurnal_flag == CONFIG["flag_normal"]
    error_target = target_flag == CONFIG["flag_error"]
    new_flag = target_flag.copy()
    new_flag = xr.where(normal_diurnal & error_target, CONFIG["flag_suspect"], new_flag)
    return new_flag


def run(da, checked_flag):
    """
    Perform a diurnal cycle check on Temperature time series
    Returns:
    --------
    The adjusted flags
    """
    flag = intra_station_check(
        da,
        checked_flag,
        da["lat"],
        da["lon"],
        da["time"],
        qc_func=_diurnal_cycle_check,
        input_core_dims=[["time"], ["time"], [], [], ["time"]],
        kwargs=CONFIG["diurnal"]["t"],
    )
    new_flag = adjust_by_diurnal(checked_flag, flag)
    return new_flag.rename("diurnal_cycle")
