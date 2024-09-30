import numpy as np
from scipy import stats
from .utils import get_config


CONFIG = get_config()


def _get_mean_and_mad(values):
    mean = np.median(values)
    mad = stats.median_abs_deviation(values)
    return mean, mad


def _time_series_comparison(
    ts1,
    ts2,
    shift_step,
    gap_scale,
    default_mad,
    suspect_std_scale,
    min_mad=0.1,
    min_num=None,
    mask=None,
):
    """
    Perform time series comparison between two datasets and flag potential errors.

    This function compares two time series (ts1 and ts2) and identifies suspect and erroneous values
    based on their differences. It uses a robust statistical approach to handle outliers and
    can accommodate temporal shifts between the series.

    Parameters:
    -----------
    ts1 : array-like
        The primary time series to be checked.
    ts2 : array-like
        The reference time series for comparison.
    shift_step : int
        The maximum number of time steps to shift ts2 when comparing with ts1.
    gap_scale : float
        The scale factor applied to the median absolute deviation (MAD) to determine the gap threshold.
    default_mad : float
        The default MAD value to use when the calculated MAD is too small or when there are insufficient data points.
    suspect_std_scale : float
        The number of standard deviations from the mean to set the initial suspect threshold.
    min_mad : float, optional
        The minimum MAD value to calculate standard deviation, default is 0.1.
    min_num : int, optional
        The minimum number of valid data points required for robust statistics calculation.
        If the number of valid points is less than this, default values are used.
    mask : array-like, optional
        Boolean mask to select a subset of ts1 for calculating bounds. If None, all values are used.
        True for normal values, False for outliers

    Returns:
    --------
    flag : numpy.ndarray
        1D array with the same length as ts, containing flags for each value.
    """
    diff = ts1 - ts2
    values = diff[~np.isnan(diff)]

    # Apply mask to diff if provided
    if mask is not None:
        masked_diff = diff[mask]
        values = masked_diff[~np.isnan(masked_diff)]
    else:
        values = diff[~np.isnan(diff)]

    if values.size == 0:
        return np.full(diff.size, CONFIG["flag_missing"], dtype=np.int8)

    if min_num is not None and values.size < min_num:
        fixed_mean = 0
        mad = default_mad
    else:
        # An estimate of the Gaussian distribution of the data which is calculated by the median and MAD
        # so that it is robust to outliers
        fixed_mean, mad = _get_mean_and_mad(values)
        mad = max(min_mad, mad)

    # Get the suspect threshold by the distance to the mean in the unit of standard deviation
    # Reference: y = 0.1, scale = 1.67; y = 0.05, scale = 2.04; y = 0.01, scale = 2.72
    # If the standard deviation estimated by MAD is too small, a default value is used
    fixed_std = max(default_mad, mad) * 1.4826
    init_upper_bound = fixed_mean + fixed_std * suspect_std_scale
    # For observations that the actual precision is integer, the upper and lower bounds are rounded up
    is_integer = np.nanmax(ts1 % 1) < 0.1 or np.nanmax(ts2 % 1) < 0.1
    if is_integer:
        init_upper_bound = np.ceil(init_upper_bound)
    # Set the erroneous threshold by find a gap larger than a multiple of the MAD
    larger_values = np.insert(np.sort(values[values > init_upper_bound]), 0, init_upper_bound)
    # Try to get the index of first value where the gap larger than min_gap
    gap = mad * gap_scale
    large_gap = np.diff(larger_values) > gap
    # If a gap is not found, no erroneous threshold is set
    upper_bound = larger_values[np.argmax(large_gap)] if large_gap.any() else np.max(values)
    if is_integer:
        upper_bound = np.ceil(upper_bound)

    init_lower_bound = fixed_mean - fixed_std * suspect_std_scale
    if is_integer:
        init_lower_bound = np.floor(init_lower_bound)
    smaller_values = np.insert(np.sort(values[values < init_lower_bound])[::-1], 0, init_lower_bound)
    small_gap = np.diff(smaller_values) < -gap
    lower_bound = smaller_values[np.argmax(small_gap)] if small_gap.any() else np.min(values)
    if is_integer:
        lower_bound = np.floor(lower_bound)

    min_diff = diff
    # If shift_step > 0, the values in ts1 are also compared to neighboring values in ts2
    # The minimum difference is kept to be inclusive of a certain degree of temporal deviation
    for shift in np.arange(-shift_step, shift_step+1, 1):
        diff_shifted = ts1 - np.roll(ts2, shift)
        if shift == 0:
            continue
        if shift > 0:
            diff_shifted[:shift] = np.nan
        elif shift < 0:
            diff_shifted[shift:] = np.nan
        min_diff = np.where(np.abs(diff_shifted - fixed_mean) < np.abs(min_diff - fixed_mean), diff_shifted, min_diff)

    flag = np.full_like(min_diff, CONFIG["flag_normal"], dtype=np.int8)
    flag[min_diff < init_lower_bound] = CONFIG["flag_suspect"]
    flag[min_diff > init_upper_bound] = CONFIG["flag_suspect"]
    flag[min_diff < lower_bound] = CONFIG["flag_error"]
    flag[min_diff > upper_bound] = CONFIG["flag_error"]
    flag[np.isnan(min_diff)] = CONFIG["flag_missing"]
    return flag
