import numpy as np
import bottleneck as bn
from .utils import get_config, intra_station_check, quality_control_statistics


CONFIG = get_config()


def _persistence_main(ts, unset_flag, min_num, max_window, min_var, error_length, exclude_value=None):
    """
    Perform persistence check on a time series.

    This function identifies periods of low variability in the time series and flags them as suspect or error.

    Parameters:
    -----------
    ts (array-like): The input time series.
    unset_flag (array-like): Pre-existing flags for the time series.
    min_num (int): Minimum number of valid values required in a window.
    max_window (int): Maximum size of the moving window.
    min_var (float): Minimum allowed standard deviation in a window.
    error_length (int): Minimum length of a period to be flagged as error.
    exclude_value (float or list, optional): Value(s) to exclude from the analysis.

    Algorithm:
    ----------
    1. Scan the time series with moving windows of decreasing size (max_window to min_num).
    2. Flag windows with standard deviation < min_var as suspect.
    3. Merge overlapping suspect windows.
    4. Flag merged windows as error if their length >= error_length.
    5. Flag remaining suspect windows based on their length and pre-existing flags.

    Returns:
    --------
    numpy.ndarray: 1D array of flags with the same length as the input time series.
    """
    flag = np.full(ts.size, CONFIG["flag_missing"], dtype=np.int8)
    suspect_windows = []

    # Bottleneck for >100x faster moving window implementations but it only works well with float64
    ts = ts.astype(np.float64)
    if exclude_value is not None:
        if isinstance(exclude_value, list):
            for value in exclude_value:
                ts[ts == value] = np.nan
        else:
            ts[ts == exclude_value] = np.nan

    for window_size in range(max_window, min_num-1, -1):
        # min_count will ensure that the std is calculated only when there are enough valid values
        std = bn.move_std(ts, window=window_size, min_count=min_num)[window_size-1:]
        valid_indices = np.argwhere(~np.isnan(std)).flatten()
        for shift in range(window_size):
            # Values checked in at least one window are considered as valid
            flag[valid_indices + shift] = CONFIG["flag_normal"]
        error_index = np.argwhere(std < min_var).flatten()
        if error_index.size == 0:
            continue
        suspect_windows.extend([(i, i+window_size) for i in error_index])

    isvalid = ~np.isnan(ts)
    if len(suspect_windows) == 0:
        flag[~isvalid] = CONFIG["flag_missing"]
        return flag

    # trim the NaNs at both ends of each window
    for idx, (start, end) in enumerate(suspect_windows):
        start = start + np.argmax(isvalid[start:end])
        end = end - np.argmax(isvalid[start:end][::-1])
        suspect_windows[idx] = (start, end)

    # Combine the overlapping windows
    suspect_windows.sort(key=lambda x: x[0])
    suspect_windows_merged = [suspect_windows[0]]
    for current in suspect_windows[1:]:
        last = suspect_windows_merged[-1]
        if current[0] < last[1]:
            suspect_windows_merged[-1] = (last[0], max(last[1], current[1]))
        else:
            suspect_windows_merged.append(current)

    for start, end in suspect_windows_merged:
        if end - start >= error_length:
            flag[start:end] = CONFIG["flag_error"]
        else:
            # Set error flag only when more than 5% values are flagged by other methods
            num_suspend = (unset_flag[start:end] == CONFIG["flag_suspect"]).sum()
            num_error = (unset_flag[start:end] == CONFIG["flag_error"]).sum()
            num_valid = isvalid[start:end].sum()
            if num_suspend + num_error > num_valid * 0.05:
                flag[start:end] = CONFIG["flag_error"]
            else:
                flag[start:end] = CONFIG["flag_suspect"]

    flag[~isvalid] = CONFIG["flag_missing"]
    return flag


def run(da, unset_flag, varname):
    flag = intra_station_check(
        da,
        unset_flag,
        qc_func=_persistence_main,
        input_core_dims=[["time"], ["time"]],
        kwargs=CONFIG["persistence"][varname],
    )
    quality_control_statistics(da, flag)
    return flag.rename("persistence")
