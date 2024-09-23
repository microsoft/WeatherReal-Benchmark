import numpy as np
import bottleneck as bn
import xarray as xr
from .utils import CONFIG, intra_station_check


def _suspect_upgrade(ts_flag):
    """
    Upgrade suspect flag period if they are surrounded by error flags
    Returns:
    --------
    new_flag: np.array, the upgraded flag array with only part of suspect flag upgraded to erroneous
    """
    new_flag = ts_flag.copy()

    # Get the start and end indices of each continous suspect period
    is_suspect = (ts_flag == CONFIG["flag_suspect"]) | (ts_flag == CONFIG["flag_missing"])
    is_suspect = np.insert(is_suspect, 0, False)
    is_suspect = np.append(is_suspect, False)
    start = np.where(is_suspect & ~np.roll(is_suspect, 1))[0] - 1
    end = np.where(is_suspect & ~np.roll(is_suspect, -1))[0] - 1

    # Filter out the suspect period in case that pure missing flags are included
    periods = [item for item in list(zip(start, end)) if (ts_flag[item[0]: item[1]+1] == CONFIG["flag_suspect"]).any()]

    length = len(ts_flag)
    for start_idx, end_idx in periods:
        if (start_idx == 0) or (end_idx == length - 1):
            continue
        is_error_left = ts_flag[start_idx-1] == CONFIG["flag_error"]
        is_error_right = ts_flag[end_idx+1] == CONFIG["flag_error"]
        if is_error_left and is_error_right:
            new_flag[start_idx:end_idx+1] = CONFIG["flag_error"]

    new_flag[ts_flag == CONFIG["flag_missing"]] = CONFIG["flag_missing"]
    return new_flag


def _upgrade_flags_window(flags, da):
    """
    Upgrade suspect flags based on the proportion of flagged points among valid data points in a sliding window
    """
    window_size = 720  # One month sliding window
    threshold = 0.5  # More than half of the data points

    # Convert to numpy arrays for faster computation
    flags_array = flags.values
    data_array = da.values

    mask = (flags_array == CONFIG["flag_suspect"]) | (flags_array == CONFIG["flag_error"])
    valid = ~np.isnan(data_array)

    # Calculate rolling sums
    rolling_sum = bn.move_sum(mask.astype(np.float64), window=window_size, min_count=1, axis=1)
    rolling_sum_valid = bn.move_sum(valid.astype(np.float64), window=window_size, min_count=1, axis=1)
    rolling_sum_valid = np.where(rolling_sum_valid == 0, np.nan, rolling_sum_valid)

    # Calculate proportion of flagged points among valid data points
    proportion_flagged = rolling_sum / rolling_sum_valid

    # Create a padded array for centered calculation
    pad_width = window_size // 2
    exceed_threshold = np.pad(proportion_flagged > threshold, ((0, 0), (pad_width, pad_width)), mode='edge')

    # Use move_max on the padded array
    expanded_mask = bn.move_max(exceed_threshold.astype(np.float64), window=window_size, min_count=1, axis=1)

    # Remove padding
    expanded_mask = expanded_mask[:, pad_width:-pad_width]

    # Upgrade suspect flags to error flags where the expanded mask is True
    upgraded_flags = np.where(
        (flags_array == CONFIG["flag_suspect"]) & (expanded_mask == 1),
        CONFIG["flag_error"],
        flags_array
    )

    # Convert back to xarray DataArray
    return xr.DataArray(upgraded_flags, coords=flags.coords, dims=flags.dims)


def _upgrade_flags_all(flags, da):
    """
    If more than half of the data points at a station are flagged as erroneous,
    all the data points at this station are flagged as erroneous
    """
    threshold = 0.5  # More than half of the data points

    mask = flags == CONFIG["flag_error"]
    valid = da.notnull()
    proportion_flagged = mask.sum(dim="time") / valid.sum(dim="time")

    upgraded_flags = flags.where(proportion_flagged < threshold, CONFIG["flag_error"])
    upgraded_flags = flags.where(valid, CONFIG["flag_missing"])

    return upgraded_flags


def run(flag, da):
    flag = intra_station_check(flag, qc_func=_suspect_upgrade)
    flag = _upgrade_flags_window(flag, da)
    flag = _upgrade_flags_all(flag, da)
    return flag.rename("suspect_upgrade")
