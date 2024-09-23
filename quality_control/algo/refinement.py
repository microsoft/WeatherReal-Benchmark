import numpy as np
from .utils import CONFIG, intra_station_check


def _is_monotonic(ts, start_idx, end_idx):
    # Search the left and right nearest non-error values
    for left_idx in range(start_idx-1, max(0, start_idx-4), -1):
        # Note that here the found value is bound to be non-error
        if np.isfinite(ts[left_idx]):
            break
    else:
        left_idx = None
    for right_idx in range(end_idx, min(end_idx+3, len(ts))):
        if np.isfinite(ts[right_idx]):
            break
    else:
        right_idx = None
    # If these values are monotonic, downgrading the error flag to suspect
    if left_idx is not None and right_idx is not None:
        ts_period = ts[left_idx: right_idx + 1]
        ts_period = ts_period[~np.isnan(ts_period)]
        diff = np.diff(ts_period)
        if (diff > 0).all() or (diff < 0).all():
            return True
    return False


def _is_ridge_or_trough(ts, start_idx, end_idx):
    # Search the left and right nearest 3 non-error values
    left_values = ts[max(0, start_idx-12): max(0, start_idx)]
    left_values = left_values[np.isfinite(left_values)]
    if len(left_values) < 4:
        return False
    left_values = left_values[-4:]
    right_values = ts[min(len(ts), end_idx): min(len(ts), end_idx+12)]
    right_values = right_values[np.isfinite(right_values)]
    if len(right_values) < 4:
        return False
    right_values = right_values[:4]

    ts_period = ts[start_idx: end_idx]
    ts_period = np.concatenate([left_values, ts_period[np.isfinite(ts_period)], right_values])
    min_idx = np.argmin(ts_period)
    if 3 < min_idx < len(ts_period) - 4:
        diff = np.diff(ts_period)
        # Check if it is a trough (e.g., low pressure)
        if (diff[:min_idx] < 0).all() and (diff[min_idx:] > 0).all():
            return True
    max_idx = np.argmax(ts_period)
    if 3 < max_idx < len(ts_period) - 4:
        diff = np.diff(ts_period)
        # Check if it is a ridge (e.g., temperature peak)
        if (diff[:max_idx] > 0).all() and (diff[max_idx:] < 0).all():
            return True
    return False


def _refine_flag(cur_flag, ts, check_monotonic, check_ridge_trough):
    """
    Refine the flags from other algorithms:
        check_monotonic: If True, downgrade error flags to suspect if they are
            situated in a monotonic period in-between non-error values
        check_ridge_trough: If True, downgrade error flags to suspect if they are
            situated in a ridge or trough in-between non-error values
    Returns:
    --------
    new_flag: The refined flags
    """
    new_flag = cur_flag.copy()
    if not check_monotonic and not check_ridge_trough:
        return new_flag
    length = len(cur_flag)
    error_flags = np.argwhere(cur_flag == CONFIG["flag_error"]).flatten()
    cur_idx = 0
    for idx in error_flags:
        if cur_idx > idx:
            continue
        cur_idx = idx + 1
        num_nan = 0
        num_error = 1
        # Search for the next non-error value
        while num_nan <= 2 and cur_idx < length:
            # If there are more than 3 consecutive missing values or error flags, do nothing
            if np.isnan(ts[cur_idx]):
                num_nan += 1
                cur_idx += 1
                continue
            elif cur_flag[cur_idx] == CONFIG["flag_error"]:
                num_nan = 0
                num_error += 1
                cur_idx += 1
                continue
            # If a non-error value is found, check if it is monotonic
            else:
                if num_error > 3:
                    break
                if check_monotonic and _is_monotonic(ts, idx, cur_idx):
                    new_flag[idx:cur_idx] = CONFIG["flag_suspect"]
                if check_ridge_trough and _is_ridge_or_trough(ts, idx, cur_idx):
                    new_flag[idx:cur_idx] = CONFIG["flag_suspect"]
                break
    new_flag[np.isnan(ts)] = CONFIG["flag_missing"]
    return new_flag


def run(da, flag, varname):
    flag = intra_station_check(
        flag,
        da,
        qc_func=_refine_flag,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[["time"]],
        kwargs=dict(
            check_monotonic=CONFIG["refinement"][varname]["check_monotonic"],
            check_ridge_trough=CONFIG["refinement"][varname]["check_ridge_trough"]
        ),
    )
    return flag.rename("refinement")
