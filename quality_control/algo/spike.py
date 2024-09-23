import numpy as np
import bottleneck as bn
from .utils import CONFIG, intra_station_check, quality_control_statistics


def _spike_check_forward(ts, unset_flag, max_change):
    """
    Perform a forward spike check on a time series.

    A spike is defined as:
    1. An abrupt change in value
    2. Followed by an abrupt change back
    3. With no abrupt changes during the spike

    This function flags values as suspect at both ends of abrupt changes and all values within a spike.
    Due to the high sensitivity of spike detection, if all values in a spike are at least suspect
    in unset_flag, they will be flagged as errors.

    Parameters:
    -----------
    ts : array-like
        The input time series to check for spikes.
    unset_flag : array-like
        Reference flags from other methods.
    max_change : array-like
        1D array with 3 elements, representing the maximum allowed change for 1, 2, and 3 steps.

    Returns:
    --------
    flag : numpy.ndarray
        1D array with the same length as ts, containing flags for each value.
    """
    length = len(ts)
    flag = np.full(length, CONFIG["flag_missing"], dtype=np.int8)
    isnan = np.isnan(ts).astype(int)
    indices = []
    gaps = []
    # Get all the indices of potential spikes
    for step in range(1, 4, 1):
        diff = np.abs(ts[step:] - ts[:-step])
        # Flag the values where the variation is checked
        flag[step:] = np.where(
            ~np.isnan(diff) & (flag[step:] == CONFIG["flag_missing"]),
            CONFIG["flag_normal"],
            flag[step:],
        )
        condition = diff > max_change[step - 1]
        if step > 1:
            # For step larger than 1, only conditions that the in-between values are all NaN are considered
            allnan = bn.move_sum(isnan, window=step - 1)[step - 1: -1] == step - 1
            condition = condition & allnan
        indices_step = np.argwhere(condition).flatten() + step
        # flag both sides of the abrupt change as suspect
        flag[indices_step] = CONFIG["flag_suspect"]
        flag[indices_step - step] = CONFIG["flag_suspect"]
        # Save the gap between abnormal variations to be used later
        gaps.extend(np.full(len(indices_step), step))
        indices.extend(indices_step)
    # Combine the potential spikes
    sorted_indices = np.argsort(indices)
    indices = np.array(indices)[sorted_indices]
    gaps = np.array(gaps)[sorted_indices]

    cur_idx = 0
    # Iterate all potential spikes
    for case_idx, start_idx in enumerate(indices):
        # To avoid checking on the end of the spike
        if start_idx <= cur_idx:
            continue
        # The value before the spike
        leftv = ts[start_idx - gaps[case_idx]]
        # The newest value in the spike
        lastv = ts[start_idx]
        # A threshold for detecting the end of a spike
        return_diff = np.abs((lastv - leftv) / 2)
        # The direction of the variation, positive(negative) for decreasing(increasing)
        change_sign = np.sign(leftv - lastv)
        cur_idx = start_idx + 1
        num_nan = 0
        # Start to search for the end of the spike. The spike should be shorter than 72 steps
        while cur_idx < length and cur_idx - start_idx < 72:
            if np.isnan(ts[cur_idx]):
                num_nan += 1
                cur_idx += 1
                # Consider 3 continuous NaNs as end of a spike
                if num_nan >= 3:
                    # In experimental tests, if the value changes drastically and then stop recording,
                    # this single value is considered as a spike
                    if cur_idx - start_idx <= 4:
                        break
                    # Else, it is not considered as a spike
                    else:
                        cur_idx = start_idx - 1
                        break
                continue
            else:
                num_nan = 0

            isabrupt = np.abs(ts[cur_idx] - lastv) > max_change[num_nan]
            isopposite = (lastv - ts[cur_idx]) * change_sign < 0
            isnear = np.abs(ts[cur_idx] - leftv) <= return_diff
            if not isabrupt:
                # if the value changes back slowly, it is considered as normal variation
                if isnear:
                    cur_idx = start_idx - 1
                    break
                # if there is no abrupt change, and the value is still far from the original value
                # continue searching for the end of the spike
                else:
                    lastv = ts[cur_idx]
                    cur_idx += 1
                    continue
            # if there is an abrupt change, stop searching
            else:
                # If the value goes back with an opposite abrupt change, it is considered as the end of the spike
                if isopposite and isnear:
                    break
                # Else, skip this case to avoid too complex situations
                else:
                    cur_idx = start_idx - 1
                    break
        else:
            cur_idx = start_idx - 1
        # Only flag the spike as erroneous when all the values are at least suspect
        if (unset_flag[start_idx:cur_idx] != CONFIG["flag_normal"]).all():
            flag[start_idx:cur_idx] = CONFIG["flag_error"]
        else:
            flag[start_idx:cur_idx] = CONFIG["flag_suspect"]
    flag[isnan.astype(bool)] = CONFIG["flag_missing"]
    return flag


def _bidirectional_spike_check(ts, unset_flag, max_change):
    """
    Perform bidirectional spike check on the time series so that when there is an abrupt change,
        both directions will be checked
    The combined result is the higher flag of the two directions
    """
    flag_forward = _spike_check_forward(ts, unset_flag, max_change)
    flag_backward = _spike_check_forward(ts[::-1], unset_flag[::-1], max_change)[::-1]
    flag = np.full(len(ts), CONFIG["flag_missing"], dtype=np.int8)
    for flag_type in ["normal", "suspect", "error"]:
        for direction in [flag_forward, flag_backward]:
            flag[direction == CONFIG[f"flag_{flag_type}"]] = CONFIG[f"flag_{flag_type}"]
    return flag


def run(da, unset_flag, varname):
    flag = intra_station_check(
        da,
        unset_flag,
        qc_func=_bidirectional_spike_check,
        input_core_dims=[["time"], ["time"]],
        kwargs=CONFIG["spike"][varname],
    )
    quality_control_statistics(da, flag)
    return flag.rename("spike")
