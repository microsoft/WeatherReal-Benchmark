import numpy as np
from sklearn.cluster import DBSCAN
from .utils import CONFIG, intra_station_check, quality_control_statistics


def _cluster_check(ts, reanalysis, min_samples_ratio, eps_scale, max_std_scale=None, min_num=None):
    """
    Perform a cluster-based check on time series data compared to reanalysis data.

    This function uses DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    to identify clusters in the difference between the time series and reanalysis data.
    It then flags outliers based on cluster membership.

    Parameters:
    -----------
    ts : array-like
        The time series data to be checked.
    reanalysis : array-like
        The corresponding reanalysis data for comparison.
    min_samples_ratio : float
        The ratio of minimum samples required to form a cluster in DBSCAN.
    eps_scale : float
        The scale factor for epsilon in DBSCAN, relative to the standard deviation of reanalysis.
    max_std_scale : float, optional
        If the ratio of standard deviations between ts and reanalysis is less than max_std_scale,
        the check is not performed
    min_num : int, optional
        Minimum number of valid data points required to perform the check.

    Returns:
    --------
    flag : np.ndarray
        1D array with the same length as ts, containing flags
    """
    flag = np.full(len(ts), CONFIG["flag_missing"], dtype=np.int8)
    isnan = np.isnan(ts)
    flag[~isnan] = CONFIG["flag_normal"]
    both_valid = (~isnan) & (~np.isnan(reanalysis))

    if min_num is not None and both_valid.sum() < min_num:
        return flag

    if max_std_scale is not None:
        if np.std(ts[both_valid]) / np.std(reanalysis[both_valid]) <= max_std_scale:
            return flag

    indices = np.argwhere(both_valid).flatten()
    values1 = ts[indices]
    values2 = reanalysis[indices]

    cluster = DBSCAN(min_samples=int(indices.size * min_samples_ratio), eps=np.std(reanalysis) * eps_scale)
    labels = cluster.fit((values1 - values2).reshape(-1, 1)).labels_

    # If all data points except noise are in the same cluster, just remove the noise
    if np.max(labels) <= 0:
        indices_outliers = indices[np.argwhere(labels == -1).flatten()]
        flag[indices_outliers] = CONFIG["flag_error"]
    # If there is more than one cluster, select the cluster nearest to the reanalysis
    else:
        cluster_labels = np.unique(labels)
        cluster_labels = [item for item in cluster_labels >= 0]
        best_cluster = 0
        min_median = np.inf
        for label in cluster_labels:
            indices_cluster = indices[np.argwhere(labels == label).flatten()]
            median = np.median(ts[indices_cluster] - reanalysis[indices_cluster])
            if np.abs(median) < min_median:
                best_cluster = label
                min_median = np.abs(median)
        indices_outliers = indices[np.argwhere(labels != best_cluster).flatten()]
        flag[indices_outliers] = CONFIG["flag_error"]

    return flag


def run(da, reanalysis, varname):
    """
    To ensure the accuracy of the parameters in the distributional gap method, a DBSCAN is used first,
    so that the median and MAD used for filtering are only calculated by the normal data
    """
    flag = intra_station_check(
        da,
        reanalysis,
        qc_func=_cluster_check,
        input_core_dims=[["time"], ["time"]],
        kwargs=CONFIG["cluster"][varname],
    )
    quality_control_statistics(da, flag)
    return flag.rename("cluster")
