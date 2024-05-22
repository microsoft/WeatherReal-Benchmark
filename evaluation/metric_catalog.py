from functools import partial

import numpy as np
import xarray as xr


def _get_mean_dims(data, group_dim):
    """
    Get the dimensions over which to compute mean.
    """
    return [dim for dim in data.dims if dim != group_dim]


def get_metric_func(settings):
    """
    Find the appropriate metric function, specified by 'settings', and
    customize it according to the rest of the 'settings'.

    All metric functions expect two arguments:
        a dataset with 3 columns, 'obs', 'fc' and 'delta',
            and with 3 dimensions: 'station', 'issue_time' 'lead_time'
        a group dimension, which is one of the dimensions of the dataset.

    They return a dataset with metric values averaged over all dims
    except the group dim.
    """
    name = settings['method']
    if name == 'rmse':
        return rmse
    elif name == 'count':
        return count
    elif name == 'mean_error':
        return mean_error
    elif name == 'mae':
        return mae
    elif name == 'sde':
        return sde
    elif name == 'min_record_error':
        return min_record_error
    elif name == 'max_record_error':
        return max_record_error
    elif name == 'step_function':
        th = settings['threshold']
        invert = settings.get('invert', False)
        return partial(step_function, threshold=th, invert=invert)
    elif name == 'accuracy':
        th = settings['threshold']
        return partial(accuracy, threshold=th)
    elif name == 'error':
        th = settings['threshold']
        return partial(error, threshold=th)
    elif name == 'f1_thresholded':
        th = settings.get('threshold', None)
        return partial(f1_thresholded, threshold=th)
    elif name == 'f1_class_averaged':
        return f1_class_averaged
    elif name == 'threat_score':
        th = settings.get('threshold', None)
        equitable = settings.get('equitable', False)
        return partial(ts_thresholded, threshold=th, equitable=equitable)
    elif name == 'reliability':
        b = settings.get('bins', None)
        return partial(reliability, bins=b)
    elif name == 'brier':
        return mse
    elif name == 'pod':
        th = settings.get('threshold')
        return partial(pod, threshold=th)
    elif name == 'far':
        th = settings.get('threshold')
        return partial(far, threshold=th)
    elif name == 'csi':
        th = settings.get('threshold')
        return partial(csi, threshold=th)
    else:
        raise ValueError(f'Unknown metric: {name}')


def rmse(data: xr.Dataset, group_dim: str) -> xr.DataArray:
    src = pow(data['delta'], 2)
    return np.sqrt(src.mean(_get_mean_dims(src, group_dim))).rename('metric')


def mse(data: xr.Dataset, group_dim: str) -> xr.DataArray:
    src = pow(data['delta'], 2)
    return src.mean(_get_mean_dims(src, group_dim)).rename('metric')


def accuracy(data: xr.Dataset, group_dim: str, threshold: float) -> xr.DataArray:
    src = abs(data['delta']) < threshold
    cnt = count(data, group_dim)
    # Weight the average to ignore missing data
    r = src.sum(_get_mean_dims(src, group_dim))
    r = r.rename('metric')
    r /= cnt
    return r


def error(data: xr.Dataset, group_dim: str, threshold: float) -> xr.DataArray:
    src = abs(data['delta']) > threshold
    cnt = count(data, group_dim)
    # Weight the average to ignore missing data
    r = src.sum(_get_mean_dims(src, group_dim))
    r = r.rename('metric')
    r /= cnt
    return r


def mean_error(data: xr.Dataset, group_dim: str) -> xr.DataArray:
    src = data['delta']
    me = src.mean(_get_mean_dims(src, group_dim))
    return me.rename('metric')


def mae(data: xr.Dataset, group_dim: str) -> xr.DataArray:
    src = abs(data['delta'])
    mae = src.mean(_get_mean_dims(src, group_dim))
    return mae.rename('metric')


def sde(data: xr.Dataset, group_dim: str) -> xr.DataArray:
    src = data['delta']
    me_all = src.mean()
    src = pow((src-me_all), 2)
    sde = src.mean(_get_mean_dims(src, group_dim))
    return np.sqrt(sde).rename('metric')


def min_record_error(data: xr.Dataset, group_dim: str) -> xr.DataArray:
    src = data['delta']
    mre = src.min(_get_mean_dims(src, group_dim))
    return mre.rename('metric')


def max_record_error(data: xr.Dataset, group_dim: str) -> xr.DataArray:
    src = data['delta']
    mre = src.max(_get_mean_dims(src, group_dim))
    return mre.rename('metric')


def step_function(data: xr.Dataset, group_dim: str, threshold: float, invert: bool) -> xr.DataArray:
    src = abs(data['delta']) < threshold
    cnt = count(data)

    # Weight the average to ignore missing data
    r = src.sum(_get_mean_dims(src, group_dim))
    r = r.rename('metric')
    r /= cnt
    if invert:
        r = 1.0 - r
    return r


def count(data: xr.Dataset, group_dim: str) -> xr.DataArray:
    src = ~data['delta'].isnull()
    cnt = src.sum(_get_mean_dims(src, group_dim))
    return cnt.rename('metric')


def _true_positives_and_false_negatives(data, labels):
    """
    Compute the true positives and false negatives for a given set of labels. Filter keeps only where truth is True and
    neither prediction nor truth is NaN.
    """
    r_list = []
    for n, label in enumerate(labels):
        recall_filter = xr.where(np.logical_and(~np.isnan(data['fc']), data['obs'] == label), 1, np.nan)
        r_list.append((data['fc'] == label) * recall_filter)
    return xr.concat(r_list, dim='category').rename('metric')


def _true_positives_and_false_positives(data, labels):
    """
    Compute the true positives and false positives for a given set of labels. Filter keeps only where prediction is
    True and neither prediction nor truth is NaN.
    """
    p_list = []
    for n, label in enumerate(labels):
        precision_filter = xr.where(np.logical_and(~np.isnan(data['obs']), data['fc'] == label), 1, np.nan)
        p_list.append((data['obs'] == label) * precision_filter)
    return xr.concat(p_list, dim='category').rename('metric')


def _geometric_mean(a, b):
    return 2 * (a * b) / (a + b)


def _count_binary_matches(data):
    data['tp'] = (data['obs'] == 1) & (data['fc'] == 1)
    data['fp'] = (data['obs'] == 0) & (data['fc'] == 1)
    data['tn'] = (data['obs'] == 0) & (data['fc'] == 0)
    data['fn'] = (data['obs'] == 1) & (data['fc'] == 0)
    return data


def _threshold_digitize(data, threshold):
    new_data = xr.Dataset()
    new_data['fc'] = xr.apply_ufunc(np.digitize, data['fc'], [threshold], dask='allowed')
    new_data['obs'] = xr.apply_ufunc(np.digitize, data['obs'], [threshold], dask='allowed')

    # Re-assign NaN where appropriate, since they are converted to 1 by digitize
    new_data['fc'] = xr.where(np.isnan(data['fc']), np.nan, new_data['fc'])
    new_data['obs'] = xr.where(np.isnan(data['obs']), np.nan, new_data['obs'])

    return new_data


def f1_thresholded(data: xr.Dataset, group_dim: str, threshold: float) -> xr.DataArray:
    """
    Compute the F1 score for a True/False classification based on values meeting or exceeding a defined threshold.
    :param data: Dataset
    :param group_dim: str: group dimension
    :param threshold: float threshold value
    :return: DataArray of scores
    """
    new_data = _threshold_digitize(data, threshold)

    # Precision/recall computation
    r = _true_positives_and_false_negatives(new_data, [1])
    p = _true_positives_and_false_positives(new_data, [1])

    # Compute and return F1
    f1 = _geometric_mean(
        p.mean(_get_mean_dims(p, group_dim)),
        r.mean(_get_mean_dims(r, group_dim))
    ).rename('metric')

    return f1.mean('category')  # mean over the single category


def ts_thresholded(data: xr.Dataset, group_dim: str, threshold: float, equitable: bool) -> xr.DataArray:
    """
    Compute the threat score for a True/False classification based on values meeting or exceeding a defined threshold.
    :param data: Dataset
    :param group_dim: str: group dimension
    :param threshold: float threshold value
    :param equitable: bool: use ETS formulation
    :return: DataArray of scores
    """
    def _count_equitable_random_chance(ds):
        n = ds['tp'] + ds['fp'] + ds['fn'] + ds['tn']
        correction = (ds['tp'] + ds['fp']) * (ds['tp'] + ds['fn']) / n
        return correction

    new_data = _threshold_digitize(data, threshold)
    new_data = _count_binary_matches(new_data)

    new_data = new_data.sum(_get_mean_dims(new_data, group_dim))
    ar = _count_equitable_random_chance(new_data) if equitable else 0
    ts = (new_data['tp'] - ar) / (new_data['tp'] + new_data['fp'] + new_data['fn'] - ar)
    return ts.rename('metric')


def f1_class_averaged(data: xr.Dataset, group_dim: str) -> xr.DataArray:
    """
    Compute the F1 score as one-vs-rest for each unique category in the data. Averages the scores for each class
    equally.
    :param data: Dataset
    :param group_dim: str: group dimension
    :return: DataArray of scores
    """
    # F1 score must be aggregated in at least one dimension
    labels = np.unique(data['obs'].values[~np.isnan(data['obs'].values)])

    # Per-label metrics
    r = _true_positives_and_false_negatives(data, labels)
    p = _true_positives_and_false_positives(data, labels)

    # Compute and return F1
    f1 = _geometric_mean(
        p.mean(_get_mean_dims(p, group_dim)),
        r.mean(_get_mean_dims(r, group_dim))
    ).rename('metric')

    return f1.mean('category')


def reliability(data: xr.Dataset, group_dim: str, bins: list) -> xr.DataArray:
    relative_freq = [0 for i in range(len(bins))]
    mean_predicted_value = [0 for i in range(len(bins))]

    truth = data['obs']
    # replace nan with 0 in truth
    truth = truth.where(truth == 1, 0)
    prediction = data['fc']

    for b in range(len(bins)):
        bin = bins[b]

        # replace predicted prob with 0 / 1 for given range in bin
        pred = prediction.where((prediction <= bin[1]) & (bin[0] < prediction), 0)
        # sum up to calculate mean later
        pred_sum = float(np.sum(pred))
        pred = pred.where(pred == 0, 1)

        # how many days have prediction fall in the given bin
        n_prediction_in_bin = np.count_nonzero(pred)

        # how many days rain in prediction set
        correct_pred = np.logical_and(pred, truth)
        n_corr_pred = np.count_nonzero(correct_pred)

        if n_prediction_in_bin != 0:
            mean_predicted_value[b] = 100 * pred_sum / n_prediction_in_bin
            relative_freq[b] = 100 * n_corr_pred / n_prediction_in_bin
        else:
            mean_predicted_value[b] = 0
            relative_freq[b] = 0

    res = xr.Dataset({'metric': ([group_dim], relative_freq)}, coords={group_dim: mean_predicted_value})
    res = res.isel({group_dim: ~res.get_index(group_dim).duplicated()})

    return res['metric']


def csi(data: xr.Dataset, group_dim: str, threshold: float) -> xr.DataArray:
    return ts_thresholded(data, group_dim, threshold, equitable=False)


def pod(data: xr.Dataset, group_dim: str, threshold: float) -> xr.DataArray:
    new_data = _threshold_digitize(data, threshold)
    r = _true_positives_and_false_negatives(new_data, [1])
    return r.mean(_get_mean_dims(r, group_dim)).rename('metric')


def far(data: xr.Dataset, group_dim: str, threshold: float) -> xr.DataArray:
    new_data = _threshold_digitize(data, threshold)
    r = _true_positives_and_false_positives(new_data, [1])
    return 1 - r.mean(_get_mean_dims(r, group_dim)).rename('metric')
