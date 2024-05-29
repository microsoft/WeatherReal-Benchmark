from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def configure_logging(verbose=1):
    verbose_levels = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
        3: logging.NOTSET
    }
    if verbose not in verbose_levels.keys():
        verbose = 1
    logger = logging.getLogger()
    logger.setLevel(verbose_levels[verbose])
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s] [PID=%(process)d] "
        "[%(levelname)s %(filename)s:%(lineno)d] %(message)s"))
    handler.setLevel(verbose_levels[verbose])
    logger.addHandler(handler)


@dataclass
class ForecastInfo:
    path: str
    forecast_name: str
    fc_var_name: str
    reformat_func: str
    file_type: str
    station_metadata_path: str
    interp_station_path: str
    output_directory: str
    cache_path: Optional[str] = None


@dataclass
class ForecastData:
    info: ForecastInfo
    forecast: Optional[xr.Dataset] = None
    merge_data: Optional[xr.Dataset] = None


@dataclass
class MetricData:
    info: ForecastInfo
    metric_data: Optional[xr.Dataset] = None


def get_metric_multiple_stations(files):
    data = {}
    files = files.split(',')
    for f in files:
        if os.path.exists(f):
            try:
                key = os.path.basename(f).replace(".csv", "")
                data[key] = [str(id) for id in pd.read_csv(f)['Station'].tolist()]
            except Exception:
                raise Exception(f"Error opening {f}!")
        else:
            raise Warning(f'File {f} do not exist!')
    return data


def generate_forecast_cache_path(info: ForecastInfo):
    if info.cache_path is not None:
        return info.cache_path
    file_name = Path(info.path).stem
    forecast_name = info.forecast_name
    fc_var_name = info.fc_var_name
    reformat_func = info.reformat_func
    interp_station = Path(info.interp_station_path).stem
    cache_directory = os.path.join(info.output_directory, 'cache')
    os.makedirs(cache_directory, exist_ok=True)
    cache_file_name = "##".join([file_name, forecast_name, fc_var_name, reformat_func, interp_station, 'cache'])
    info.cache_path = os.path.join(cache_directory, cache_file_name)
    return info.cache_path


def cache_reformat_forecast(forecast_ds, cache_path):
    logger.info(f"saving forecast to cache at {cache_path}")
    forecast_ds.to_zarr(cache_path, mode='w')


def load_reformat_forecast(cache_path):
    forecast_ds = xr.open_zarr(cache_path)
    return forecast_ds


def get_ideal_xticks(min_lead, max_lead, tick_count=8):
    """
    Pick the best interval for the x axis (hours) that optimizes the number of ticks
    """
    candidate_intervals = [1, 3, 6, 12, 24]
    tick_counts = []
    for interval in candidate_intervals:
        num_ticks = (max_lead - min_lead) / interval
        tick_counts.append(abs(num_ticks - tick_count))
    best_interval = candidate_intervals[tick_counts.index(min(tick_counts))]
    return np.arange(min_lead, max_lead + best_interval, best_interval)
