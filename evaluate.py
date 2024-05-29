import argparse
from functools import reduce
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from evaluation.forecast_reformat_catalog import reformat_filter_forecast
from evaluation.obs_reformat_catalog import reformat_and_filter_obs, obs_to_verification
from evaluation.metric_catalog import get_metric_func
from evaluation.utils import configure_logging, get_metric_multiple_stations, generate_forecast_cache_path, \
    cache_reformat_forecast, load_reformat_forecast, ForecastData, MetricData, get_ideal_xticks

import warnings

from evaluation.utils import ForecastInfo

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def intersect_all_forecast(forecast_list: List[ForecastData]) -> List[ForecastData]:
    """
    Generates new versions of the forecast data objects where the dataframes have been aligned between all
    forecasts in the sequence. If only one forecast is provided, the data is returned as is.
    """
    raise NotImplementedError("Forecast alignment is not yet implemented.")


def get_forecast_data(forecast_info: ForecastInfo, start_lead: int, end_lead: int,
                      convert_t: bool, cache_forecast: bool) -> ForecastData:
    """
    Open all the forecasts from the forecast_info_list and reformat them.

    Parameters
    ----------
    forecast_info: ForecastInfo
        The forecast information.
    start_lead: int
        The first lead time to include in the evaluation.
    end_lead: int
        The last lead time to include in the evaluation.
    convert_t: bool
        Convert temperature from Kelvin to Celsius.
    cache_forecast: bool
        If true, cache the reformat forecast data, next time will load from cache.

    Returns
    -------
    forecast: ForecastData
        Forecast data and the forecast information.
    """
    info = forecast_info
    cache_path = generate_forecast_cache_path(info)
    if not os.path.exists(cache_path) or not cache_forecast:
        logger.info(f"open forecast file: {info.path}")
        if info.file_type is None:
            if Path(info.path).is_dir():
                forecast = xr.open_zarr(info.path)
            else:
                forecast = xr.open_dataset(info.path, chunks={})
        elif info.file_type == 'zarr':
            forecast = xr.open_zarr(info.path)
        else:
            forecast = xr.open_dataset(info.path, chunks={})
        forecast_ds = reformat_filter_forecast(forecast, info, start_lead, end_lead, convert_t)
        if cache_forecast:
            cache_reformat_forecast(forecast_ds, cache_path)
            logger.info(f"save forecast file to cache: {cache_path}")
    else:
        forecast_ds = load_reformat_forecast(cache_path)
        logger.info(f"load forecast: {info.forecast_name}, from cache: {cache_path}")
    logger.debug(f"opened forecast dataset: {forecast_ds}")
    return ForecastData(info=info, forecast=forecast_ds)


def get_observation_data(obs_base_path: str, obs_var_name: str, station_metadata_path: str,
                         obs_file_type: str, obs_start_month: str, obs_end_month: str) -> xr.Dataset:
    """
    Open the observation file and reformat it. Required fields: station, valid_time, obs_var_name.

    Parameters
    ----------
    obs_base_path: str
        Path to the observation file.
    obs_var_name: str
        Name of the observation variable.
    station_metadata_path: str
        Path to the station metadata file.
    obs_file_type: str
        Type of the observation file.
    obs_start_month: str
        Obs start month, for multi-file netCDF data.
    obs_end_month: str
        Obs end month, for multi-file netCDF data.

    Returns
    -------
    obs: xr.Dataset
        Observation data with required fields.
    """
    if obs_start_month is not None and obs_end_month is not None:
        if obs_start_month is None or obs_end_month is None:
            raise ValueError("Both obs_start_month and obs_end_month must be provided.")
        month_list = pd.date_range(obs_start_month, obs_end_month, freq='MS')
        suffix = obs_file_type or 'nc'
        obs_path = [os.path.join(obs_base_path, month.strftime(f'%Y%m.{suffix}')) for month in month_list]
        obs_path_filter = []
        for path in obs_path:
            if not os.path.exists(path):
                logger.warning(f"expected observation path does not exist: {path}")
            else:
                obs_path_filter.append(path)
        obs = xr.open_mfdataset(obs_path_filter, chunks={})
    else:
        if obs_file_type is None:
            if Path(obs_base_path).is_dir():
                obs = xr.open_zarr(obs_base_path)
            else:
                obs = xr.open_dataset(obs_base_path, chunks={})
        elif obs_file_type == 'zarr':
            obs = xr.open_zarr(obs_base_path)
        else:
            obs = xr.open_dataset(obs_base_path, chunks={})

    obs = reformat_and_filter_obs(obs, obs_var_name, station_metadata_path)
    logger.debug(f"opened observation dataset: {obs}")
    return obs


def merge_forecast_obs(forecast: ForecastData, obs: xr.Dataset) -> ForecastData:
    """
    Merge the forecast and observation data.
    """
    new_obs = obs_to_verification(
        obs,
        steps=forecast.forecast.lead_time.values,
        max_lead=forecast.forecast.lead_time.values.max(),
        issue_times=forecast.forecast.issue_time.values
    )
    merge_data = xr.merge([forecast.forecast, new_obs], compat='override')
    merge_data['delta'] = merge_data['fc'] - merge_data['obs']
    result = ForecastData(info=forecast.info, forecast=forecast.forecast, merge_data=merge_data)
    logger.debug(f"after merge forecast and obs: {result.merge_data}")
    return result


def filter_by_region(forecast: ForecastData, region_name: str, station_list: List[str]) \
        -> ForecastData:
    """
    Apply a selection on forecast based on the region_name and station_list.
    """
    if region_name == 'all':
        return forecast
    else:
        merge_data = forecast.merge_data
        filtered_merge_data = ForecastData(
            merge_data=merge_data.sel(station=merge_data.station.values.isin(station_list)),
            info=forecast.info
        )
        return filtered_merge_data


def calculate_all_metrics(forecast_data: ForecastData, group_dim: str, metrics_params: Dict[str, Any]) \
        -> MetricData:
    """
    Calculate all the metrics together for dask graph efficiency.

    Parameters
    ----------
    forecast_data: ForecastData
        The forecast data.
    group_dim: str
        The dimension to group the metric calculation.
    metrics_params: dict
        Dictionary containing the metrics configs.

    Returns
    -------
    metric_data: MetricData
        Metric data and the forecast information.
    """
    metrics = MetricData(info=forecast_data.info, metric_data=xr.Dataset())
    for metric_name in metrics_params.keys():
        metric_func = get_metric_func(metrics_params[metric_name])
        metrics.metric_data[metric_name] = metric_func(forecast_data.merge_data, group_dim)
    metrics.metric_data = metrics.metric_data.compute()
    return metrics


def get_plot_detail(forecast_data: ForecastData, group_dim: str):
    """
    Get some added data to show on plots
    """
    merge_data = forecast_data.merge_data
    counts = {key: coord.size for key, coord in merge_data.coords.items() if key not in ['lat', 'lon']}
    counts.update({'fc': merge_data.fc.size})
    all_dims = ['valid_time', 'issue_time', 'lead_time']
    all_dims.remove(group_dim)
    dim_info = []
    for dim in all_dims:
        if dim == 'lead_time':
            vmax, vmin = merge_data[dim].max(), merge_data[dim].min()
        else:
            vmax, vmin = pd.Timestamp(merge_data[dim].values.max()).strftime("%Y-%m-%d %H:%M:%S"), \
                pd.Timestamp(merge_data[dim].values.min()).strftime("%Y-%m-%d %H:%M:%S")
        dim_info.append(f'{dim} min: {vmin}, max: {vmax}')
    data_distribution = f"dim count: {str(counts)}\n{dim_info[0]}\n{dim_info[1]}"
    return data_distribution


def plot_metric(
        example_data: ForecastData,
        metric_data_list: List[MetricData],
        group_dim: str,
        metric_name: str,
        base_plot_setting: Dict[str, Any],
        metrics_params: Dict[str, Any],
        output_dir: str,
        region_name: str,
        plot_save_format: Optional[str] = 'png'
) -> plt.Figure:
    """
    A generic, basic plot for a single metric.

    Parameters
    ----------
    example_data: ForecastData
        Example forecast data to get some extra information for the plot.
    metric_data_list: list of MetricData
        List of MetricData objects containing the metric data and the forecast information.
    group_dim: str
        The dimension to group the metric calculation.
    metric_name: str
        The name of the metric.
    base_plot_setting: dict
        Dictionary containing the base plot settings.
    metrics_params: dict
        Dictionary containing the metric method and other kwargs.
    output_dir: str
        The output directory for the plots.
    region_name: str
        The name of the region.
    plot_save_format: str, optional
        The format to save the plot in. Default is 'png'.

    Returns
    -------
    fig: plt.Figure
        The plot figure.
    """
    data_distribution = get_plot_detail(example_data, group_dim)

    fig = plt.figure(figsize=(5.5, 6.5))
    font = {'weight': 'medium', 'fontsize': 11}
    title = base_plot_setting['title']
    xlabel = base_plot_setting['xlabel']
    if 'plot_setting' in metrics_params:
        plot_setting = metrics_params['plot_setting']
        title = plot_setting.get('title', title)
        xlabel = plot_setting.get('xlabel', xlabel)
    plt.title(title)
    plt.suptitle(data_distribution, fontsize=7)
    plt.gca().set_xlabel(xlabel[group_dim], fontdict=font)
    plt.gca().set_ylabel(metric_name, fontdict=font)

    for metrics in metric_data_list:
        metric_data = metrics.metric_data
        forecast_name = metrics.info.forecast_name
        plt.plot(metric_data[group_dim], metric_data[metric_name], label=forecast_name, linewidth=1.5)

    if group_dim == 'lead_time':
        plt.gca().set_xticks(get_ideal_xticks(metric_data[group_dim].min(), metric_data[group_dim].max(), 8))
    plt.grid(linestyle=':')
    plt.legend(loc='upper center', bbox_to_anchor=(0.45, -0.14), frameon=False, ncol=3, fontsize=10)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plot_path = os.path.join(output_dir, region_name)
    os.makedirs(plot_path, exist_ok=True)
    plt.savefig(os.path.join(plot_path, f"{metric_name}.{plot_save_format}"))
    return fig


def metrics_to_csv(
        metric_data_list: List[MetricData],
        group_dim: str,
        output_dir: str,
        region_name: str
):
    """
    A generic, basic function to save the metric data to a CSV file.

    Parameters
    ----------
    metric_data_list: list of MetricData
        List of MetricData objects containing the metric data and the forecast information.
    group_dim: str
        The dimension to group the metric calculation.
    output_dir: str
        The output directory for the CSV files.
    region_name: str
        The name of the region.

    Returns
    -------
    merged_df: pd.DataFrame
        The merged DataFrame with metric data.
    """
    df_list = []
    for metrics in metric_data_list:
        df_list.append(metrics.metric_data.rename(
            {metric: f"{metrics.info.forecast_name}_{metric}" for metric in metrics.metric_data.data_vars.keys()}
        ).to_dataframe())
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=group_dim, how='inner'), df_list)

    output_path = os.path.join(output_dir, region_name)
    os.makedirs(output_path, exist_ok=True)
    merged_df.to_csv(os.path.join(output_path, "metrics.csv"))
    return merged_df


def parse_args(args: argparse.Namespace) -> Tuple[List[ForecastInfo], Any, Any, Union[
        Dict[str, List[Any]], Any], Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, bool, bool, bool]:
    forecast_info_list = []
    forecast_name_list = args.forecast_names
    forecast_var_name_list = args.forecast_var_names
    forecast_reformat_func_list = args.forecast_reformat_funcs
    station_metadata_path = args.station_metadata_path
    for index, forecast_path in enumerate(args.forecast_paths):
        forecast_info = ForecastInfo(
            path=forecast_path,
            forecast_name=forecast_name_list[index] if index < len(forecast_name_list) else f"forecast_{index}",
            fc_var_name=forecast_var_name_list[index] if index < len(forecast_var_name_list) else
            forecast_var_name_list[0],
            reformat_func=forecast_reformat_func_list[index] if index < len(forecast_reformat_func_list) else
            forecast_reformat_func_list[0],
            file_type=args.forecast_file_types[index] if index < len(args.forecast_file_types) else None,
            station_metadata_path=station_metadata_path,
            interp_station_path=station_metadata_path,
            output_directory=args.output_directory
        )
        forecast_info_list.append(forecast_info)

    metrics_settings_path = args.config_path if args.config_path is not None else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'metric_config.yml')
    with open(metrics_settings_path, 'r') as fs:
        metrics_settings = yaml.safe_load(fs)

    try:
        metrics_settings = metrics_settings[args.variable_type]
    except KeyError:
        raise ValueError(f"Unknown variable type: {args.variable_type}. Check config file {metrics_settings_path}")
    metrics_dict = metrics_settings['metrics']
    base_plot_setting = metrics_settings['base_plot_setting']
    if args.eval_region_files is not None:
        try:
            region_dict = get_metric_multiple_stations(','.join(args.eval_region_files))
        except Exception as e:
            logger.info(f"get_metric_multiple_stations failed, use default region: all {e}")
            region_dict = {}
    else:
        region_dict = {}
    region_dict['all'] = []
    group_dim = args.group_dim
    obs_var_name = args.obs_var_name
    obs_base_path = args.obs_path
    obs_start_month = args.obs_start_month
    obs_end_month = args.obs_end_month
    output_dir = args.output_directory
    start_lead = args.start_lead
    end_lead = args.end_lead

    return (forecast_info_list, metrics_dict, base_plot_setting,
            region_dict, group_dim, obs_var_name, obs_base_path, args.obs_file_type, obs_start_month,
            obs_end_month, output_dir, start_lead, end_lead, station_metadata_path,
            bool(args.convert_fcst_temperature_k_to_c), bool(args.cache_forecast), bool(args.align_forecasts))


def main(args):
    logger.info("===================== parse args =====================")
    (forecast_info_list, metrics_dict, base_plot_setting, region_dict, group_dim, obs_var_name,
     obs_base_path, obs_file_type, obs_start_month, obs_end_month, output_dir, start_lead, end_lead,
     station_metadata_path, convert_t, cache_forecast, align_forecasts) = parse_args(args)

    logger.info("===================== start get_observation_data =====================")
    obs_ds = get_observation_data(obs_base_path, obs_var_name, station_metadata_path, obs_file_type, obs_start_month,
                                  obs_end_month)

    if align_forecasts:
        # First load all forecasts, then compute and return metrics.
        logger.info("===================== start get_forecast_data =====================")
        forecast_list = [get_forecast_data(fi, start_lead, end_lead, convert_t, cache_forecast)
                         for fi in forecast_info_list]
        logger.info("===================== start intersect_all_forecast =====================")
        forecast_list = intersect_all_forecast(forecast_list)
    else:
        forecast_list = [None] * len(forecast_info_list)

    # For each forecast, compute all its metrics in every region.
    metric_data = {r: [] for r in region_dict.keys()}
    for forecast, forecast_info in zip(forecast_list, forecast_info_list):
        try:
            del merged_forecast  # noqa: F821
        except NameError:
            pass
        logger.info(f"===================== compute metrics for forecast {forecast_info.forecast_name} "
                    f"=====================")
        if forecast is None:
            logger.info("===================== get_forecast_data =====================")
            forecast = get_forecast_data(forecast_info, start_lead, end_lead, convert_t, cache_forecast)

        logger.info("===================== start merge_forecast_obs =====================")
        merged_forecast = merge_forecast_obs(forecast, obs_ds)

        for region_name in region_dict.keys():
            logger.info(f"===================== filter_by_region: {region_name} =====================")
            filtered_forecast = filter_by_region(merged_forecast, region_name, region_dict[region_name])
            if region_name != 'all':
                logger.info(f"after filter_by_region: {region_name}; "
                            f"stations: {filtered_forecast.merge_data.station.size}")

            logger.info(f"start calculate_metrics, region: {region_name}")
            metric_data[region_name].append(
                calculate_all_metrics(filtered_forecast, group_dim, metrics_dict)
            )
        forecast = None

    # Plot all metrics and save data
    for region_name in region_dict.keys():
        for metric_name in metrics_dict.keys():
            logger.info(f"===================== plot_metric: {metric_name}, region: {region_name} "
                        f"=====================")
            plot_metric(merged_forecast, metric_data[region_name], group_dim, metric_name,
                        base_plot_setting, metrics_dict[metric_name], output_dir, region_name)

            metrics_to_csv(metric_data[region_name], group_dim, output_dir, region_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Forecast evaluation script. Given a set of forecasts and a file of reference observations, "
                    "computes requested metrics as specified in the `metric_catalog.yml` file. Includes the ability "
                    "to interpret either grid-based or point-based forecasts. Grid-based forecasts are interpolated "
                    "to observation locations. Point-based forecasts are directly compared to nearest observations."
    )
    parser.add_argument(
        "--forecast-paths",
        nargs='+',
        type=str,
        required=True,
        help="List of paths containing forecasts. If a directory is provided, assumes forecast is a zarr store, "
             "and calls xarray's `open_zarr` method. "
             "Required dimensions: lead_time (or step), issue_time (or time), lat (or latitude), lon (or longitude)."
    )
    parser.add_argument(
        '--forecast-names',
        type=str,
        nargs='+',
        default=[],
        help="List of names to assign to the forecasts. If there are more forecast paths than names, fills in the "
             "remaining names with 'forecast_{index}'"
    )
    parser.add_argument(
        '--forecast-var-names',
        type=str,
        nargs='+',
        required=True,
        help="List of names (one per forecast path) of the forecast variable of interest in each file. If only one "
             "value is provided, assumes all forecast files have the same variable name. Raises an error if the "
             "number of listed values is less than the number of forecast paths."
    )
    parser.add_argument(
        '--forecast-reformat-funcs',
        type=str,
        nargs='+',
        required=True,
        help="For each forecast path, provide the name of the reformat function to apply. This function is based on "
             "the schema of the forecast file. Can be only a single value to apply to all forecasts. Options: "
             "\n  - 'grid_standard': input is a grid forecast with dimensions lead_time, issue_time, lat, lon."
             "\n  - 'point_standard': input is a point forecast with dimensions lead_time, issue_time, station."
             "\n  - 'grid_v1': custom reformat function for grid forecasts with dims time, step, latitude, longitude."
    )
    parser.add_argument(
        '--forecast-file-types',
        type=str,
        nargs='+',
        default=[],
        help="List of file types for each forecast path. Options: 'nc', 'zarr'. If not provided, or not enough "
             "entries, will assume zarr store if forecast is a directory, and otherwise will use xarray's "
             "`open_dataset` method."
    )
    parser.add_argument(
        "--obs-path",
        type=str,
        required=True,
        help="Path to the verification folder or file"
    )
    parser.add_argument(
        "--obs-file-type",
        type=str,
        default=None,
        help="Type of the observation file. Options: 'nc', 'zarr'. If not provided, will assume zarr store if this is "
             "a directory, and otherwise will use xarray's `open_dataset` method."
    )
    parser.add_argument(
        "--obs-start-month",
        type=str,
        default=None,
        help="Option to read multiple netCDF files as a single dataset. These files are named 'YYYYMM.nc'. Provide "
             "the start month in the format 'YYYY-MM'. Not needed if obs-path is a single nc/zarr store."
    )
    parser.add_argument(
        "--obs-end-month",
        type=str,
        default=None,
        help="Option to read multiple netCDF files as a single dataset. These files are named 'YYYYMM.nc'. Provide "
             "the end month in the format 'YYYY-MM'. Not needed if obs-path is a single nc/zarr store."
    )
    parser.add_argument(
        "--obs-var-name",
        type=str,
        help="Name of the variable of interest in the observation data.",
        required=True
    )
    parser.add_argument(
        "--station-metadata-path",
        type=str,
        help="Path to the station list containing metadata. Must include columns 'station', 'lat', 'lon'. "
             "If not provided, assumes the station lat/lon are coordinates in the observation file.",
        required=False
    )
    parser.add_argument(
        "--config-path",
        type=str,
        help="Path to custom config yml file containing metric settings. Defaults to `metric_config.yml` in this "
             "script directory.",
        default=None
    )
    parser.add_argument(
        "--variable-type",
        type=str,
        help="The type of the variable, as used in `--config-path` to select the appropriate metric settings. For "
             "example, 'temperature' or 'wind'.",
        required=True
    )
    parser.add_argument(
        "--output-directory",
        type=str,
        help="Output directory for all evaluation artifacts",
        required=True
    )
    parser.add_argument(
        "--eval-region-files",
        type=str,
        default=None,
        nargs='+',
        help="A list of files containing station lists for evaluation in certain regions"
    )
    parser.add_argument(
        "--start-lead",
        type=int,
        default=-1,
        help="First lead time (in hours) to include in evaluation"
    )
    parser.add_argument(
        "--end-lead",
        type=int,
        default=-1,
        help="Last lead time (in hours) to include in evaluation"
    )
    parser.add_argument(
        "--group-dim",
        type=str,
        default="lead_time",
        help="Group dimension for metric computation, options: lead_time, issue_time, valid_time"
    )
    parser.add_argument(
        "--convert-fcst-temperature-k-to-c",
        action='store_true',
        help="Convert forecast field from Kelvin to Celsius. Use only for evaluating temperature!"
    )
    parser.add_argument(
        "--cache-forecast",
        action='store_true',
        help="If true, cache the intermediate interpolated forecast data in the output directory."
    )
    parser.add_argument(
        "--align-forecasts",
        action='store_true',
        help="If set, load all forecasts first and then align them based on the intersection of issue/lead times. "
             "Note this uses substantially more memory to store all data at once."
    )
    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        help="Verbosity level for logging. Options are 0 (WARNING), 1 (INFO), 2 (DEBUG), 3 (NOTSET). Default is 1."
    )

    run_args = parser.parse_args()
    configure_logging(run_args.verbose)
    main(run_args)
