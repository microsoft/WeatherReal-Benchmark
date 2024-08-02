import logging

import numpy as np
import pandas as pd
import xarray as xr

from .utils import convert_to_binary, ForecastInfo

logger = logging.getLogger(__name__)


def convert_grid_to_point(grid_forecast: xr.Dataset, metadata: pd.DataFrame) -> xr.Dataset:
    """
    Convert grid forecast to point dataframe via interpolation
    input data dims must be: lead_time, issue_time, lat, lon

    Parameters
    ----------
    grid_forecast: xarray Dataset: grid forecast data
    metadata: pd.DataFrame: station metadata

    Returns
    -------
    xr.Dataset: interpolated forecast data
    """
    # grid_forecast = grid_forecast.load()
    grid_forecast = grid_forecast.assign_coords(lon=[lon if (lon < 180) else (lon - 360)
                                                     for lon in grid_forecast['lon'].values])
    # Optionally roll the longitude if the minimum longitude is not at index 0. This should make longitudes
    # monotonically increasing.
    if grid_forecast['lon'].argmin().values != 0:
        grid_forecast = grid_forecast.roll(lon=grid_forecast['lon'].argmin().values, roll_coords=True)

    # Default interpolate
    interp_meta = xr.Dataset.from_dataframe(metadata[['station', 'lon', 'lat']].set_index(['station']))
    interp_forecast = grid_forecast.interp(lon=interp_meta['lon'], lat=interp_meta['lat'], method='linear')
    return interp_forecast


def get_lead_time_slice(start_lead, end_lead):
    start = pd.Timedelta(start_lead, 'H') if start_lead is not None else None
    end = pd.Timedelta(end_lead, 'H') if end_lead is not None else None
    return slice(start, end)


def update_unit_conversions(forecast: xr.Dataset, info: ForecastInfo):
    """
    Check if the forecast data needs to be converted to the required units. Will not perform correction if
    not specified by user, however, if specified and data are already in converted units, then do not perform
    the conversion.

    Parameters
    ----------
    forecast: xarray Dataset: forecast data
    info: ForecastInfo: forecast info
    """
    if info.reformat_func == 'omg_v1':
        logger.info(f"Unit conversion not needed for forecast {info.forecast_name}")
        return
    unit = forecast['fc'].attrs.get('units', '') or forecast['fc'].attrs.get('unit', '')
    if info.convert_temperature:
        if 'C' in unit:
            info.convert_temperature = False
            logger.info(f"Temperature conversion not needed for forecast {info.forecast_name}")
    if info.convert_pressure:
        if any(u in unit.lower() for u in ['hpa', 'mb', 'millibar']):
            info.convert_pressure = False
            logger.info(f"Pressure conversion not needed for forecast {info.forecast_name}")
    if info.convert_cloud:
        if any(u in unit.lower() for u in ['okta', '0-8']):
            info.convert_cloud = False
            logger.info(f"Cloud cover conversion not needed for forecast {info.forecast_name}")
    if info.precip_proba_threshold is not None:
        if not any(u in unit.lower() for u in ['mm', 'milli']):
            info.precip_proba_threshold /= 1e3
            logger.info(f"Probability thresholding not needed for forecast {info.forecast_name}")


def convert_cloud(forecast: xr.Dataset):
    """
    Convert cloud cover inplace from percentage or fraction to okta. Assumes fraction by default unless units
    attribute says otherwise.

    Parameters
    ----------
    forecast: xarray Dataset: forecast data

    Returns
    -------
    xr.Dataset: converted forecast data
    """
    unit = forecast['fc'].attrs.get('units', '') or forecast['fc'].attrs.get('unit', '')
    if any(u in unit.lower() for u in ['percent', '%', '100']):
        forecast['fc'] *= 8 / 100.
    else:
        forecast['fc'] *= 8
    return forecast


def convert_precip_binary(forecast: xr.Dataset, info: ForecastInfo):
    """
    Convert precipitation forecast to binary based on the threshold in the forecast info

    Parameters
    ----------
    forecast: xarray Dataset: forecast data
    info: ForecastInfo: forecast info

    Returns
    -------
    xr.Dataset: converted forecast data
    """
    if 'pp' in info.fc_var_name:
        logger.info(f"Forecast {info.forecast_name} is already in probability (%) format. Skipping thresholding.")
        forecast['fc'] /= 100.
        return forecast
    forecast['fc'] = convert_to_binary(forecast['fc'], info.precip_proba_threshold)
    return forecast


def reformat_forecast(forecast: xr.Dataset, info: ForecastInfo) -> xr.Dataset:
    """
    Format the forecast data to the required format for evaluation, and keep only the required stations.

    Parameters
    ----------
    forecast: xarray Dataset: forecast data
    info: ForecastInfo: forecast info

    Returns
    -------
    xr.Dataset: formatted forecast data
    """
    if info.reformat_func in ['grid_v1']:
        reformat_data = reformat_grid_v1(forecast, info)
    elif info.reformat_func in ['omg_v1', 'grid_v2']:
        reformat_data = reformat_grid_v2(forecast, info)
    elif info.reformat_func in ['grid_standard']:
        reformat_data = reformat_grid_standard(forecast, info)
    elif info.reformat_func in ['point_standard']:
        reformat_data = reformat_point_standard(forecast, info)
    else:
        raise ValueError(f"Unknown reformat method {info.reformat_func} for forecast {info.forecast_name}")

    # Update unit conversions
    update_unit_conversions(reformat_data, info)
    if info.convert_temperature:
        reformat_data['fc'] -= 273.15
    if info.convert_pressure:
        reformat_data['fc'] /= 100
    if info.convert_cloud:
        convert_cloud(reformat_data)
    if info.precip_proba_threshold is not None:
        convert_precip_binary(reformat_data, info)

    # Convert coordinates
    reformat_data = reformat_data.assign_coords(valid_time=reformat_data['issue_time'] + reformat_data['lead_time'])
    reformat_data = reformat_data.assign_coords(lead_time=reformat_data['lead_time'] / np.timedelta64(1, 'h'))
    return reformat_data


def select_forecasts(forecast: xr.Dataset, info: ForecastInfo) -> xr.Dataset:
    """
    Select the forecast data based on the issue time and lead time

    Parameters
    ----------
    forecast: xarray Dataset: forecast data
    info: ForecastInfo: forecast info

    Returns
    -------
    xr.Dataset: selected forecast data
    """
    forecast = forecast.sel(
        lead_time=get_lead_time_slice(info.start_lead, info.end_lead),
        issue_time=slice(info.start_date, info.end_date)
    )
    if info.issue_time_freq is not None:
        forecast = forecast.resample(issue_time=info.issue_time_freq).nearest()
    return forecast


def reformat_grid_v1(grid_forecast: xr.Dataset, info: ForecastInfo) -> xr.Dataset:
    """
    Standard grid forecast format following ECMWF schema
    input nc|zarr file dims must be: time, step, latitude, longitude

    Parameters
    ----------
    grid_forecast: xarray Dataset: grid forecast data
    info: dict: forecast info

    Returns
    -------
    xr.Dataset: formatted forecast data
    """
    grid_forecast = grid_forecast.rename(
        {
            'latitude': 'lat',
            'longitude': 'lon',
            'step': 'lead_time',
            'time': 'issue_time',
            info.fc_var_name: 'fc'
        }
    )
    grid_forecast = select_forecasts(grid_forecast, info)
    interp_forecast = convert_grid_to_point(grid_forecast[['fc']], info.metadata)
    return interp_forecast


def reformat_grid_v2(grid_forecast: xr.Dataset, info: ForecastInfo) -> xr.Dataset:
    """
    Grid format following OMG schema
    input nc|zarr file dims must be: lead_time, issue_time, y, x, index
    Must have lat, lon, index, var_name as coordinates

    Parameters
    ----------
    grid_forecast: xarray Dataset: grid forecast data
    info: dict: forecast info

    Returns
    -------
    xr.Dataset: formatted forecast data
    """
    lat = grid_forecast['lat'].isel(index=0).lat.values
    lon = grid_forecast['lon'].isel(index=0).lon.values
    issue_time = grid_forecast['issue_time'].values
    grid_forecast = grid_forecast.drop_vars(['lat', 'lon', 'index', 'var_name'])
    grid_forecast = grid_forecast.rename({'y': 'lat', 'x': 'lon', 'index': 'issue_time'})
    grid_forecast = grid_forecast.assign_coords(lat=lat, lon=lon, issue_time=issue_time)
    grid_forecast = grid_forecast.squeeze(dim='var_name')

    grid_forecast = grid_forecast.rename({info.fc_var_name: 'fc'})
    grid_forecast = select_forecasts(grid_forecast, info)
    interp_forecast = convert_grid_to_point(grid_forecast[['fc']], info.metadata)
    return interp_forecast


def reformat_grid_standard(grid_forecast: xr.Dataset, info: ForecastInfo) -> \
        xr.Dataset:
    """
    Grid format following standard schema
    input nc|zarr file dims must be: lead_time, issue_time, lat, lon

    Parameters
    ----------
    grid_forecast: xarray Dataset: grid forecast data
    info: dict: forecast info

    Returns
    -------
    xr.Dataset: formatted forecast data
    """
    fc_var_name = info.fc_var_name
    grid_forecast = grid_forecast.rename({fc_var_name: 'fc'})
    grid_forecast = select_forecasts(grid_forecast, info)
    interp_forecast = convert_grid_to_point(grid_forecast[['fc']], info.metadata)
    return interp_forecast


def reformat_point_standard(point_forecast: xr.Dataset, info: ForecastInfo) \
        -> xr.Dataset:
    """
    Standard point forecast format
    input nc|zarr file dims must be: lead_time, issue_time, station

    Parameters
    ----------
    point_forecast: xarray Dataset: grid forecast data
    info: dict: forecast info

    Returns
    -------
    xr.Dataset: formatted forecast data
    """
    fc_var_name = info.fc_var_name
    point_forecast = point_forecast.rename({fc_var_name: 'fc'})
    point_forecast = select_forecasts(point_forecast, info)
    return point_forecast[['fc']]
