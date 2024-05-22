import numpy as np
import pandas as pd
import xarray as xr

from .utils import ForecastInfo


def get_interp_station_list(interp_station_path: str) -> xr.Dataset:
    """
    Read the station metadata from the interpolation station list file, and return a dataset for interpolation.

    Parameters
    ----------
    interp_station_path: str: path to the interpolation station list

    Returns
    -------
    xr.Dataset: station metadata in dataset
    """
    interp_station = pd.read_csv(interp_station_path)
    interp_station = interp_station.rename({c: c.lower() for c in interp_station.columns}, axis=1)
    interp_station['lon'] = interp_station['lon'].apply(lambda lon: lon if (lon < 180) else (lon - 360))
    if 'station' not in interp_station.columns:
        interp_station['station'] = interp_station['id']
    metadata = xr.Dataset.from_dataframe(interp_station[['station', 'lon', 'lat']].set_index(['station']))
    return metadata


def convert_grid_to_point(grid_forecast: xr.Dataset, interp_station_path: str) -> xr.Dataset:
    """
    Convert grid forecast to point dataframe via interpolation
    input data dims must be: lead_time, issue_time, lat, lon

    Parameters
    ----------
    grid_forecast: xarray Dataset: grid forecast data
    interp_station_path: str: path to the interpolation station list

    Returns
    -------
    xr.Dataset: interpolated forecast data
    """
    metadata = get_interp_station_list(interp_station_path)

    # grid_forecast = grid_forecast.load()
    grid_forecast = grid_forecast.assign_coords(lon=[lon if (lon < 180) else (lon - 360)
                                                     for lon in grid_forecast['lon'].values])
    # Optionally roll the longitude if the minimum longitude is not at index 0. This should make longitudes
    # monotonically increasing.
    if grid_forecast['lon'].argmin().values != 0:
        grid_forecast = grid_forecast.roll(lon=grid_forecast['lon'].argmin().values, roll_coords=True)

    # Default interpolate
    interp_forecast = grid_forecast.interp(lon=metadata['lon'], lat=metadata['lat'], method='linear')
    return interp_forecast


def get_lead_time_slice(start_lead, end_lead):
    start = pd.Timedelta(start_lead, 'H') if start_lead >= 0 else None
    end = pd.Timedelta(end_lead, 'H') if end_lead > 0 else None
    return slice(start, end)


def reformat_filter_forecast(forecast: xr.Dataset, info: ForecastInfo, start_lead: int, end_lead: int,
                             convert_t: bool) -> xr.Dataset:
    """
    Format the forecast data to the required format for evaluation, and keep only the required stations.

    Parameters
    ----------
    forecast: xarray Dataset: forecast data
    info: ForecastInfo: forecast info
    start_lead: int: start lead time
    end_lead: int: end lead time
    convert_t: bool: if True, convert temperature to Celsius

    Returns
    -------
    xr.Dataset: formatted forecast data
    """
    if info.reformat_func in ['grid_v1']:
        reformat_data = reformat_grid_v1(forecast, info, start_lead, end_lead)
    elif info.reformat_func in ['omg_v1', 'grid_v2']:
        reformat_data = reformat_grid_v2(forecast, info, start_lead, end_lead)
    elif info.reformat_func in ['grid_standard']:
        reformat_data = reformat_grid_standard(forecast, info, start_lead, end_lead)
    elif info.reformat_func in ['point_standard']:
        reformat_data = reformat_point_standard(forecast, info, start_lead, end_lead)
    else:
        raise ValueError(f"Unknown reformat method {info.reformat_func} for forecast {info.forecast_name}")

    if convert_t and info.reformat_func != 'omg_v1':  # omg_v1 already converted
        reformat_data['fc'] -= 273.15

    # Convert coordinates
    reformat_data = reformat_data.assign_coords(valid_time=reformat_data['issue_time'] + reformat_data['lead_time'])
    reformat_data = reformat_data.assign_coords(lead_time=reformat_data['lead_time'] / np.timedelta64(1, 'h'))
    return reformat_data


def reformat_grid_v1(grid_forecast: xr.Dataset, info: ForecastInfo, start_lead: int, end_lead: int) -> xr.Dataset:
    """
    Standard grid forecast format following ECMWF schema
    input nc|zarr file dims must be: time, step, latitude, longitude

    Parameters
    ----------
    grid_forecast: xarray Dataset: grid forecast data
    info: dict: forecast info
    start_lead: int: start lead time
    end_lead: int: end lead time

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
    grid_forecast = grid_forecast.sel(lead_time=get_lead_time_slice(start_lead, end_lead))
    interp_forecast = convert_grid_to_point(grid_forecast[['fc']], info.interp_station_path)
    return interp_forecast


def reformat_grid_v2(grid_forecast: xr.Dataset, info: ForecastInfo, start_lead: int, end_lead: int) -> xr.Dataset:
    """
    Grid format following OMG schema
    input nc|zarr file dims must be: lead_time, issue_time, y, x, index
    Must have lat, lon, index, var_name as coordinates

    Parameters
    ----------
    grid_forecast: xarray Dataset: grid forecast data
    info: dict: forecast info
    start_lead: int: start lead time
    end_lead: int: end lead time

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
    grid_forecast = grid_forecast.sel(lead_time=get_lead_time_slice(start_lead, end_lead))
    interp_forecast = convert_grid_to_point(grid_forecast[['fc']], info.interp_station_path)
    return interp_forecast


def reformat_grid_standard(grid_forecast: xr.Dataset, info: ForecastInfo, start_lead: int, end_lead: int) -> \
        xr.Dataset:
    """
    Grid format following standard schema
    input nc|zarr file dims must be: lead_time, issue_time, lat, lon

    Parameters
    ----------
    grid_forecast: xarray Dataset: grid forecast data
    info: dict: forecast info
    start_lead: int: start lead time
    end_lead: int: end lead time

    Returns
    -------
    xr.Dataset: formatted forecast data
    """
    fc_var_name = info.fc_var_name
    grid_forecast = grid_forecast.rename({fc_var_name: 'fc'})
    grid_forecast = grid_forecast.sel(lead_time=get_lead_time_slice(start_lead, end_lead))
    interp_forecast = convert_grid_to_point(grid_forecast[['fc']], info.interp_station_path)
    return interp_forecast


def reformat_point_standard(point_forecast: xr.Dataset, info: ForecastInfo, start_lead: int, end_lead: int) \
        -> xr.Dataset:
    """
    Standard point forecast format
    input nc|zarr file dims must be: lead_time, issue_time, station

    Parameters
    ----------
    point_forecast: xarray Dataset: grid forecast data
    info: dict: forecast info
    start_lead: int: start lead time
    end_lead: int: end lead time

    Returns
    -------
    xr.Dataset: formatted forecast data
    """
    fc_var_name = info.fc_var_name
    point_forecast = point_forecast.rename({fc_var_name: 'fc'})
    point_forecast = point_forecast.sel(lead_time=get_lead_time_slice(start_lead, end_lead))
    return point_forecast[['fc']]
