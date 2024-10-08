import logging
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import xarray as xr
try:
    from metpy.calc import specific_humidity_from_dewpoint
    from metpy.units import units
except ImportError:
    specific_humidity_from_dewpoint = None
    units = None

from .utils import convert_to_binary

logger = logging.getLogger(__name__)


def _convert_time_step(dt):  # pylint: disable=invalid-name
    return pd.Timedelta(hours=dt) if isinstance(dt, (float, int)) else pd.Timedelta(dt)


def reformat_and_filter_obs(obs: xr.Dataset, obs_var_name: str, interp_station_path: Optional[str],
                            precip_threshold: Optional[float] = None) -> xr.Dataset:
    """
    Reformat and filter the observation data, and return the DataFrame with required fields.
    Required fields: station, valid_time, obs
    Filter removes the stations that are not in the interp_station list

    Parameters
    ----------
    obs: xarray Dataset: observation data
    obs_var_name: str: observation variable name
    interp_station_path: str: path to the interpolation station list
    precip_threshold: float: threshold for precipitation

    Returns
    -------
    xr.Dataset: formatted observation data
    """
    if interp_station_path is not None:
        interp_station = get_interp_station_list(interp_station_path)
        intersect_station = np.intersect1d(interp_station['station'].values, obs['station'].values)
        logger.debug(f"intersect_station count: {len(intersect_station)}, \
                     obs_station count: {len(obs['station'].values)}, \
                     interp_station count: {len(interp_station['station'].values)}")
        obs = obs.sel(station=intersect_station)
    if 'valid_time' not in obs.dims:
        obs = obs.rename({'time': 'valid_time'})

    if obs_var_name in ['u10', 'v10']:
        obs = calculate_u_v(obs, ws_name='ws', wd_name='wd', u_name='u10', v_name='v10')
    elif obs_var_name == 'q':
        obs['q'] = xr.apply_ufunc(calculate_q, obs['td'])
    elif obs_var_name == 'pp':
        precip_var = 'ra' if 'ra' in obs.data_vars else 'ra1'
        logger.info(f"User requested precipitation probability from obs. Using variable '{precip_var}' with "
                    f"threshold of 0.1 mm/hr.")
        obs['pp'] = convert_to_binary(obs[precip_var], 0.1)

    if precip_threshold is not None:
        obs[obs_var_name] = convert_to_binary(obs[obs_var_name], precip_threshold)

    return obs[[obs_var_name]].rename({obs_var_name: 'obs'})


def obs_to_verification(
        obs: Union[xr.Dataset, xr.DataArray],
        max_lead: Union[pd.Timedelta, int] = 168,
        steps: Optional[Sequence[pd.Timestamp]] = None,
        issue_times: Optional[Sequence[pd.Timestamp]] = None
) -> Union[xr.Dataset, xr.DataArray]:
    """
    Convert a Dataset or DataArray of continuous time-series observations
    according to the obs data spec into the forecast data spec for direct
    comparison to forecasts.

    Parameters
    ----------
    obs: xarray Dataset or DataArray of observation data
    max_lead: maximum lead time for verification dataset. If int, interpreted
        as hours.
    steps: optional sequence of lead times to retain
    issue_times: issue times for the forecast result. If
        not specified, uses all available obs times.
    """
    issue_dim = 'issue_time'
    lead_dim = 'lead_time'
    time_dim = 'valid_time'
    max_lead = _convert_time_step(max_lead)
    if issue_times is None:
        issue_times = obs[time_dim].values
    obs_series = []
    for issue in issue_times:
        try:
            obs_series.append(
                obs.sel(**{time_dim: slice(issue, issue + max_lead)}).rename({time_dim: lead_dim})
            )
            obs_series[-1] = obs_series[-1].assign_coords(
                **{issue_dim: [issue], lead_dim: obs_series[-1][lead_dim] - issue})
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f'Failed to sel {issue} due to {e}')
            continue
    verification_ds = xr.concat(obs_series, dim=issue_dim)
    if steps is not None:
        steps = [_convert_time_step(s) for s in steps]
        verification_ds = verification_ds.sel(**{lead_dim: steps})
    verification_ds = verification_ds.assign_coords({lead_dim: verification_ds[lead_dim] / np.timedelta64(1, 'h')})

    return verification_ds


def calculate_q(td):
    if specific_humidity_from_dewpoint is None:
        raise ImportError('metpy is not installed, specific_humidity_from_dewpoint cannot be calculated')
    q = specific_humidity_from_dewpoint(1013.25 * units.hPa, td * units.degC).magnitude
    return q


def calculate_u_v(data, ws_name='ws', wd_name='wd', u_name='u10', v_name='v10'):
    data[u_name] = data[ws_name] * np.sin(data[wd_name] / 180 * np.pi - np.pi)
    data[v_name] = data[ws_name] * np.cos(data[wd_name] / 180 * np.pi - np.pi)
    return data


def get_interp_station_list(interp_station_path: str) -> pd.DataFrame:
    """
    Read the station metadata from the interpolation station list file, and return a dataset for interpolation.

    Parameters
    ----------
    interp_station_path: str: path to the interpolation station list

    Returns
    -------
    pd.DataFrame: station metadata with columns station, lat, lon
    """
    interp_station = pd.read_csv(interp_station_path)
    interp_station = interp_station.rename({c: c.lower() for c in interp_station.columns}, axis=1)
    interp_station['lon'] = interp_station['lon'].apply(lambda lon: lon if (lon < 180) else (lon - 360))
    if 'station' not in interp_station.columns:
        interp_station['station'] = interp_station['id']
    return interp_station
