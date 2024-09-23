import xarray as xr
from .utils import CONFIG, intra_station_check, quality_control_statistics
from .time_series import _time_series_comparison


def _distributional_gap(
    ts,
    reanalysis,
    mask,
    shift_step,
    gap_scale,
    default_mad,
    suspect_std_scale,
    min_mad=0.1,
    min_num=None,
):
    flag = _time_series_comparison(
        ts1=ts,
        ts2=reanalysis,
        shift_step=shift_step,
        gap_scale=gap_scale,
        default_mad=default_mad,
        suspect_std_scale=suspect_std_scale,
        min_mad=min_mad,
        min_num=min_num,
        mask=mask,
    )
    return flag


def run(da, reanalysis, varname, mask=None):
    """
    Perform a distributional gap check on time series data compared to reanalysis data.
    """
    flag = intra_station_check(
        da,
        reanalysis,
        mask if mask is not None else xr.full_like(da, True, dtype=bool),
        qc_func=_distributional_gap,
        input_core_dims=[["time"], ["time"], ["time"]],
        kwargs=CONFIG["distribution"][varname],
    )
    quality_control_statistics(da, flag)
    return flag.rename("distributional_gap")
