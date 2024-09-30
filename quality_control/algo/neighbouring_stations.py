import numpy as np
import xarray as xr
from .time_series import _time_series_comparison
from .utils import get_config, quality_control_statistics


CONFIG = get_config()


def _select_neighbouring_stations(similarity, stn_data, lat, lon, data, max_dist, max_elev_diff, min_data_overlap):
    """
    Select neighbouring stations based on distance, elevations and data overlap
    At most 8 stations will be selected, 2 at most for each direction (northwest, northeast, southwest, southeast)
    Parameters:
        similarity: xr.Dataset generated at the previous step with only the checked station
        stn_data: xr.DataArray with the data of the station
        lat: latitude of the station
        lon: longitude of the station
        data: xr.Dataset with the data of all stations
        max_dist: maximum distance in km
        max_elev_diff: maximum elevation difference in m
        min_data_overlap: minimum data overlap in fraction
    Return:
        candidates: list of neighbouring station names
    """
    similar_position = similarity["dist"] >= np.exp(-max_dist / 25)
    similar_elevation = similarity["elev"] >= np.exp(-max_elev_diff / 100)
    candidates = similarity["target"][similar_position & similar_elevation].values
    candidates = [item for item in candidates if item != similarity["station"].item()]

    # Filter out stations with insufficient data overlap
    is_valid = stn_data.notnull()
    if not is_valid.any():
        return []
    is_valid_neighboring = data.sel(station=candidates).notnull()
    overlap = (is_valid & is_valid_neighboring).sum(dim="time") / is_valid.sum()
    candidates = overlap["station"].values[overlap >= min_data_overlap]

    similarity = similarity.sel(target=candidates).sortby("dist", ascending=False)
    lon_diff = similarity["lon"] - lon
    # In case of the longitudes cross the central meridian
    iswest = ((lon_diff < 0) & (np.abs(lon_diff) < 180)) | ((lon_diff > 0) & (np.abs(lon_diff) > 180))
    isnorth = (similarity["lat"] - lat) > 0
    northwest_stations = similarity["target"][iswest & isnorth].values[:2]
    northeast_stations = similarity["target"][~iswest & isnorth].values[:2]
    southwest_stations = similarity["target"][iswest & ~isnorth].values[:2]
    southeast_stations = similarity["target"][~iswest & ~isnorth].values[:2]
    return np.concatenate([northwest_stations, northeast_stations, southwest_stations, southeast_stations])


def _load_similarity(fpath, dataset):
    """
    Load the similarity calculated at the previous step
    """
    similarity = xr.load_dataset(fpath)
    similarity = similarity.sel(station1=dataset["station"].values, station2=dataset["station"].values)
    similarity = similarity.assign_coords(lon=("station2", dataset["lon"].values))
    similarity = similarity.assign_coords(lat=("station2", dataset["lat"].values))
    similarity = similarity.rename({"station1": "station", "station2": "target"})
    return similarity


def _neighbouring_stations_check_base(
    flag,
    similarity,
    data,
    max_dist,
    max_elev_diff,
    min_data_overlap,
    shift_step,
    gap_scale,
    default_mad,
    suspect_std_scale,
    min_num,
):
    """
    Check all stations in data by comparing them with neighbouring stations
    For each time step, when there are at least 3 neighbouring values available,
    and 2 / 3 of them are in agreement (either verified or dubious), save the result
    Parameters:
    -----------
        flag: a initialized DataArray with the same station dimension as the data
        similarity: similarity matrix between stations used for selecting neighbouring stations
        data: DataArray to be checked
        max_dist/max_elev_diff/min_data_overlap: parameters for selecting neighbouring stations
        shift_step/gap_scale/default_mad/suspect_std_scale: parameters for the time series comparison
        min_num: minimum number of valid data points to perform the check
    Returns:
    --------
    flag: Updated flag DataArray
    """
    for station in flag["station"].values:
        neighbours = _select_neighbouring_stations(
            similarity.sel(station=station),
            data.sel(station=station),
            lat=data.sel(station=station)["lat"].item(),
            lon=data.sel(station=station)["lon"].item(),
            data=data,
            max_dist=max_dist,
            max_elev_diff=max_elev_diff,
            min_data_overlap=min_data_overlap,
        )
        # Only apply the check to stations with at least 3 neighbouring stations
        if len(neighbours) < 3:
            continue
        results = []

        for target in neighbours:
            ith_flag = _time_series_comparison(
                data.sel(station=station).values,
                data.sel(station=target).values,
                shift_step=shift_step,
                gap_scale=gap_scale,
                default_mad=default_mad,
                suspect_std_scale=suspect_std_scale,
                min_num=min_num,
            )
            results.append(ith_flag)
        results = np.stack(results, axis=0)
        # For each time step, count the number of valid neighbouring data points
        num_neighbours = np.sum(results != CONFIG["flag_missing"], axis=0)
        # For each time step, still only consider the data points with at least 3 neighbouring data points
        num_neighbours = np.where(num_neighbours < 3, np.nan, num_neighbours)
        # Mask when 2 / 3 of the neighbouring data points are in agreement
        # Specifically, set erroneous only when there is no normal flags
        min_stn = num_neighbours * 2 / 3
        normal = (results == CONFIG["flag_normal"]).sum(axis=0)
        suspect = ((results == CONFIG["flag_suspect"]) | (results == CONFIG["flag_error"])).sum(axis=0)
        erroneous = (results == CONFIG["flag_error"]).sum(axis=0)
        aggregated = np.full_like(normal, CONFIG["flag_missing"])
        aggregated = np.where(normal >= min_stn, CONFIG["flag_normal"], aggregated)
        aggregated = np.where(suspect >= min_stn, CONFIG["flag_suspect"], aggregated)
        aggregated = np.where((erroneous >= min_stn) & (normal == 0), CONFIG["flag_error"], aggregated)
        flag.loc[{"station": station}] = aggregated
    return flag


def run(da, f_similarity, varname):
    """
    Check the data by comparing with neighbouring stations
    Time series comparison is performed for each station with at least 3 neighbouring stations
    `map_blocks` is used to parallelize the calculation
    """
    similarity = _load_similarity(f_similarity, da)

    flag = xr.DataArray(
        np.full(da.shape, CONFIG["flag_missing"], dtype=np.int8),
        dims=["station", "time"],
        coords={k: da.coords[k].values for k in da.dims}
    )
    ret = xr.map_blocks(
        _neighbouring_stations_check_base,
        flag.chunk({"station": 500}),
        args=(similarity.chunk({"station": 500}), ),
        kwargs={"data": da, **CONFIG["neighbouring"][varname]},
        template=flag.chunk({"station": 500}),
    ).compute(scheduler='processes')
    ret = ret.where(da.notnull(), CONFIG["flag_missing"])
    quality_control_statistics(da, ret)
    return ret.rename("neighbouring_stations")
