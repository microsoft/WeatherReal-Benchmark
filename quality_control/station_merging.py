"""
1. Load the metadata of ISD data online. Post-processing includes:
   1. Drop rows with missing values. These stations are considered not trustworthy
   2. Use USAF and WBAN to create a unique station ID
2. Calculate pairwise similarity between stations and saved as NetCDF files
   - Distance and elevation similarity is calculated using an exponential decay
   - Distance of two stations are calculated using great circle distance
3. Load ISD hourly station data with valid metadata, and merge stations based on both metadata and data similarity
   - Metadata similarity is calculated based on a weighted sum of distance, elevation, and name similarity
   - Data similarity is calculated based on the proportion of identical values among common data points
4. The merged data and the similarity will be saved in NetCDF format
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
import xarray as xr
from algo.utils import configure_logging
from geopy.distance import great_circle as geodist
from tqdm import tqdm

# URL of the official ISD metadata file
URL_ISD_HISTORY = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.txt"
logger = logging.getLogger(__name__)


def load_metadata(station_list):
    """
    Load the metadata of ISD data
    """
    meta = pd.read_fwf(
        URL_ISD_HISTORY,
        skiprows=20,
        usecols=["USAF", "WBAN", "STATION NAME", "CTRY", "CALL", "LAT", "LON", "ELEV(M)"],
        dtype={"USAF": str, "WBAN": str},
    )
    # Drop rows with missing values. These stations are considered not trustworthy
    meta = meta.dropna(how="any", subset=["LAT", "LON", "STATION NAME", "ELEV(M)", "CTRY"])
    meta["STATION"] = meta["USAF"] + meta["WBAN"]
    meta = meta[["STATION", "CALL", "STATION NAME", "CTRY", "LAT", "LON", "ELEV(M)"]].set_index("STATION", drop=True)
    return meta[meta.index.isin(station_list)]


def calc_distance_similarity(latlon1, latlon2, scale_dist=25):
    distance = geodist(latlon1, latlon2).kilometers
    similarity = np.exp(-distance / scale_dist)
    return similarity


def calc_elevation_similarity(elevation1, elevation2, scale_elev=100):
    similarity = np.exp(-abs(elevation1 - elevation2) / scale_elev)
    return similarity


def calc_id_similarity(ids1, ids2):
    """
    Compare the USAF/WBAN/CALL IDs of two stations
    """
    usaf1, wban1, call1, ctry1 = ids1
    usaf2, wban2, call2, ctry2 = ids2
    if usaf1 != "999999" and usaf1 == usaf2:
        return 1
    if wban1 != "99999" and wban1 == wban2:
        return 1
    if call1 == call2:
        return 1
    # A special case for the CALL ID, e.g., KAIO and AIO are the same stations
    if isinstance(call1, str) and len(call1) == 3 and ("K" + call1) == call2:
        return 1
    if isinstance(call2, str) and len(call2) == 3 and ("K" + call2) == call1:
        return 1
    # For a special case in Germany, 09xxxx and 10xxxx are the same stations
    # See https://gi.copernicus.org/articles/5/473/2016/gi-5-473-2016.html
    if usaf1.startswith("09") and usaf2.startswith("10") and usaf1[2:] == usaf2[2:] and ctry1 == ctry2 == "DE":
        return 1
    return 0


def calc_name_similarity(name1, name2):
    """
    Jaccard Index for calculating name similarity
    """
    set1 = set(name1)
    set2 = set(name2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    jaccard_index = len(intersection) / len(union)
    return jaccard_index


def post_process_similarity(similarity, ids):
    """
    Post-process the similarity matrix
    """
    # Fill the lower triangle of the matrix
    similarity = similarity + similarity.T
    # Set the diagonal (self-similarity) to 1
    np.fill_diagonal(similarity, 1)
    similarity = xr.DataArray(similarity, dims=["station1", "station2"], coords={"station1": ids, "station2": ids})
    return similarity


def calc_similarity(meta, scale_dist=25, scale_elev=100):
    """
    Calculate pairwise similarity between stations
    1. Distance similarity: great circle distance between two stations using an exponential decay
    2. Elevation similarity: absolute difference of elevation between two stations using an exponential decay
    3. ID similarity: whether the USAF/WBAN/CALL IDs of two stations are the same
    """
    latlon = meta[["LAT", "LON"]].apply(tuple, axis=1).values
    elev = meta["ELEV(M)"].values
    usaf = meta.index.str[:6].values
    wban = meta.index.str[6:].values
    name = meta["STATION NAME"].values
    ids = list(zip(usaf, wban, meta["CALL"].values, meta["CTRY"].values))
    num = len(meta)
    dist_similarity = np.zeros((num, num))
    elev_similarity = np.zeros((num, num))
    id_similarity = np.zeros((num, num))
    name_similarity = np.zeros((num, num))
    for idx1 in tqdm(range(num - 1), desc="Calculating similarity"):
        for idx2 in range(idx1 + 1, num):
            dist_similarity[idx1, idx2] = calc_distance_similarity(latlon[idx1], latlon[idx2], scale_dist)
            elev_similarity[idx1, idx2] = calc_elevation_similarity(elev[idx1], elev[idx2], scale_elev)
            id_similarity[idx1, idx2] = calc_id_similarity(ids[idx1], ids[idx2])
            name_similarity[idx1, idx2] = calc_name_similarity(name[idx1], name[idx2])
    dist_similarity = post_process_similarity(dist_similarity, meta.index.values)
    elev_similarity = post_process_similarity(elev_similarity, meta.index.values)
    id_similarity = post_process_similarity(id_similarity, meta.index.values)
    name_similarity = post_process_similarity(name_similarity, meta.index.values)
    similarity = xr.merge(
        [
            dist_similarity.rename("dist"),
            elev_similarity.rename("elev"),
            id_similarity.rename("id"),
            name_similarity.rename("name"),
        ]
    )
    return similarity


def load_raw_data(data_dir, station_list):
    """
    Load raw hourly ISD data in csv files
    """
    data = []
    for stn in tqdm(station_list, desc="Loading data"):
        df = pd.read_csv(os.path.join(data_dir, f"{stn}.csv"), index_col="time", parse_dates=["time"])
        df["station"] = stn
        df = df.set_index("station", append=True)
        data.append(df.to_xarray())
    data = xr.concat(data, dim="station")
    data["time"] = pd.to_datetime(data["time"])
    return data


def calc_meta_similarity(similarity):
    """
    Calculate the metadata similarity according to horizontal distance, elevation, and name similarity
    """
    meta_simi = (similarity["dist"] * 9 + similarity["elev"] * 1 + similarity["name"] * 5) / 15
    # meta_simi is set to 1 if IDs are the same,
    # or it is set to a weighted sum of distance, elevation, and name similarity
    meta_simi = np.maximum(meta_simi, similarity["id"])
    # set the diagonal and lower triangle to NaN to avoid duplicated pairs
    rows, cols = np.indices(meta_simi.shape)
    meta_simi.values[rows >= cols] = np.nan
    return meta_simi


def need_merge(da, stn_source, stn_target, threshold=0.7):
    """
    Distinguish whether two stations need to be merged based on the similarity of their data
    It is possible that one of them only has few data points
    In this case, it can be treated as removing low-quality stations
    """
    ts1 = da.sel(station=stn_source)
    ts2 = da.sel(station=stn_target)
    diff = np.abs(ts1 - ts2)
    if (ts1.dropna(dim="time") % 1 < 1e-3).all() or (ts2.dropna(dim="time") % 1 < 1e-3).all():
        max_diff = 0.5
    else:
        max_diff = 0.1
    data_simi = (diff <= max_diff).sum() / diff.notnull().sum()
    return data_simi.item() >= threshold


def merge_pairs(ds1, ds2):
    """
    Merge two stations. Each of the two ds should have only one station
    If there are only one variable, fill the missing values in ds1 with ds2
    If there are more than one variables, to ensure that all variables are from the same station,
    for each timestep, the ds with more valid variables will be selected
    """
    if len(ds1.data_vars) == 1:
        return ds1.fillna(ds2)
    else:
        da1 = ds1.to_array()
        da2 = ds2.to_array()
        mask = da1.count(dim="variable") >= da2.count(dim="variable")
        return xr.where(mask, da1, da2).to_dataset(dim="variable")


def merge_stations(ds, meta_simi, main_var, appendant_var=[], meta_simi_th=0.35, data_simi_th=0.7):
    """
    For ds, merge stations based on metadata similarity and data similarity
    """
    result = []
    # Flags to avoid duplications
    is_merged = xr.DataArray(
        np.full(ds["station"].size, False), dims=["station"], coords={"station": ds["station"].values}
    )
    for station in tqdm(ds["station"].values, desc=f"Merging {main_var}"):
        if is_merged.sel(station=station).item():
            continue
        # Station list to be merged
        merged_stations = [station]
        # Candidates that pass the metadata similarity threshold
        candidates = meta_simi["station2"][meta_simi.sel(station1=station) >= meta_simi_th].values
        # Stack to store the station pairs to be checked
        stack = [(station, item) for item in candidates]
        # Search for all stations that need to be merged
        # If A and B should be merged, and B and C should be merged, then all of them are merged together
        while stack:
            stn_source, stn_target = stack.pop()
            if stn_target in merged_stations:
                continue
            if need_merge(ds[main_var], stn_source, stn_target, threshold=data_simi_th):
                is_merged.loc[stn_target] = True
                merged_stations.append(stn_target)
                candidates = meta_simi["station2"][meta_simi.sel(station1=stn_target) >= meta_simi_th].values
                stack.extend([(stn_target, item) for item in candidates])
        # Merge stations according to the number of valid data points
        num_valid = ds[main_var].sel(station=merged_stations).notnull().sum(dim="time")
        sorted_stns = num_valid["station"].sortby(num_valid).values
        variables = [main_var] + appendant_var
        stn_data = ds[variables].sel(station=sorted_stns[0])
        for target_stn in sorted_stns[1:]:
            stn_data = merge_pairs(stn_data, ds[variables].sel(station=target_stn))
        stn_data = stn_data.assign_coords(station=station)
        result.append(stn_data)
    result = xr.concat(result, dim="station")
    return result


def merge_all_variables(data, meta_simi, meta_simi_th=0.35, data_simi_th=0.7):
    """
    Merge stations for each variable
    The key is the main variable used to compare, and the value is the list of appendant variables
    """
    variables = {
        "t": ["td"],
        "ws": ["wd"],
        "sp": [],
        "msl": [],
        "c": [],
        "ra1": ["ra3", "ra6", "ra12", "ra24"],
    }
    merged = []
    for var in variables:
        ret = merge_stations(data, meta_simi, var, variables[var], meta_simi_th=meta_simi_th, data_simi_th=data_simi_th)
        merged.append(ret)
    merged = xr.merge(merged).dropna(dim="station", how="all")
    return merged


def assign_meta_coords(ds, meta):
    meta = meta.loc[ds["station"].values]
    ds = ds.assign_coords(
        call=("station", meta["CALL"].values),
        name=("station", meta["STATION NAME"].values),
        lat=("station", meta["LAT"].values.astype(np.float32)),
        lon=("station", meta["LON"].values.astype(np.float32)),
        elev=("station", meta["ELEV(M)"].values.astype(np.float32)),
    )
    return ds


def main(args):
    station_list = [item.rsplit(".", 1)[0] for item in os.listdir(args.data_dir) if item.endswith(".csv")]
    meta = load_metadata(station_list)

    similarity = calc_similarity(meta)
    os.makedirs(args.output_dir, exist_ok=True)
    similarity_path = os.path.join(args.output_dir, f"similarity_{args.scale_dist}km_{args.scale_elev}m.nc")
    similarity.astype("float32").to_netcdf(similarity_path)
    logger.info(f"Saved similarity to {similarity_path}")

    data = load_raw_data(args.data_dir, meta.index.values)
    # some stations have no data, they have already been removed in load_raw_data
    meta = meta.loc[data["station"].values]

    similarity = similarity.sel(station1=meta.index.values, station2=meta.index.values)
    meta_simi = calc_meta_similarity(similarity)
    merged = merge_all_variables(data, meta_simi, meta_simi_th=args.meta_simi_th, data_simi_th=args.data_simi_th)
    # Save all metadata information in the NetCDF file
    merged = assign_meta_coords(merged, meta)
    data_path = os.path.join(args.output_dir, "data.nc")
    merged.astype(np.float32).to_netcdf(data_path)
    logger.info(f"Saved data to {data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output-dir", type=str, required=True, help="Directory of output similarity and data files"
    )
    parser.add_argument(
        "-d", "--data-dir", type=str, required=True, help="Directory of ISD csv files from the previous step"
    )
    parser.add_argument("--scale-dist", type=int, default=25, help="e-fold scale of distance similarity")
    parser.add_argument("--scale-elev", type=int, default=100, help="e-fold scale of elevation similarity")
    parser.add_argument("--meta-simi-th", type=float, default=0.35, help="Threshold of metadata similarity")
    parser.add_argument("--data-simi-th", type=float, default=0.7, help="Threshold of data similarity")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level (int >= 0)")
    args = parser.parse_args()
    configure_logging(args.verbose)
    main(args)
