"""
This script is used for parsing ISD files and converting them to hourly data
1. Parse corresponding columns of each variable
2. Aggregate / reorganize columns if needed
3. Simple unit conversion
4. Aggregate to hourly data. Rows that represent same hour are merged by rules
5. No quality control is applied except removing records with original erroneous flags
The outputs are still saved as csv files
"""

import argparse
import multiprocessing
import os
from functools import partial

import pandas as pd
from tqdm import tqdm


# Quality code for erroneous values flagged by ISD
# To collect as much data as possible and avoid some values flagged by unknown codes being excluded,
# We choose the strategy "only values marked with known error tags will be rejected"
# instead of "only values marked with known correct tags will be accepted"
ERRONEOUS_FLAGS = ["3", "7"]


def parse_temperature_col(data):
    """
    Process temperature and dew point temperature columns
    TMP/DEW column format: -0100,1
    Steps:
    1. Set values flagged as erroneous/missing to NaN
    2. Convert to float in Celsius
    """
    if "TMP" in data.columns and data["TMP"].notnull().any():
        data[["t", "t_qc"]] = data["TMP"].str.split(",", expand=True)
        data["t"] = data["t"].where(data["t"] != "+9999", pd.NA)
        data["t"] = data["t"].where(~data["t_qc"].isin(ERRONEOUS_FLAGS), pd.NA)
        # Scaling factor: 10
        data["t"] = data["t"].astype("Float32") / 10
    if "DEW" in data.columns and data["DEW"].notnull().any():
        data[["td", "td_qc"]] = data["DEW"].str.split(",", expand=True)
        data["td"] = data["td"].where(data["td"] != "+9999", pd.NA).astype("Float32") / 10
        data["td"] = data["td"].where(~data["td_qc"].isin(ERRONEOUS_FLAGS), pd.NA)
    data = data.drop(columns=["TMP", "DEW", "t_qc", "td_qc"], errors="ignore")
    return data


def parse_wind_col(data):
    """
    Process wind speed and direction column
    WND column format: 267,1,N,0142,1
    N indicates normal (other values include Beaufort, Calm, etc.). Not used currently
    Steps:
    1. Set values flagged as erroneous/missing to NaN
       Note that if one of ws or wd is missing, both are set to NaN
       Exception: If ws is 0 and wd is missing, wd is set to 0 (calm)
    2. Convert wd to integer and ws to float in m/s
    """
    if "WND" in data.columns and data["WND"].notnull().any():
        data[["wd", "wd_qc", "wt", "ws", "ws_qc"]] = data["WND"].str.split(",", expand=True)
        # First, set wd to 0 if ws is valid 0 and wd is missing
        calm = (data["ws"] == "0000") & (~data["ws_qc"].isin(ERRONEOUS_FLAGS)) & (data["wd"] == "999")
        data.loc[calm, "wd"] = "000"
        data.loc[calm, "wd_qc"] = "1"
        # After that, if one of ws or wd is missing/erroneous, both are set to NaN
        non_missing = (data["wd"] != "999") & (data["ws"] != "9999")
        non_error = (~data["wd_qc"].isin(ERRONEOUS_FLAGS)) & (~data["ws_qc"].isin(ERRONEOUS_FLAGS))
        valid = non_missing & non_error
        data["wd"] = data["wd"].where(valid, pd.NA)
        data["ws"] = data["ws"].where(valid, pd.NA)
        data["wd"] = data["wd"].astype("Int16")
        # Scaling factor: 10
        data["ws"] = data["ws"].astype("Float32") / 10
    data = data.drop(columns=["WND", "wd_qc", "wt", "ws_qc"], errors="ignore")
    return data


def parse_cloud_col(data):
    """
    Process total cloud cover column
    All known columns including GA1-6, GD1-6, GF1 and GG1-6 are parsed and Maximum value of them is selected
    1. GA1-6 column format: 07,1,+00800,1,06,1
       The 1st and 2nd items are c and its quality
    2. GD1-6 column format: 3,99,1,+05182,9,9
       The 1st item is cloud cover in 0-4 and is converted to octas by multiplying 2
       The 2st and 3nd items are c in octas and its quality
    3. GF1 column format: 07,99,1,07,1,99,9,01000,1,99,9,99,9
       The 1st and 3rd items are total coverage and its quality
    4. GG1-6 column format: 01,1,01200,1,06,1,99,9
       The 1st and 2nd items are c and its quality
    Cloud/sky-condition related data is very complex and worth further investigation
    Steps:
    1. Set values flagged as erroneous/missing to NaN
    2. Select the maximum value of all columns
    """
    num = 0
    for group in ["GA", "GG"]:
        for col in [f"{group}{i}" for i in range(1, 7)]:
            if col in data.columns and data[col].notnull().any():
                data[[f"c{num}", "c_qc", "remain"]] = data[col].str.split(",", n=2, expand=True)
                # 99 will be removed later
                data[f"c{num}"] = data[f"c{num}"].where(~data["c_qc"].isin(ERRONEOUS_FLAGS), pd.NA)
                data[f"c{num}"] = data[f"c{num}"].astype("Int16")
                num += 1
            else:
                break
    for col in [f"GD{i}" for i in range(1, 7)]:
        if col in data.columns and data[col].notnull().any():
            data[[f"c{num}", f"c{num+1}", "c_qc", "remain"]] = data[col].str.split(",", n=3, expand=True)
            c_cols = [f"c{num}", f"c{num+1}"]
            data[c_cols] = data[c_cols].where(~data["c_qc"].isin(ERRONEOUS_FLAGS), pd.NA)
            data[c_cols] = data[c_cols].astype("Int16")
            # The first item is 5-level cloud cover and is converted to octas by multiplying 2
            data[f"c{num}"] = data[f"c{num}"] * 2
            num += 2
        else:
            break
    if "GF1" in data.columns and data["GF1"].notnull().any():
        data[[f"c{num}", "opa", "c_qc", "remain"]] = data["GF1"].str.split(",", n=3, expand=True)
        data[f"c{num}"] = data[f"c{num}"].where(~data["c_qc"].isin(ERRONEOUS_FLAGS), pd.NA)
        data[f"c{num}"] = data[f"c{num}"].astype("Int16")
        num += 1
    c_cols = [f"c{i}" for i in range(num)]
    # Mask all values larger than 8 to NaN to avoid overwriting the correct values
    data[c_cols] = data[c_cols].where(data[c_cols] <= 8, pd.NA)
    # Maximum value of all columns is selected to represent the total cloud cover
    data["c"] = data[c_cols].max(axis=1)
    data = data.drop(
        columns=[
            "GF1",
            *[f"GA{i}" for i in range(1, 7)],
            *[f"GG{i}" for i in range(1, 7)],
            *[f"GD{i}" for i in range(1, 7)],
            *[f"c{i}" for i in range(num)],
            "c_5",
            "opa",
            "c_qc",
            "remain",
        ],
        errors="ignore",
    )
    return data


def parse_surface_pressure_col(data):
    """
    Process surface pressure (station-level pressure) column
    Currently MA1 column is used. Column format: 99999,9,09713,1
    The 3rd and 4th items are station pressure and its quality
    The 1st and 2nd items are altimeter setting and its quality which are not used currently
    Steps:
    1. Set values flagged as erroneous/missing to NaN
    2. Convert to float in hPa
    """
    if "MA1" in data.columns and data["MA1"].notnull().any():
        data[["MA1_remain", "sp", "sp_qc"]] = data["MA1"].str.rsplit(",", n=2, expand=True)
        data["sp"] = data["sp"].where(data["sp"] != "99999", pd.NA)
        data["sp"] = data["sp"].where(~data["sp_qc"].isin(ERRONEOUS_FLAGS), pd.NA)
        # Scaling factor: 10
        data["sp"] = data["sp"].astype("Float32") / 10
    data = data.drop(columns=["MA1", "MA1_remain", "sp_qc"], errors="ignore")
    return data


def parse_sea_level_pressure_col(data):
    """
    Process mean sea level pressure column
    MSL Column format: 09725,1
    Steps:
    1. Set values flagged as erroneous/missing to NaN
    2. Convert to float in hPa
    """
    if "SLP" in data.columns and data["SLP"].notnull().any():
        data[["msl", "msl_qc"]] = data["SLP"].str.rsplit(",", expand=True)
        data["msl"] = data["msl"].where(data["msl"] != "99999", pd.NA)
        data["msl"] = data["msl"].where(~data["msl_qc"].isin(ERRONEOUS_FLAGS), pd.NA)
        # Scaling factor: 10
        data["msl"] = data["msl"].astype("Float32") / 10
    data = data.drop(columns=["SLP", "msl_qc"], errors="ignore")
    return data


def parse_single_precipitation_col(data, col):
    """
    Parse one of the precipitation columns AA1-4
    """
    if data[col].isnull().all():
        return pd.DataFrame()
    datacol = data[[col]].copy()
    # Split the column to get the period first
    datacol[["period", f"{col}_remain"]] = datacol[col].str.split(",", n=1, expand=True)
    # Remove weird periods to avoid unexpected errors
    datacol = datacol[datacol["period"].isin(["01", "03", "06", "12", "24"])]
    if len(datacol) == 0:
        return pd.DataFrame()
    # Set the period as index and unstack so that different periods are converted to different columns
    datacol = datacol.set_index("period", append=True)[f"{col}_remain"]
    datacol = datacol.unstack("period")
    # Rename the columns according to the period, e.g., 03 -> ra3
    datacol.columns = [f"ra{item.lstrip('0')}" for item in datacol.columns]
    # Further split the remaining sections
    for var in datacol.columns:
        datacol[[var, f"{var}_cond", f"{var}_qc"]] = datacol[var].str.split(",", expand=True)
    return datacol


def parse_precipitation_col(data):
    """
    Process precipitation columns
    Currently AA1-4 columns are used. Column format: 24,0073,3,1
    The items are period, depth, condition, quality. Condition is not used currently
    It is more complex than other variables as values during different periods are stored in same columns
    It is needed to separate them to different columns
    Steps:
    1. Separate and recombine columns by period
    2. Set values flagged as erroneous/missing to NaN
    3. Convert to float in mm
    """
    for col in ["AA1", "AA2", "AA3", "AA4"]:
        if col in data.columns:
            datacol = parse_single_precipitation_col(data, col)
            # Same variable (e.g., ra24) may be stored in different original columns
            # Combine_first so that same variables can be merged to the same columns
            data = data.combine_first(datacol)
        else:
            # Assuming that the remaining columns are also not present
            break
    # Quality status treated as valid records. 3/7 indicates erroneous value
    for col in [item for item in data.columns if item.startswith("ra") and item[2:].isdigit()]:
        data[col] = data[col].where(data[col] != "9999", pd.NA)
        data[col] = data[col].where(~data[f"{col}_qc"].isin(ERRONEOUS_FLAGS), pd.NA)
        data[col] = data[col].astype("Float32") / 10
        data = data.drop(columns=[f"{col}_cond", f"{col}_qc"])
    data = data.drop(columns=["AA1", "AA2", "AA3", "AA4"], errors="ignore")
    return data


def parse_single_file(fpath, fpath_last_year):
    """
    Parse columns of each variable in a single ISD file
    """
    # Gxn for cloud cover, MA1 for surface pressure, AAn for precipitation
    cols_var = [
        "TMP",
        "DEW",
        "WND",
        "SLP",
        "MA1",
        "AA1",
        "AA2",
        "AA3",
        "AA4",
        "GF1",
        *[f"GA{i}" for i in range(1, 7)],
        *[f"GD{i}" for i in range(1, 7)],
        *[f"GG{i}" for i in range(1, 7)],
    ]
    cols = ["DATE"] + list(cols_var)

    def _load_csv(fpath):
        return pd.read_csv(fpath, parse_dates=["DATE"], usecols=lambda c: c in set(cols), low_memory=False)

    data = _load_csv(fpath)
    if fpath_last_year is not None and os.path.exists(fpath_last_year):
        data_last_year = _load_csv(fpath_last_year)
        # Load the last day of the last year for better hourly aggregation
        data_last_year = data_last_year.loc[
            (data_last_year["DATE"].dt.month == 12) & (data_last_year["DATE"].dt.day == 31)
        ]
        data = pd.concat([data_last_year, data], ignore_index=True)
    data = data[[item for item in cols if item in data.columns]]

    data = parse_temperature_col(data)
    data = parse_wind_col(data)
    data = parse_cloud_col(data)
    data = parse_surface_pressure_col(data)
    data = parse_sea_level_pressure_col(data)
    data = parse_precipitation_col(data)

    data = data.rename(columns={"DATE": "time"})
    value_cols = [col for col in data.columns if col != "time"]
    # drop all-NaN rows
    data = data[["time"] + value_cols].sort_values("time").dropna(how="all", subset=value_cols)
    return data


def aggregate_to_hourly(data):
    """
    Aggregate rows that represent same hour to one row
    Order the rows from same hour by difference from the top of the hour,
    then use ffill at each hour to get the nearest valid values for each variable
    Specifically, For t/td, avoid combining two records from different rows together
    """
    data["hour"] = data["time"].dt.round("h")
    # Sort data by difference from the top of the hour so that bfill can be applied
    # to give priority to the closer records
    data["hour_dist"] = (data["time"] - data["hour"]).dt.total_seconds().abs() // 60
    data = data.sort_values(["hour", "hour_dist"])

    if data["hour"].duplicated().any():
        # Consruct a new column of (t, td) tuples. Values are not NaN only when both of them are valid
        data["t_td"] = data.apply(
            lambda row: (row["t"], row["td"]) if row[["t", "td"]].notnull().all() else pd.NA, axis=1
        )
        # For same hour, fill NaNs at the first row in the order of difference from the top of the hour
        data = data.groupby("hour").apply(lambda df: df.bfill().iloc[0], include_groups=False)

        # 1st priority: for hours that has both valid t and td originally (decided by t_td),
        # fill values to t_new and td_new
        # Specifically, for corner cases that all t_td is NaN, we need to convert pd.NA to (pd.NA, pd.NA)
        # so that to_list() will not raise an error
        data["t_td"] = data["t_td"].apply(lambda item: (pd.NA, pd.NA) if pd.isna(item) else item)
        data[["t_new", "td_new"]] = pd.DataFrame(data["t_td"].to_list(), index=data.index)
        # 2nd priority: Remaining hours can only provide at most one of t and td. Try to fill t first
        rows_to_fill = data[["t_new", "td_new"]].isnull().all(axis=1)
        data.loc[rows_to_fill, "t_new"] = data.loc[rows_to_fill, "t"]
        # 3nd priority: Remaining hours has no t during time window. Try to fill td
        rows_to_fill = data[["t_new", "td_new"]].isnull().all(axis=1)
        data.loc[rows_to_fill, "td_new"] = data.loc[rows_to_fill, "td"]

        data = data.drop(columns=["t", "td", "t_td"]).rename(columns={"t_new": "t", "td_new": "td"})

    data = data.reset_index(drop=True)
    data["time"] = data["time"].dt.round("h")
    return data


def post_process(data, year):
    """
    Some post-processing steps after aggregation
    """
    data = data.set_index("time")
    sorted_ra_columns = sorted([col for col in data.columns if col.startswith("ra")], key=lambda x: int(x[2:]))
    other_columns = [item for item in ["t", "td", "ws", "wd", "sp", "msl", "c"] if item in data.columns]
    data = data[other_columns + sorted_ra_columns]
    data = data[f"{year}-01-01":f"{year}-12-31"]
    return data


def pipeline(input_path, output_dir, year, overwrite=True):
    """
    The pipeline function for processing a single ISD file
    """
    output_path = os.path.join(output_dir, os.path.basename(input_path))
    if not overwrite and os.path.exists(output_path):
        return
    input_dir = os.path.dirname(input_path)
    if input_dir.endswith(str(year)):
        input_path_last_year = os.path.join(input_dir[:-4] + str(year - 1), os.path.basename(input_path))
    else:
        input_path_last_year = None
    data = parse_single_file(input_path, input_path_last_year)
    data = aggregate_to_hourly(data)
    data = post_process(data, year)
    data.astype("Float32").to_csv(output_path, float_format="%.1f")


def main(args):
    output_dir = os.path.join(args.output_dir, str(args.year))
    os.makedirs(output_dir, exist_ok=True)
    input_dir = os.path.join(args.input_dir, str(args.year))
    input_list = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".csv")]
    if args.num_proc == 1:
        for fpath in tqdm(input_list):
            pipeline(fpath, output_dir=output_dir, year=args.year, overwrite=args.overwrite)
    else:
        func = partial(pipeline, output_dir=output_dir, year=args.year, overwrite=args.overwrite)
        with multiprocessing.Pool(args.num_proc) as pool:
            for _ in tqdm(pool.imap(func, input_list), total=len(input_list), desc="Processing files"):
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", type=str, required=True, help="Directory of ISD csv files")
    parser.add_argument("-o", "--output-dir", type=str, required=True, help="Directory of output NetCDF files")
    parser.add_argument("-y", "--year", type=int, required=True, help="Target year")
    parser.add_argument("--num-proc", type=int, default=16, help="Number of parallel processes")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parsed_args = parser.parse_args()
    main(parsed_args)
