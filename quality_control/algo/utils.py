import logging
import os
import numpy as np
import xarray as xr
import yaml


logger = logging.getLogger(__name__)


def config_loader(config_path):
    global CONFIG
    with open(config_path, "r") as file:
        CONFIG = yaml.safe_load(file)
    return CONFIG


CONFIG = config_loader(os.path.dirname(os.path.realpath(__file__)) + "/config.yaml")


def intra_station_check(
    *dataarrays,
    qc_func=lambda da: da,
    input_core_dims=[["time"]],
    output_core_dims=[["time"]],
    kwargs=dict(),
):
    """
    A wrapper function to apply the quality control functions to each station in a DataArray
    Multiprocessing is implemented by Dask
    Parameters:
    dataarrays: DataArrays to be checked, with more auxiliary DataArrays if needed
    qc_func: quality control function to be applied
    input_core_dims: core dimensions (to be remained for the function) of each DataArray
    output_core_dims: core dimensions (to be remained from the function) of each function result
    kwargs: keyword arguments for the quality control function
    Return:
    flag: DataArray with the same shape as the first input DataArray
    """
    dataarrays_chunked = []
    for item in dataarrays:
        if isinstance(item, xr.DataArray):
            dataarrays_chunked.append(
                item.chunk({k: 100 if k == "station" else -1 for k in item.dims})
            )
        else:
            dataarrays_chunked.append(item)
    flag = xr.apply_ufunc(
        qc_func,
        *dataarrays_chunked,
        input_core_dims=input_core_dims,
        output_core_dims=output_core_dims,
        kwargs=kwargs,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.int8],
    ).compute(scheduler="processes")
    return flag


def merge_flags(flags, priority=["normal", "suspect", "error"]):
    """
    Merge flags from different quality control functions in the order of priority
    Prior flags will be overwritten by subsequent flags
    """
    ret = xr.full_like(flags[0], CONFIG["flag_missing"], dtype=np.int8)
    for flag_type in priority:
        for item in flags:
            ret = xr.where(item == CONFIG[f"flag_{flag_type}"], CONFIG[f"flag_{flag_type}"], ret)
    return ret


def quality_control_statistics(data, flag):
    num_valid = data.notnull().sum().item()
    num_normal = (flag == CONFIG["flag_normal"]).sum().item()
    num_suspect = (flag == CONFIG["flag_suspect"]).sum().item()
    num_error = (flag == CONFIG["flag_error"]).sum().item()
    num_checked = num_normal + num_suspect + num_error
    logger.debug(f"{num_valid / data.size:.5%} of the data are valid")
    logger.debug(f"{num_checked / num_valid:.5%} of the valid data are checked")
    logger.debug(
        f"{num_normal / num_checked:.5%}/{num_suspect / num_checked:.5%}/{num_error / num_checked:.5%} " +
        "of the checked data are flagged as normal/suspect/erroneous"
    )
    return num_valid, num_normal, num_suspect, num_error


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
