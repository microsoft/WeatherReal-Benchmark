import logging
import os
import numpy as np
import xarray as xr
import yaml
try:
    import xesmf as xe
    XESMF_AVAILABLE = True
except ImportError:
    XESMF_AVAILABLE = False


logger = logging.getLogger(__name__)


def configure_logging(verbose=1):
    verbose_levels = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
        3: logging.NOTSET
    }
    if verbose not in verbose_levels:
        verbose = 1
    root_logger = logging.getLogger()
    root_logger.setLevel(verbose_levels[verbose])
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s] [PID=%(process)d] "
        "[%(levelname)s %(filename)s:%(lineno)d] %(message)s"))
    handler.setLevel(verbose_levels[verbose])
    root_logger.addHandler(handler)


class Config:
    _instance = None

    def __new__(cls, config_path=None):
        if cls._instance is None:
            if config_path is None:
                config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance.__init__(config_path)
        return cls._instance

    def __init__(self, config_path):
        if not hasattr(self, 'config'):
            self._load_config(config_path)

    def _load_config(self, config_path):
        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

    def get(self, key):
        if key not in self.config:
            raise KeyError(f"Key '{key}' not found in configuration")
        return self.config[key]


def get_config(config_path=None):
    return Config(config_path).config


def intra_station_check(
    *dataarrays,
    qc_func=lambda da: da,
    input_core_dims=None,
    output_core_dims=None,
    kwargs=None,
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
    if input_core_dims is None:
        input_core_dims = [["time"]]
    if output_core_dims is None:
        output_core_dims = [["time"]]
    if kwargs is None:
        kwargs = {}
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


CONFIG = get_config()


def merge_flags(flags, priority=None):
    """
    Merge flags from different quality control functions in the order of priority
    Prior flags will be overwritten by subsequent flags
    """
    ret = xr.full_like(flags[0], CONFIG["flag_missing"], dtype=np.int8)
    if priority is None:
        priority = ["normal", "suspect", "error"]
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
        "%s/%s/%s of the checked data are flagged as normal/suspect/erroneous",
        f"{num_normal / num_checked:.5%}",
        f"{num_suspect / num_checked:.5%}",
        f"{num_error / num_checked:.5%}"
    )
    return num_valid, num_normal, num_suspect, num_error


def interpolate_from_grid_to_station(grid, station):
    """
    Interpolate data from a grid to stations
    """
    if not XESMF_AVAILABLE:
        raise ImportError("xesmf is required for interpolation.")

    if "latitude" in grid.dims:
        grid = grid.rename({"latitude": "lat", "longitude": "lon"})
    regridder = xe.Regridder(grid, station, "bilinear", locstream_out=True, periodic=True)
    grid = regridder(grid, keep_attrs=True)
    return grid
