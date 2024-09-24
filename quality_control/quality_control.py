import argparse
import logging
import numpy as np
import xarray as xr
from algo import (
    record_extreme,
    cluster,
    distributional_gap,
    neighbouring_stations,
    spike,
    persistence,
    cross_variable,
    refinement,
    diurnal_cycle,
    suspect_upgrade,
)
from algo.utils import merge_flags, Config, get_config, configure_logging


logger = logging.getLogger(__name__)
CONFIG = get_config()


def load_data(obs_path, rnl_path):
    """
    Load observation and reanalysis data
    The reanalysis data should be in the same format as observation data (interpolated to the same stations)
    """
    logger.info("Loading observation and reanalysis data")
    obs = xr.load_dataset(obs_path)
    # Please prepare the Reanalysis data in the same format as obs
    rnl = xr.load_dataset(rnl_path)
    if obs.sizes != rnl.sizes:
        raise ValueError("The sizes of obs and rnl are different")
    return obs, rnl


def cross_variable_check(obs):
    flag_cross = {}
    # Super-saturation check
    flag_cross["t"] = cross_variable.supersaturation(obs["t"], obs["td"])
    flag_cross["td"] = flag_cross["t"].copy()
    # Wind consistency check
    flag_cross["ws"] = cross_variable.wind_consistency(obs["ws"], obs["wd"])
    flag_cross["wd"] = flag_cross["ws"].copy()
    flag_ra = cross_variable.ra_consistency(obs[["ra1", "ra3", "ra6", "ra12", "ra24"]])
    for varname in ["ra1", "ra3", "ra6", "ra12", "ra24"]:
        flag_cross[varname] = flag_ra[varname].copy()
    return flag_cross


def quality_control(obs, rnl, f_similarity):
    varlist = obs.data_vars.keys()
    result = obs.copy()

    # Record extreme check
    flag_extreme = {}
    for varname in CONFIG["record"]:
        if varname not in varlist:
            continue
        logger.info(f"Record extreme check for {varname}...")
        flag_extreme[varname] = record_extreme.run(result[varname], varname)
        # For extreme value check, outliers are directly removed
        result[varname] = result[varname].where(flag_extreme != CONFIG["flag_error"])

    # Cluster deviation check
    flag_cluster = {}
    for varname in CONFIG["cluster"]:
        if varname not in varlist:
            continue
        logger.info(f"Cluster deviation check for {varname}...")
        flag_cluster[varname] = cluster.run(result[varname], rnl[varname], varname)

    # Distributional gap check
    flag_distribution = {}
    for varname in CONFIG["distribution"]:
        if varname not in varlist:
            continue
        logger.info(f"Distributional gap check for {varname}...")
        # Mask from cluster deviation check is used to exclude abnormal data in the following distributional gap check
        mask = flag_cluster[varname] == CONFIG["flag_normal"]
        flag_distribution[varname] = distributional_gap.run(result[varname], rnl[varname], varname, mask)

    # Neighbouring station check
    flag_neighbour = {}
    for varname in CONFIG["neighbouring"]:
        if varname not in varlist:
            continue
        logger.info(f"Neighbouring station check for {varname}...")
        flag_neighbour[varname] = neighbouring_stations.run(result[varname], f_similarity, varname)

    # Spike check
    flag_dis_neigh = {}
    flag_spike = {}
    for varname in CONFIG["spike"]:
        if varname not in varlist:
            continue
        logger.info(f"Spike check for {varname}...")
        # Merge flags from distributional gap and neighbouring station check for spike and persistence check
        flag_dis_neigh[varname] = merge_flags(
            [flag_distribution[varname], flag_neighbour[varname]], priority=["error", "suspect", "normal"]
        )
        flag_spike[varname] = spike.run(result[varname], flag_dis_neigh[varname], varname)

    # Persistence check
    flag_persistence = {}
    for varname in CONFIG["persistence"]:
        if varname not in varlist:
            continue
        logger.info(f"Persistence check for {varname}...")
        # Some variables are not checked by distributional gap or neighbouring station check
        if varname not in flag_dis_neigh:
            flag_dis_neigh_cur = xr.full_like(result[varname], CONFIG["flag_missing"], dtype=np.int8)
        else:
            flag_dis_neigh_cur = flag_dis_neigh[varname]
        flag_persistence[varname] = persistence.run(result[varname], flag_dis_neigh_cur, varname)

    # Cross variable check
    flag_cross = cross_variable_check(result)

    # Merge all flags
    flags = {}
    for varname in varlist:
        logger.info(f"Merging flags for {varname}...")
        merge_list = [
            item[varname]
            for item in [flag_extreme, flag_dis_neigh, flag_spike, flag_persistence, flag_cross]
            if varname in item
        ]
        flags[varname] = merge_flags(merge_list, priority=["normal", "suspect", "error"])

    # Flag refinement
    flags_refined = {}
    for varname in varlist:
        if varname not in CONFIG["refinement"].keys():
            flags_refined[varname] = flags[varname].copy()
        else:
            logger.info(f"Flag refinement for {varname}...")
            flags_refined[varname] = refinement.run(result[varname], flags[varname], varname)
        if varname == "t":
            flags_refined[varname] = diurnal_cycle.run(result[varname], flags_refined[varname])

    # Suspect upgrade
    flags_upgraded = {}
    for varname in varlist:
        logger.info(f"Upgrade suspect flags for {varname}...")
        flags_upgraded[varname] = suspect_upgrade.run(flags_refined[varname], obs[varname])

    return flags_upgraded


def main(args):
    obs, rnl = load_data(args.obs_path, args.rnl_path)
    flags = quality_control(obs, rnl, args.similarity_path)
    for varname in obs.data_vars.keys():
        obs[varname] = obs[varname].where(flags[varname] != CONFIG["flag_error"])
    obs.to_netcdf(args.output_path)
    logger.info(f"Quality control finished. The results are saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--obs-path", type=str, required=True, help="Data path of observation to be quality controlled"
    )
    parser.add_argument(
        "--rnl-path", type=str, required=True, help="Data path of reanalysis data to be used for quality control"
    )
    parser.add_argument("--similarity-path", type=str, required=True, help="Data path of similarity matrix")
    parser.add_argument("--output-path", type=str, required=True, help="Data path of output data")
    parser.add_argument(
        "--config-path", type=str, help="Path to the configuration file, default is config.yaml in the algo directory"
    )
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level (int >= 0)")
    parsed_args = parser.parse_args()
    configure_logging(parsed_args.verbose)
    Config(parsed_args.config_path)
    main(parsed_args)
