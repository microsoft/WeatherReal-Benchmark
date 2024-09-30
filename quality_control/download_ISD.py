"""
Download ISD data from NOAA
All available stations on the server will be downloaded
"""

import argparse
import logging
import multiprocessing
from functools import partial
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from algo.utils import configure_logging


URL_DATA = "https://www.ncei.noaa.gov/data/global-hourly/access/"

# Disable the logging from urllib3
logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)


def download_file(station, url, output_dir, overwrite=False):
    response = requests.get(url + f"{station}.csv", timeout=30)
    output_path = Path(output_dir) / f"{station}.csv"
    if output_path.exists() and not overwrite:
        return True
    if response.status_code == 200:
        with open(output_path, "wb") as file:
            file.write(response.content)
        return True
    return False


def download_ISD(year, station_list, output_dir, num_proc, overwrite=False):
    """
    Download the data from the source for the stations in station_list.
    """
    logger.info("Start downloading")
    output_dir = Path(output_dir) / str(year)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    url = URL_DATA + f"/{year}/"

    if num_proc == 1:
        for item in tqdm(station_list, desc="Downloading files"):
            download_file(item, url, output_dir, overwrite)
    else:
        func = partial(download_file, url=url, output_dir=output_dir, overwrite=overwrite)
        with multiprocessing.Pool(num_proc) as pool:
            results = list(tqdm(pool.imap(func, station_list), total=len(station_list), desc="Downloading files"))

        successful_downloads = sum(results)
        logger.info(f"Successfully downloaded {successful_downloads} out of {len(station_list)} files")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output-dir", type=str, required=True, help="Parent directory to save the downloaded data"
    )
    parser.add_argument("-y", "--year", type=int, default=pd.Timestamp.now().year, help="Year to download")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level (int >= 0)")
    parser.add_argument("--num-proc", type=int, default=16, help="Number of parallel processes")
    args = parser.parse_args()
    configure_logging(args.verbose)

    response = requests.get(URL_DATA + f"/{args.year}/", timeout=30)
    soup = BeautifulSoup(response.text, "html.parser")
    file_list = [link.get("href") for link in soup.find_all("a") if link.get("href").endswith(".csv")]
    station_list = [item.split(".")[0] for item in file_list]
    logger.info(f"Found {len(station_list)} stations for year {args.year}")
    download_ISD(args.year, station_list, args.output_dir, args.num_proc, args.overwrite)


if __name__ == "__main__":
    main()
