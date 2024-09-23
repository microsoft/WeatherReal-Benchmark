The quality control code in this directory is written in Python and is used by WeatherReal for downloading, post-processing, and quality control of ISD data. However, its modules can also be used for quality control of observation data from other sources. It includes four launch scripts:

	1.	`download_ISD.py`, used to download ISD data from the NCEI server;
	2.	`raw_ISD_to_hourly.py`, used to convert ISD data into hourly data;
	3.	`station_merging.py`, used to merge data from duplicate stations;
	4.	`quality_control.py`, used to perform quality control on hourly data.

For the specific workflow, please refer to the WeatherReal paper.