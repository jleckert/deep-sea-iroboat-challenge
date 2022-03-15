"""
A simple script to test the net_cdf_wrapper.py methods and class
"""

from src.net_cdf.net_cdf_wrapper import *
import json
import pandas as pd
from src.common.log import logger


file_name = 'tauxy20110105'
file_name = 'air.sig995.2012'
nc_f = f'data/raw/{file_name}.nc'
out_csv_path = f'data/pre-processed/{file_name}.csv'

# Extract metadata about the .nc file and print it
nc_dump = parse_nc_file(nc_f)
nc_dump.print_all()

# Extract the actual data: variables as a function of the dimensions
data = extract_vars_data(nc_f)

# Store the data as a huge CSV
df = pd.DataFrame.from_dict(data)
df = df.set_index('time')
df.to_csv(out_csv_path)
logger.info(f'Stored data in {out_csv_path}')
