import math
import sys
import pandas as pd
import boto3
import shutil
import tempfile
import urllib.request
import re
import os
import ast
from distutils import util
import os.path
if os.path.isfile(".env"):
    print('Found an environnement file, loading it!')
    from dotenv import load_dotenv
    load_dotenv()


s3 = boto3.client('s3')
BUCKET_NAME = os.environ['bucket_wind_data_name']
TESTING = bool(util.strtobool(os.environ['testing']))


def parse_wnd(wnd_file_path):
    """Parse a proprietary-formatted .WND file into a dictionary

    Args:
        wnd_file_path (string): .WND file path

    Returns:
        list of dictionary: parsed data
        [{'latitude': 180, 'longitude': 0, 'u': 3.4, 'v': 4.5},{...}]
    """
    longitude = -180
    latitude = 90
    byte_couple_count = 0
    wind_values = []
    wind_value_couple = []
    with open(wnd_file_path, "rb") as f:
        byte = f.read(1)
        while byte:
            byte = f.read(1)
            if byte_couple_count == 2:
                # Store data
                wind_values.append({'latitude': latitude, 'longitude': longitude,
                                    'u': wind_value_couple[0], 'v': wind_value_couple[1]})

                # Reset local variables
                latitude, longitude = update_lat_lon(latitude, longitude)
                byte_couple_count = 0
                wind_value_couple = []

            raw_value = int.from_bytes(
                byte, byteorder=sys.byteorder, signed=True)
            # Ukmph = sign(Ub) * sqr(Ub / 8)
            wind_value_couple.append(
                math.copysign(1, raw_value) * raw_value**2/8)
            byte_couple_count += 1

        # Store last couple
        wind_values.append({'latitude': latitude, 'longitude': longitude,
                            'u': wind_value_couple[0], 'v': wind_value_couple[1]})
    return wind_values


def update_lat_lon(latitude, longitude):
    """Custom logic to iterate through lat/lon linked to the .wnd file structure

    Args:
        latitude (int): latitude
        longitude (int): longitude

    Returns:
        int, int: updated lat/lon
    """
    if longitude < 179:
        longitude += 1
    elif longitude == 179:
        longitude = -180
        latitude -= 1
    else:
        print(f'Error with the longitude value: {longitude}')
    return latitude, longitude


def lambda_handler(event, context):
    """This lambda function takes a SNS message as an input. It downloads the file mentioned in the SNS message, and if it is a .WND file it will parse it and store it.

    Args:
        event (JSON): SNS message data
    """
    message = event['Records'][0]['Sns']['Message']
    if isinstance(message, str):
        print("payload is a string! Converting into a dict...")
        message_dict = ast.literal_eval(message)
    else:
        message_dict = message

    source_bucket_name = message_dict['Records'][0]['s3']['bucket']['name']
    object_key = message_dict['Records'][0]['s3']['object']['key']
    print(f'File received: {object_key} from bucket: {source_bucket_name}')

    # Store the file locally
    with urllib.request.urlopen('https://' + source_bucket_name + '/' + object_key) as response:
        with tempfile.NamedTemporaryFile(delete=False) as in_tmp_file:
            shutil.copyfileobj(response, in_tmp_file)

    # Upload the .file as-is to S3
    print(f'Uploading file {object_key} to bucket {BUCKET_NAME}')
    if TESTING:
        print('-----------------------------------------------------')
        print('Testing only, not storing anything to S3')
        print('-----------------------------------------------------')
    else:
        s3.upload_file(in_tmp_file.name, BUCKET_NAME, object_key)
        print(
            f'Succesfully uploaded input file in bucket {BUCKET_NAME}, file name {object_key}')

    # Nothing else to do if we're dealing with something else than WND
    if(object_key[-3:] != 'wnd'):
        print(f'File {object_key} is not a WND file!')
        return "Not a WND file, aborting..."

    # Keep only the 20200901/06/177 in winds/live/20200901/06/177.wnd
    regex = r"\d"
    matches = re.finditer(regex, object_key, re.MULTILINE)
    for _, match in enumerate(matches, start=1):
        target_obj_key = object_key[match.start():-4]
        break

    # Parse the .wnd file
    print('Converting file as a CSV...')
    wind_values = parse_wnd(in_tmp_file.name)
    df = pd.DataFrame(wind_values)
    df = df.set_index('latitude')

    # Save the parsed file (as CSV) locally
    with tempfile.NamedTemporaryFile(delete=False) as out_tmp_file:
        df.to_csv(out_tmp_file.name)

    # Upload the parsed file to S3
    target_name = os.path.join('winds', 'csv', target_obj_key + '.csv')
    print(f'Uploading file {target_name} to bucket {BUCKET_NAME}')
    if TESTING:
        print('-----------------------------------------------------')
        print('Testing only, not storing anything to S3')
        print('-----------------------------------------------------')
    else:
        s3.upload_file(out_tmp_file.name, BUCKET_NAME, target_name)
        print(
            f'Succesfully uploaded CSV file in bucket {BUCKET_NAME}, file name {target_name}')
    return 'Success!'
