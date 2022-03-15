import json
import numpy as np
import os
import urllib3
import time
import boto3
import csv
from distutils import util
import os.path
if os.path.isfile(".env"):
    # Running the code locally
    print('Found an environnement file, loading it!')
    from dotenv import load_dotenv
    load_dotenv()
    import src.s3_helper as s3_helper
else:
    # Running the code in AWS: all files are in the same folder
    import s3_helper

ISOS_BATCH_ENDPOINT = os.environ['isos_batch_endpoint']
BUCKET_BOAT_POSITION = os.environ['bucket_boat_position']
PREFIX_BOAT_POSITION = os.environ['prefix_boat_position']
BUCKET_ISOS = os.environ['bucket_isos']
PREFIX_ISOS_BATCH = os.environ['prefix_isos_batch']
# variavle to be removed once we stop having boat status for these legacy races
LEGACY_RACES = [429, 431, 436, 437, 443, 447, 456, 466]
TESTING = bool(util.strtobool(os.environ['testing']))


def get_candidate_boats():
    """Get a list of boats (i.e. combination of users and races) for whom there is data stored

    Returns:
        list: list of string containing the folders with data
    """
    list_users = s3_helper.list_folders(
        BUCKET_BOAT_POSITION, f"{PREFIX_BOAT_POSITION}/")
    print('Users found:')
    print(list_users)
    list_races = []
    for prefix_user in list_users:
        race = s3_helper.list_folders(BUCKET_BOAT_POSITION, prefix_user)
        if race != []:
            list_races.append(race)
    print(list_races)
    return list_races


def get_all_boats_position(candidate_folders):
    """Get boat position for a list of given users/races

    Args:
        candidate_folders (list): list of string: folder containing boat data

    Returns:
        list: list of dict
        {"user": "abc", "race": "123", "position": [x,y]}
    """
    boats = []
    for user_folders in candidate_folders:
        for prefix in user_folders:
            boat_position = get_single_boat_position(prefix)
            folder_split = prefix.split('/')
            if boat_position is not None and int(folder_split[2]) not in LEGACY_RACES:
                boats.append(
                    {"user": folder_split[1], "race": folder_split[2], "position": boat_position})
    return boats


def get_single_boat_position(prefix):
    """Get a boat position given an user and a race

    Args:
        prefix (string): root folder to look in (typically user/race)

    Returns:
        list: boat position as a float list [x,y]
    """
    race = prefix.split('/')[-2]
    print('-----------------------')
    print(f'Processing race {race}')
    boat_status_folder = s3_helper.get_latest_folder(
        BUCKET_BOAT_POSITION, prefix)
    if boat_status_folder is None:
        print(f'Could not locate the latest boat folder for race {race}')
        return None

    key = s3_helper.get_most_recent_file(
        BUCKET_BOAT_POSITION, prefix, folder=boat_status_folder)
    print(f'Latest boat file: {key}')

    s3 = boto3.resource('s3')
    obj = s3.Object(BUCKET_BOAT_POSITION, key)
    csv_str = obj.get()['Body'].read().decode('utf-8')

    reader = csv.reader(csv_str.split('\n'), delimiter=',')

    for row in reader:
        if row != []:
            data = row

    # lastCalcDate	lat	lon	heading	speed	twa	tws	twd	sail	distanceFromStart	distanceToEnd	credits	timeToNextCards	aground
    lat = float(data[1])
    lon = float(data[2])
    return [lon, lat]


def build_payload(filename, bucket):
    """Build the payload for sending the requests to the isochrones API

    Args:
        filename (string): where to get the wind data (folder)
        bucket (string): where to get the wind data (bucket)

    Returns:
        list: list of payloads (each one is a JSON encoded dict)
    """
    path_s3 = filename[:-4]
    candidate_folders = get_candidate_boats()
    all_boats = get_all_boats_position(candidate_folders)
    if all_boats is []:
        return None
    list_payload = []
    for boat in all_boats:
        user = boat['user']
        race = boat['race']
        path_s3_isos = f'{PREFIX_ISOS_BATCH}/{user}/{race}'
        with open(f'./race_data/{race}.json') as json_file:
            data_json = json.load(json_file)
        data_context = {'bucket_wind': bucket, 'path_s3_wind': path_s3,
                        'bucket_isos': BUCKET_ISOS, 'path_s3_isos': path_s3_isos,
                        'start_coords': boat['position']}
        final_data = dict(data_json, **data_context)
        list_payload.append(final_data)
    return list_payload


def trigger_isochrones(filename, bucket):
    """Call the isochrones API

    Args:
        filename (string): where to get the wind data (folder)
        bucket (string): where to get the wind data (bucket)

    Returns:
        None if the list of payload is None
    """
    http = urllib3.PoolManager()
    list_payload = build_payload(filename, bucket)
    if list_payload is None:
        return None
    for payload in list_payload:
        print('------------------------------')
        print('Sending to the isochrones API:')
        print(payload)
        ISOS_BATCH_ENDPOINT = os.environ['isos_batch_VG_endpoint']
        print(f'Using endpoint: {ISOS_BATCH_ENDPOINT}')

        if TESTING:
            print('-----------------------------------------------------')
            print('Testing only, not calling the isochrones API')
            print('-----------------------------------------------------')
        else:
            response = http.request("POST", ISOS_BATCH_ENDPOINT, body=json.dumps(payload).encode('utf-8'), headers={
                                    'Content-Type': 'application/json'})
            print(response.status)
            print(json.dumps(json.loads(response.data.decode('utf-8'))))


def lambda_handler(event, context):
    """This lambda functions is trigerred by a new batch of wind data (4 times a day). It updates the batch isochrones predictions for all recent users/races

    Args:
        event (JSON): S3 Put event (new specific .CSV file, e.g. 06/177.csv)
    """
    object_key = event['Records'][0]['s3']['object']['key']
    filename, _ = os.path.splitext(object_key)

    bucket = event['Records'][0]['s3']['bucket']['name']

    trigger_isochrones(filename, bucket)
    return 'Done!'
