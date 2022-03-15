import json
import boto3
import os
import urllib3
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

s3 = boto3.resource('s3')

ISOS_LIVE_ENDPOINT = os.environ['isos_live_VG_endpoint']
BUCKET_ISOS = os.environ['bucket_isos']
PREFIX_ISOS_LIVE = os.environ['prefix_isos_live']
PREFIX_ISOS_BATCH = os.environ['prefix_isos_batch']
BUCKET_WIND = os.environ['bucket_wind']
PREFIX_WIND_DATA = os.environ['prefix_wind_data']
TESTING = bool(util.strtobool(os.environ['testing']))


def get_latest_wind_data():
    wind_folder = s3_helper.get_latest_folder(
        BUCKET_WIND, f'{PREFIX_WIND_DATA}/', depth_limit=7, suffix='/00/')
    if wind_folder is None:
        print('Could not locate latest wind data, aborting...')
        return None

    if s3_helper.key_exists(BUCKET_WIND, f'{PREFIX_WIND_DATA}/{wind_folder}/18/006.csv'):
        return f'{PREFIX_WIND_DATA}/{wind_folder}/18'
    if s3_helper.key_exists(BUCKET_WIND, f'{PREFIX_WIND_DATA}/{wind_folder}/12/006.csv'):
        return f'{PREFIX_WIND_DATA}/{wind_folder}/12'
    if s3_helper.key_exists(BUCKET_WIND, f'{PREFIX_WIND_DATA}/{wind_folder}/06/006.csv'):
        return f'{PREFIX_WIND_DATA}/{wind_folder}/06'
    return f'{PREFIX_WIND_DATA}/{wind_folder}/00'


def build_isos_payload(start_coords, path_wind, waypoints, prefix, race):
    """Build the payload before sending the request to the isochrones API

    Args:
        start_coords (list): boat position as a float list [x,y]
        path_wind (string): where to find the wind (bucket/folder)
        waypoints (list): list of previously calculated waypoints (typically by the batch isochrones)
        prefix (string): where to store the isochrones results
        race (string): the race ID (in-game)

    Returns:
        JSON: API response
    """
    with open(f'./race_data/{race}.json') as json_file:
        data_json = json.load(json_file)
    data_context = {'bucket_wind': BUCKET_WIND, 'path_s3_wind': path_wind,
                    'bucket_isos': BUCKET_ISOS, 'path_s3_isos': prefix,
                    'start_coords': start_coords, 'waypoints': waypoints, }
    final_data = dict(data_json, **data_context)
    return json.dumps(final_data).encode('utf-8')


def trigger_isochrones(start_coords, path_wind, waypoints, prefix, race):
    """Call the isochornes API

    Args:
        start_coords (list): boat position as a float list [x,y]
        path_wind (string): where to find the wind (bucket/folder)
        waypoints (list): list of previously calculated waypoints (typically by the batch isochrones)
        prefix (string): where to store the isochrones results
        race (string): the race ID (in-game)
    """
    http = urllib3.PoolManager()
    payload = build_isos_payload(
        start_coords, path_wind, waypoints, prefix, race)
    print('Sending to the isochrones API:')
    print(payload)
    print(f'Using endpoint: {ISOS_LIVE_ENDPOINT}')

    if TESTING:
        print('-----------------------------------------------------')
        print('Testing only, not calling the isochrones API')
        print('-----------------------------------------------------')
    else:
        response = http.request("POST", ISOS_LIVE_ENDPOINT, body=payload, headers={
                                'Content-Type': 'application/json'})
        print(response.status)
        print(json.dumps(json.loads(response.data.decode('utf-8'))))


def lambda_handler(event, context):
    """This lambda function is trigerred by a new boat status. It will call the isochrones API with the updated boat position (and passing the previously claculated waypoints)

    Args:
        event (JSON): S3 Put event (new boat status)
    """
    source_bucket_name = event['Records'][0]['s3']['bucket']['name']
    object_key = event['Records'][0]['s3']['object']['key']
    object_key_list = object_key.split('/')
    user = object_key_list[1]
    race = object_key_list[2]

    prefix_in = f'{PREFIX_ISOS_BATCH}/{user}/{race}'
    prefix_out = f'{PREFIX_ISOS_LIVE}/{user}/{race}'

    waypoints_file = s3_helper.get_most_recent_file(BUCKET_ISOS, prefix_in)
    print(f'Latest waypoints file: {waypoints_file}')
    if waypoints_file is None:
        return 'Error'

    path_wind = get_latest_wind_data()
    if path_wind is None:
        return 'Error'
    print(f'Latest wind data: {path_wind}')

    content_object = s3.Object(BUCKET_ISOS, waypoints_file)
    file_content = content_object.get()['Body'].read().decode('utf-8')
    json_content = json.loads(file_content)
    waypoints = json_content['waypoints']

    obj = s3.Object(source_bucket_name, object_key)
    csv_str = obj.get()['Body'].read().decode('utf-8')

    reader = csv.reader(csv_str.split('\n'), delimiter=',')

    for row in reader:
        if row != []:
            data = row

    # lastCalcDate	lat	lon	heading	speed	twa	tws	twd	sail	distanceFromStart	distanceToEnd	credits	timeToNextCards	aground
    lat = float(data[1])
    lon = float(data[2])

    trigger_isochrones(
        [lon, lat], path_wind, waypoints, prefix_out, race)
    return 'Done!'
