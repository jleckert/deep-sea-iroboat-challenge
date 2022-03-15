from collections import defaultdict
import os
import time
from base64 import b64encode
import json
from distutils import util
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
import urllib3
import boto3
from botocore.config import Config
import s3_helper
from log import logger

secret = os.environ["VR_SECRET_KEY"]
api_user_id = os.environ["USER_ID"]
race_id_list = os.environ["RACE_ID_LIST"].split(',')
base_url = os.environ["VR_AWS_API_BASE_URL"]
BUCKET_BOAT_POSITION = os.environ['BUCKET_BOAT_POSITION']
TESTING = bool(util.strtobool(os.environ['TESTING']))
RUNNING_LOCALLY = False
if os.path.isfile('.env_sample'):
    RUNNING_LOCALLY = True
json_followed_path = 'followed.json'
if RUNNING_LOCALLY:
    json_followed_path = f'./src/{json_followed_path}'
with open(json_followed_path) as json_file:
    FOLLOWED = json.load(json_file)['data']


def generate_api_key():
    now = str(round(time.time() * 1000))

    now_bytes = bytes(now, 'utf-16LE')
    secret_bytes = bytes(secret, 'utf-16LE')
    key = secret_bytes[:32]
    iv = secret_bytes[32:48]

    cipher = AES.new(key, AES.MODE_CBC, iv)
    ciphertext = cipher.encrypt(pad(now_bytes, AES.block_size))

    return b64encode(ciphertext).decode("utf-8")


def call_api(http, route, encoded_data):
    api_key = generate_api_key()
    r = http.request(
        'POST',
        base_url+route,
        body=encoded_data,
        headers={"Content-Type": 'application/json', 'x-api-key': api_key}
    )
    return json.loads(r.data.decode('utf-8'))


def get_fleet(http, race_id):
    # Some races may be using a leg number >1
    fleet_list = defaultdict()
    for leg_num in range(1, 10):
        data = {
            "user_id": api_user_id,
            "race_id": race_id,
            "leg_num": leg_num,
            "filter": "followed",
            "followed": FOLLOWED
        }
        encoded_data = json.dumps(data).encode('utf-8')
        fleet = call_api(http, 'getfleet', encoded_data)
        if fleet['rc'] != "ok":
            logger.error("Error while getting fleet, aborting")
            return None
        if fleet['res'] != []:
            fleet_list[leg_num] = fleet
    return fleet_list


def get_race_info(http, race_id, max_leg=10):
    # Some races may be using a leg number >1
    races_info = {}
    for leg_num in range(1, max_leg):
        data = {
            "user_id": api_user_id,
            "race_id": race_id,
            "leg_num": leg_num,
            "infos": "leg"
        }
        # Other options for the key "infos" are: "engine,leg,bs,ba,track"
        encoded_data = json.dumps(data).encode('utf-8')
        race_info = call_api(
            http, 'getboatinfos', encoded_data)
        if race_info['rc'] != "ok":
            logger.error("Error while getting fleet, aborting")
            return None
        if race_info['res']['leg'] is not None:
            races_info[f'leg_{leg_num}'] = race_info['res']['leg']
    return races_info


def lambda_handler(event, context):
    """This lambda function calls the VR API for getting races and fleet data.
    It then passes on the data to a Kinesis stream

    Returns:
        string: Success or not
    """
    http = urllib3.PoolManager()
    s3_resource = boto3.resource('s3')
    kinesis_client = boto3.client(
        'kinesis', config=Config(region_name='eu-west-1'))

    for race_id in race_id_list:
        logger.info(f"Processing race {race_id}")

        if not race_files_exists(race_id):
            logger.debug("Calling VR API - Get boat infos route")
            race_info = get_race_info(http, race_id)
            if race_info is None:
                logger.error(f"Could not store race info of race {race_id}")
            else:
                write_race_to_s3(s3_resource, race_info, race_id)
        else:
            logger.info('Race info files already exist, not calling VR API')

        logger.debug("Calling VR API - Get fleet route")
        fleet_list = get_fleet(http, race_id)

        if fleet_list is None:
            logger.error(f"Skipping race {race_id}")
            continue

        for leg, fleet in fleet_list.items():
            logger.info(f'Leg {leg}')
            n_records = len(fleet['res'])
            logger.debug(f"Got {n_records} records!")

            for record in fleet['res']:
                if('tws' in record):
                    record['leg'] = leg
                    record['raceId'] = race_id
                    put_record_kinesis(kinesis_client, record)

            logger.info(f'Successfully processed {n_records} records.')
    return 'Success!'


def race_files_exists(race_id):
    key = f'race_info/race_{race_id}.json'
    exists = s3_helper.key_exists(BUCKET_BOAT_POSITION, key)
    if not exists:
        return False
    return True


def write_race_to_s3(s3_resource, data, race_id):
    key = f'race_info/race_{race_id}.json'
    if TESTING:
        logger.debug(
            '-----------------------------------------------------')
        logger.debug('Testing only, not storing anything to S3')
        logger.debug(
            '-----------------------------------------------------')
    else:
        s3_resource.Object(BUCKET_BOAT_POSITION, key).put(
            Body=json.dumps(data))


def put_record_kinesis(client, data):
    response = client.put_record(
        StreamName='vr-test',
        Data=json.dumps(data),
        PartitionKey='string'
    )
    logger.debug(response)


if __name__ == "__main__":
    lambda_handler(None, None)
