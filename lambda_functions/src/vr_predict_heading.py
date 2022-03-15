import json
import boto3
import math
import os
import numpy as np
import urllib3
from distutils import util
import os.path
if os.path.isfile(".env"):
    print('Found an environnement file, loading it!')
    from dotenv import load_dotenv
    load_dotenv()
    import src.s3_helper as s3_helper
    import src.geometry_helper as geometry_helper
else:
    # Running the code in AWS: all files are in the same folder
    import s3_helper
    import geometry_helper

runtime = boto3.client('runtime.sagemaker')
s3 = boto3.resource('s3')
TESTING = bool(util.strtobool(os.environ['testing']))
BUCKET_BOAT_POSITION = os.environ['bucket_boat_position']
PREFIX_BOAT_POSITION = os.environ['prefix_boat_position']
VR_API_ENDPOINT = os.environ['vr_api_endpoint']
COLLISIONS_ENDPOINT = os.environ['collisions_endpoint']
COS_ENDPOINT = os.environ['cos_endpoint']
SIN_ENDPOINT = os.environ['sin_endpoint']


def call_regression_model(endpoint, payload):
    """Call the cosine/sine heading prediction models

    Args:
        endpoint (string): model endpoint to send request to
        payload (dictionary): payload to send to the model

    Returns:
        float: cosine/sine prediction
    """
    response = runtime.invoke_endpoint(EndpointName=endpoint,
                                       ContentType='text/csv',
                                       Body=payload)
    result = json.loads(response['Body'].read().decode())
    pred = [r['score'] for r in result['predictions']][0]
    pred = min(1, pred)
    pred = max(-1, pred)
    return pred


def get_boat_data(prefix):
    """Get the latest boat status

    Args:
        prefix (string): root folder to look in (typically user/race)

    Returns:
        list: boat status data as a list
    """
    boat_status_folder = s3_helper.get_latest_folder(
        BUCKET_BOAT_POSITION, prefix)
    if boat_status_folder is None:
        print('Could not retrieve latest boat folder, aborting...')
        return None
    key = s3_helper.get_most_recent_file(
        BUCKET_BOAT_POSITION, prefix, boat_status_folder)
    print(f'Latest boat file: {key}')
    return s3_helper.read_csv_boat_status(BUCKET_BOAT_POSITION, key)


def unpack_boat_data(data):
    """unpack boat data from the data as a list

    Args:
        data (list): boat data

    Returns:
        float,float,float,float: data unpacked
    """
    boat_speed = float(data[4])
    angle_of_attack = float(data[5])  # angle of attack = twa
    wind_speed = float(data[6])
    lat = float(data[1])
    lon = float(data[2])
    return boat_speed, angle_of_attack, wind_speed, lon, lat


def get_waypoints(event, object_key):
    """Get waypoints to reach (by exploiting the JSON referenced in the input)

    Args:
        event (JSON): S3 put event
        object_key (string): key of the latest isochrones result

    Returns:
        list: waypoints (list of float)
    """
    bucket = event['Records'][0]['s3']['bucket']['name']
    content_object = s3.Object(bucket, object_key)
    file_content = content_object.get()['Body'].read().decode('utf-8')
    print(f'Isochrones file: {bucket}/{object_key}')
    json_content = json.loads(file_content)
    return json_content['waypoints']


def check_collisions(start, candidate_ends):
    """Call the collisions API with the list of waypoints given by the isochrones

    Args:
        start (list): boat position [x,y]
        candidate_ends (list): waypoints [[x,y],...]

    Returns:
        list: collisions API output: [0,1,0,1,1,...]
    """
    http = urllib3.PoolManager()
    payload = {"start": start, "candidate_ends": candidate_ends}
    response = http.request("POST", COLLISIONS_ENDPOINT, body=json.dumps(payload).encode('utf-8'), headers={
        'Content-Type': 'application/json'})
    return json.loads(response.data.decode('utf-8'))['collisions']


def predict_heading(lat, lon, next_waypoint, boat_speed, angle_of_attack, wind_speed):
    """Overall logic for predicting the bearing/heading given the boat status and the isochrones predictions

    Args:
        lat (float): boat latitude (Y)
        lon (float): boat longitude (X)
        next_waypoint (list): point as a float list [x,y]
        boat_speed (float): boat speed (in knots)
        angle_of_attack (float): True Wind Angle
        wind_speed (float): True Wind Speed

    Returns:
        float: predicted angle
    """
    lat = lat * math.pi / 180
    lon = lon * math.pi / 180
    next_waypoint[0] = next_waypoint[0] * math.pi / 180
    next_waypoint[1] = next_waypoint[1] * math.pi / 180
    target_angle_compass = geometry_helper.angleFromCoordinate(
        lat, lon, next_waypoint[1], next_waypoint[0])

    print(f"Target angle in VR compass: {int(target_angle_compass)}")
    cos_target_angle = math.cos(np.deg2rad(target_angle_compass))
    sin_target_angle = math.sin(np.deg2rad(target_angle_compass))

    payload_cos = f'{boat_speed},{angle_of_attack},{wind_speed},{cos_target_angle},{sin_target_angle}'
    payload_sin = f'{boat_speed},{angle_of_attack},{wind_speed},{sin_target_angle},{cos_target_angle}'
    pred_cos = call_regression_model(COS_ENDPOINT, payload_cos)
    pred_sin = call_regression_model(SIN_ENDPOINT, payload_sin)

    if pred_sin > 0:
        predicted_angle = round(math.degrees(math.acos(pred_cos)), 0)
    else:
        predicted_angle = round(360 - math.degrees(math.acos(pred_cos)), 0)
    print(f'Raw predicted angle (in VR compass) {int(predicted_angle)}')

    predicted_angle = geometry_helper.adjust_predicted_angle(
        predicted_angle, target_angle_compass)
    print(
        f'Adjusted predicted angle (in VR compass) {int(predicted_angle)}')
    return predicted_angle


def call_update_heading_api(angle, user, race):
    """Call the heading update API (game API)

    Args:
        angle (int): new heading to take
        user (string): user ID in-game
        race (string): race ID in-game

    Returns:
        http response: API response
    """
    http = urllib3.PoolManager()
    payload = json.dumps(
        {"newheading": str(angle), "raceid": race, "userid": user})
    print('Sending to the changeHeading API:')
    print(payload)
    response = http.request("POST", VR_API_ENDPOINT,
                            body=payload, headers={'Content-Type': 'application/json'})
    return response


def lambda_handler(event, context):
    """This lambda holds the logic for predicting a heading given an isochrones calculation and a boat status

    Args:
        event (JSON): S3 put event (new isochrones file stored with the supervised/ prefix)
    """
    object_key = event['Records'][0]['s3']['object']['key']
    object_key_list = object_key.split('/')
    user = object_key_list[1]
    race = object_key_list[2]
    prefix = f'{PREFIX_BOAT_POSITION}/{user}/{race}/'

    # Get the latest acquired boat data
    data = get_boat_data(prefix)
    if data is None:
        return 'Error while getting boat data'
    boat_speed, angle_of_attack, wind_speed, lon, lat = unpack_boat_data(data)

    waypoints = get_waypoints(event, object_key)

    collisions = check_collisions([lon, lat], waypoints)
    next_waypoint = geometry_helper.compute_next_wp(
        lat, lon, waypoints, collisions)
    predicted_angle = predict_heading(
        lat, lon, next_waypoint, boat_speed, angle_of_attack, wind_speed)

    if TESTING:
        # Not doing the actual API call
        print('-----------------------------------------------------')
        print('Testing finished, not calling the update heading API')
        print('-----------------------------------------------------')
        return 'Success!'
    api_resp = call_update_heading_api(int(predicted_angle), user, race)
    api_resp_data = json.dumps(json.loads(api_resp.data.decode('utf-8')))
    print('API response:')
    print(f'statusCode {api_resp.status}')
    print(f'response {api_resp_data}')

    return 'Success!'
