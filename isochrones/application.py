import os
import time
from flask import Flask, jsonify, request, Response, abort
import threading
from threading import Thread
from pprint import pformat
import datetime as dt
from src.isochrones_API.aws_helper import AWSHelper
import src.isochrones_API.isochroneVR_API as IsoAPI
from pathlib import Path
from src.common.log import logger
import uuid
from collections import defaultdict
import shutil
import json

# EB looks for an 'application' callable by default.
application = Flask(__name__)
application.secret_key = os.urandom(42)


@application.route("/", methods=['GET'])
def hello():
    """Hello world Flask route, used to check the correct deployment of the app

    Returns:
        string: Hello World string
    """
    return 'Well, hello!'


@application.route("/check_threads", methods=['GET'])
def check_threads():
    """Check all running (CPU) threads and return them

    Returns:
        dictionary: list of running threads, format:
        {
            "140332514547456": {
                "is_alive": true,
                "name": "ThreadPoolExecutor-0_0"
            },
            "140333389694784": {
                "is_alive": true,
                "name": "MainThread"
            }
        }
    """
    try:
        mythreads = threading.enumerate()
        output = defaultdict()
        for thread in mythreads:
            output[thread.ident] = {
                'name': thread.name, 'is_alive': thread.is_alive()}
        return output
    except Exception as _:
        return {}


@application.route("/compute_full_isos", methods=['POST'])
def compute_full_isos():
    """Main route, used to compute isochrones

    Returns:
        dictionary: acknowledgment from the API, format:
        {"request_id": "60dfe519-870c-4be2-b1a1-c99b34ce7810", "started": true, "thread_id": 140331681052416, "thread_name": "Thread-3"}
    """
    return launch_async_task(request)


# Might not be needed right now to have 2 endpoint, but in the future we could enforce that waypoints is specified here
@application.route("/update_isos", methods=['POST'])
def update_isos():
    """Second main route, used to refine isochrones from a previous calculation (passed in waypoints)

    Returns:
        dictionary: acknowledgment from the API, format:
        {"request_id": "60dfe519-870c-4be2-b1a1-c99b34ce7810", "started": true, "thread_id": 140331681052416, "thread_name": "Thread-3"}
    """
    return launch_async_task(request)


@application.route("/check_collisions", methods=['POST'])
def check_collisions():
    """A route to check collisions between a starting point and a list of end points

    Returns:
        dictionary: A binary array representing the collisions:
        {"collisions": [0,1]}
    """
    collisions = IsoAPI.check_collisions(request.get_json())
    collisions_formatted = []
    for c in collisions:
        # 1 -> collision
        # 0 -> no collision
        if c:
            collisions_formatted.append(1)
        else:
            collisions_formatted.append(0)
    return jsonify({"collisions": collisions_formatted})


def launch_async_task(request):
    """The method called by the isochrones calculation routes

    Args:
        request (JSON): input JSON with the parameters, format:
        {
            "bucket_wind": "virtual-regatta-wind-data",
            "path_s3_wind": "winds/csv/20201116/06",
            "bucket_isos": "virtual-regatta-isochrones",
            "path_s3_isos": "live/5f8f047a52e859ef03117e71/440",
            "user": "5f8f047a52e859ef03117e71",
            "race": "440",
            "start_hour": 1,
            "precision_level": 1,
            "start_coords": [-18.824485603236177,30.277923509985012],
            "end_coords": [-66.280513,-58.282422,-54.753339,-63.594742],
            "intermediate_goals": [[23.634847,23.634847,-32.726926,-70.931893], [116.112332,116.112332,-33.781067,-66.701605]],
            "waypoints": [[-18.78320354279635,29.734051955943745], [-18.936962338658308,28.883903978535674]],
            "hour_scale": 12,
            "constrain_radius": 20,
            "map": [18771,9385],
            "ratio_delta_hour": 3
        }

    Returns:
        dictionary: acknowledgment from the API, format:
        {"request_id": "60dfe519-870c-4be2-b1a1-c99b34ce7810", "started": true, "thread_id": 140331681052416, "thread_name": "Thread-3"}
    """
    request_json = request.get_json()
    request_id = str(uuid.uuid4())
    thread = Thread(target=compute_full_isos_task,
                    args=(request_json, request_id))
    thread.daemon = True
    thread.start()  # Asynchronous call
    return jsonify({'thread_id': thread.ident, 'thread_name': str(thread.name), 'request_id': request_id, 'started': True})


def create_folders(request_id):
    """Create folders needed for local wind data download & processing

    Args:
        request_id (string): the ID of the Flask request (for naming uniquely the folders)
    """
    Path(f"{request_id}").mkdir(exist_ok=True)
    Path(f"{request_id}/winds").mkdir(exist_ok=True)
    Path(f"{request_id}/winds_processed").mkdir(exist_ok=True)


def compute_full_isos_task(request_json, request_id):
    """The actual isochrones computation

    Args:
        request_json (dictionary): cf. example in launch_async_task()
        request_id (string): the ID of the FLask request (for prefixing log messages and not get confused)
    """
    logger.info(f'Request ID {request_id}: starting to work on it...')
    try:
        aws_helper = AWSHelper()
        iso_api_params = parse_input(request_json, request_id)
        results = compute_waypoints(iso_api_params, request_id, aws_helper)

        if results != {}:
            now = dt.datetime.now()
            result_s3_key = f"{iso_api_params.path_s3_isos}/{now.year}{now.month:02d}{now.day:02d}/{now.hour}-{now.minute}-{now.second}-{now.microsecond}.json"

            logger.info(f'Request ID {request_id}: Storing results')
            aws_helper.store_result(
                iso_api_params.bucket_isos, result_s3_key, results)
            logger.info(
                f'Request ID {request_id}: Done! Results stored in {iso_api_params.bucket_isos}/{result_s3_key}')
            wp = results['waypoints']
            logger.info(f'Waypoints found: {wp}')
        else:
            logger.warning(
                f'Request ID {request_id}: Could not compute isochrones, aborting')
    except Exception as e:
        logger.error(
            f'Request ID {request_id}: Error received when computing the isochrones.')
        logger.exception(f'Request ID {request_id}: {e}')
    finally:
        shutil.rmtree(f"{request_id}", ignore_errors=True)


def parse_input(request_json, request_id):
    """Parse the Flask input

    Args:
        request_json (dictionary): cf. example in launch_async_task()
        request_id (string): the ID of the Flask request (for prefixing log messages and not get confused)

    Returns:
        dictionary: the parsed input with a few data type updates and additional fields
    """
    iso_api_params = IsoAPI.IsoAPIParams(request_json)
    logger.info(
        f'Request ID {request_id} Parsed input:')
    logger.info(pformat(iso_api_params))
    return iso_api_params


def compute_waypoints(iso_api_params, request_id, aws_helper):
    """The actual isochrones computation

    Args:
        iso_api_params (dictionary): the output of parse_input()
        request_id (string): the ID of the Flask request (for prefixing log messages and not get confused)
        aws_helper (custom class: AWSHelper): an instance of AWSHelper used to handle S3 actions

    Returns:
        dictionary: the isochrones result:
        {
            waypoints: [[0.12121,1.52154],[0.3244,1.37837]]
            ETA: [4,8]
        }
    """
    results = {}
    try:
        create_folders(request_id)
        f, we = IsoAPI.update_wind_predictions(
            aws_helper, request_id, iso_api_params)

        logger.info(f'Request ID {request_id}: Computing isochrones')
        win_pts = IsoAPI.compute_isos(iso_api_params, f, we, request_id)

        results = {
            'start': iso_api_params.start_coords,
            'end': iso_api_params.end_coords_list,
            'waypoints': win_pts[0].tolist(),
            'ETA': win_pts[1].tolist()
        }
    except Exception as e:
        logger.error(
            f'Request ID {request_id}: Error received when computing the isochrones.')
        logger.exception(f'Request ID {request_id}: {e}')
    finally:
        shutil.rmtree(f"{request_id}", ignore_errors=True)
    return results


if __name__ == '__main__':
    application.run(debug=True)
