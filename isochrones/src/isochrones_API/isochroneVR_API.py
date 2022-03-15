import src.core.isochroneVR_sphere as sphere
import src.core.isochroneVR_isochrone as isochrone
import src.core.isochroneVR_vecfields as fields
import src.core.isochroneVR_envelopes as envelopes
import matplotlib
import datetime as dt
import pandas as pd
import numpy as np
from src.common.log import logger
from src.common.misc import safe_config_parse
from sqlalchemy import create_engine, select, MetaData, Table
import os


def update_wind_predictions(aws_helper, request_id, iso_api_params, download_raw_wind=True, process_winds=True):
    """Download and pre-process wind data

    Args:
        aws_helper (custom class: AwsHelper): An instance of the AwsHelper class used for handling S3 operations
        request_id (string): Flask request ID, used for tying logs to a specific request
        iso_api_params (dictionary): The request parsed input
        download_raw_wind (bool, optional): Download raw wind file (False is used for local testing). Defaults to True.
        process_winds (bool, optional): Pre-process wind file (False is used for local testing). Defaults to True.

    Returns:
        [custom class: WindField]: A vector field that utilises wind data from the NOAA, 
                                   and can be used to get a time and space-wise smoothly interpolated wind direction vector for any arbitrary point in space and time.,
        [custom class: Wind Envelope]: A class that defines methods to obtain the displacement envelope for boats relative to wind speed and direction
        vectors.
    """
    if download_raw_wind:
        logger.info(f'Request ID {request_id}: Downloading wind files')
        aws_helper.get_target_files(
            iso_api_params.bucket_wind, iso_api_params.path_s3_wind, request_id)

    logger.info(f'Request ID {request_id}: Pre-processing wind files')
    f = fields.WindField(request_id, process_winds=process_winds)

    race_id = iso_api_params.path_s3_isos.split("/")[-1]
    logger.info(
        f'Request ID {request_id}: Using polar file polar_{race_id}.csv')
    wind_data = pd.read_csv(
        f"polar_{race_id}.csv", delimiter=";").to_numpy()[:, 1:]
    we = envelopes.WindEnvelope(wind_data, sphere.Sphere(6371))
    return f, we


def compute_isos(iso_api_params, f, we, request_id, plot_result=False, bounds=(-180, 180, -90, 90)):
    """Compute isochrones

    Args:
        iso_api_params (dictionary): The request parsed input
        f ([custom class: WindField]): A vector field that utilises wind data from the NOAA, 
                                   and can be used to get a time and space-wise smoothly interpolated wind direction vector for any arbitrary point in space and time.,
        we {[custom class: Wind Envelope]): A class that defines methods to obtain the displacement envelope for boats relative to wind speed and direction
        request_id (string): Flask request ID, used for tying logs to a specific request
        plot_result (bool, optional): Plot results (using Matplolib) (True is used for local testing). Defaults to False.
        bounds (tuple, optional): Constrain the search to an area of the globe - used for local testing. Defaults to (-180, 180, -90, 90).

    Returns:
        dictionary: the isochrones result:
        {
            waypoints: [[0.12121,1.52154],[0.3244,1.37837]]
            ETA: [4,8]
        }
    """
    iso = None
    iso_collide = None

    start_lon, start_lat = np.array(iso_api_params.start_coords[0]), np.array(
        iso_api_params.start_coords[1])
    end_goal = isochrone.GoalCollision(iso_api_params.end_coords)
    intermediate_goals = evaluate_intermediate_goals(
        iso_api_params, request_id)
    intermediate_goals = intermediate_goals + [end_goal]
    end_goal = intermediate_goals[0]
    intermediate_goals = intermediate_goals[1:]

    base_constrain_radius = iso_api_params.constrain_radius

    for epoch in range(iso_api_params.precision_level):
        logger.info(
            f'Request ID {request_id}: iteration {epoch}/{iso_api_params.precision_level}')
        logger.info(f'Request ID {request_id}: Using map {iso_api_params.map}')
        iso = isochrone.Isochrone(
            np.pi / 20, we, f, 70, end_goal, hour_scale=iso_api_params.hour_scale/(1 + 2*epoch), check_intersect=True, map=iso_api_params.map)

        # Specify previously found waypoints before first iteration
        if epoch == 0 and iso_api_params.waypoints is not None:
            waypoints = iso_api_params.waypoints
            for _ in range(0, 5):
                # Solves a bug where there was a collision passing from positive X to negative X
                # i.e. when passing New Zealand and continuing East
                # WARNING: the iteration number (5) is arbitrary
                waypoints = augment_waypoint(waypoints)
            iso_collide = np.array(waypoints)

        if intermediate_goals != []:
            iso.propagate_goals(intermediate_goals)
        win_pts = iso.full_compute(start_lon, start_lat, request_id, plot_result=plot_result, constrain_path=iso_collide,
                                   constrain_radius=base_constrain_radius / (1 + 2*epoch), ratio_delta_h=iso_api_params.ratio_delta_hour,
                                   start_time=iso_api_params.start_hour, bounds=bounds)
        iso_collide = win_pts[0]

    return win_pts


def augment_waypoint(waypoints):
    """Augment the waypoints length by 2 by adding points in the middle of every 2 points.

    Args:
        waypoints (list): list of waypoints: [[0.1,2.3],[4.5,6.7]]

    Returns:
        list: list of augmented waypoints (twice the initial length)
    """
    augmented_wp = []
    for i, wp in enumerate(waypoints):
        augmented_wp.append(wp)
        if i == len(waypoints) - 1:
            break
        next_wp = waypoints[i+1]
        if next_wp[0] < 0 and wp[0] > 0:
            # Only problematic case: passing NZ
            intermediate_wp = [(next_wp[0]-wp[0])/2, (next_wp[1]+wp[1])/2]
        else:
            intermediate_wp = [(next_wp[0]+wp[0])/2, (next_wp[1]+wp[1])/2]
        augmented_wp.append(intermediate_wp)
    return augmented_wp


def evaluate_intermediate_goals(iso_api_params, request_id):
    """Check if intermediate goals have already been traversed by the boat, and returns a filtered list of these goals

    Args:        
        iso_api_params (dictionary): The request parsed input
        request_id (string): Flask request ID, used for tying logs to a specific request

    Returns:
        list: list of filtered intermediate goals: the traversed goals are removed
        [[0.1,1.2],[2.3,3.4]]
    """
    intermediate_goals = []
    for goal in iso_api_params.intermediate_goals:
        intermediate_goals.append(goal)

    trajectory = []
    try:
        trajectory = get_historical_trajectory(
            iso_api_params.user, iso_api_params.race)
    except Exception as e:
        logger.error(
            f'Request ID {request_id}: Exception received when connecting to the DB, skipping traversed goals check.')
        logger.error(f'Request ID {request_id}: Exception: {e}')

    if trajectory == []:
        logger.warning(
            f'Request ID {request_id}: No points found in historical trajectory, not checking traversed goals')
        return [isochrone.GoalCollision(g) for g in intermediate_goals]

    logger.info(
        f'Request ID {request_id}: Found {len(trajectory)} points in the trajectory, checking if intermediate goals have already been traversed...')
    int_goals_clean = []
    for goal in intermediate_goals:
        traversed = isochrone.Isochrone.check_traversed_goals(
            [goal], trajectory, 70)
        if not traversed[0]:
            int_goals_clean.append(isochrone.GoalCollision(goal))
            logger.info(
                f'Request ID {request_id}: Goal not traversed, keeping it: {goal}')
        else:
            logger.warning(
                f'Request ID {request_id}: Already traversed a goal, not keeping it: {goal}')
    return int_goals_clean


def get_historical_trajectory(user, race):
    """Get the boat trajectory (list of points) stored in a DB

    Args:
        user (string): user ID (in game)
        race (string): race ID (in game)

    Returns:
        list: list of points representing the boat's trajectory
    """
    rds_host = os.environ['endpoint']
    db_user = os.environ['db_username']
    password = os.environ['db_password']
    db_name = os.environ['db_name']
    db_port = int(os.environ['db_port'])

    engine = create_engine(
        f'mysql+pymysql://{db_user}:{password}@{rds_host}:{db_port}/{db_name}', echo=False)
    metadata = MetaData(bind=None)
    table = Table('logs', metadata, autoload=True, autoload_with=engine)
    stmt = select([table.columns.lon, table.columns.lat]).where(
        table.columns.user_id == user).where(table.columns.race_id == race)

    connection = engine.connect()
    results = connection.execute(stmt).fetchall()
    return results


def check_collisions(request_json):
    """[summary]

    Args:
        request_json (dictionary): the Flask input:
        {
            "start": [
                16.1569439675,
                -32.6219254749
            ],
            "candidate_ends": [
                [
                    -35.238039,
                    12.512996
                ],
                [
                    23.634847,
                    22.216214
                ]
            ]
        }

    Returns:
        list: list of booleans (True -> collision with the point that has the same index on the input list, False -> no collision)
        {
            "collisions": [
                0,
                1
            ]
        }
    """
    start = safe_config_parse(request_json, "start", "")
    candidate_ends = safe_config_parse(request_json, "candidate_ends", "")
    if start == "" or candidate_ends == "":
        return None
    collisions = isochrone.Isochrone.check_collision(
        start, candidate_ends, 70)
    # [False, True, ...]
    # True -> collision
    # False -> no collision
    return collisions


class IsoAPIParams:
    """A class used to hold the Flask input parameters (and additional ones)
    """

    def __init__(self, request_json):
        """Constructor

        Args:
            request_json (dictionary): Flask input
        """
        self.bucket_wind = safe_config_parse(
            request_json, "bucket_wind", "virtual-regatta-wind-data")
        self.path_s3_wind = safe_config_parse(
            request_json, "path_s3_wind", "winds/csv/20200917/00")

        self.bucket_isos = safe_config_parse(
            request_json, "bucket_isos", "virtual-regatta-isochrones")
        self.path_s3_isos = safe_config_parse(
            request_json, "path_s3_isos", "")

        self.user = self.path_s3_isos.split('/')[-2]
        self.race = self.path_s3_isos.split('/')[-1]

        now_hour = dt.datetime.utcnow().hour
        # Wind data: in folder 00, first file arrives at 4AM UTC+1
        wind_hour = int(self.path_s3_wind.split('/')[-1]) + 3

        if now_hour < wind_hour:
            # Wind folder = 18 -> wind hour = 21h (9PM) UTC+1
            # Case where start hour is the next day (i.e. between midnight and 3AM UTC+1)
            self.start_hour = now_hour+2
        else:
            self.start_hour = now_hour - wind_hour

        self.precision_level = int(safe_config_parse(
            request_json, "precision_level", 1))
        self.start_coords = safe_config_parse(
            request_json, "start_coords", [0., 0.])
        self.end_coords_list = safe_config_parse(
            request_json, "end_coords", [-10, -5, -10, -5])
        self.end_coords = [(self.end_coords_list[0], self.end_coords_list[1],
                            self.end_coords_list[2], self.end_coords_list[3])]
        self.intermediate_goals_list = safe_config_parse(
            request_json, "intermediate_goals", [])
        self.intermediate_goals = []
        for goal in self.intermediate_goals_list:
            self.intermediate_goals.append(
                [(goal[0], goal[1], goal[2], goal[3])])
        self.waypoints = safe_config_parse(request_json, "waypoints", None)
        self.hour_scale = safe_config_parse(request_json, "hour_scale", 4)
        self.constrain_radius = safe_config_parse(
            request_json, "constrain_radius", 20)
        self.map = safe_config_parse(request_json, "map", [18771, 9385])
        self.ratio_delta_hour = safe_config_parse(
            request_json, "ratio_delta_hour", 1)

    def __repr__(self):
        return str(self.__dict__)
