from functools import reduce
import boto3
from distutils import util
import numpy as np
from sklearn import svm
import json
import os.path
if os.path.isfile(".env"):
    print('Found an environnement file, loading it!')
    from dotenv import load_dotenv
    load_dotenv()
    import src.s3_helper as s3_helper
else:
    # Running the code in AWS: all files are in the same folder
    import s3_helper


TESTING = bool(util.strtobool(os.environ['testing']))
ISOS_NUMBER = int(os.environ['isos_supervision_isos_number'])
R2_MIN = float(os.environ['isos_supervision_r2_min'])
WP_NUMBER = int(os.environ['isos_supervision_wp_number'])
BUCKET_ISOS = os.environ['bucket_isos']
PREFIX_ISOS_LIVE = os.environ['prefix_isos_live']
PREFIX_ISOS_SUPERVISED = os.environ['prefix_isos_supervised']
THRESHOLD_SCORE = float(os.environ['isos_supervision_threshold_score'])
MAX_ISO_SKIP = int(os.environ['isos_supervision_max_skip'])


class Isochrone:
    """A class to store isochrones-result related data
    """

    def __init__(self, source_key, waypoints):
        self.source_key = source_key
        self.waypoints = waypoints

        source_key_list = source_key.split('/')
        self.user = source_key_list[1]
        self.race = source_key_list[2]
        self.file_name = source_key_list[-1]
        dest_key_list = [PREFIX_ISOS_SUPERVISED] + source_key_list[1:]
        self.dest_key = reduce(lambda x, y: x+'/'+y, dest_key_list)


class Trajectory:
    """A class that extends the Isochrone one with the output of the SVR-fitting calculation
    """

    def __init__(self, isochrone, iso_x, iso_y, svr):
        self.isochrone = isochrone
        self.iso_x = iso_x
        self.iso_y = iso_y
        self.svr = svr


class Score:
    """A class that extends the Trajectory one with two custom metrics (used for comparing isochrones results)
    """

    def __init__(self, traj, sv_score, coef_score):
        self.traj = traj
        self.sv_score = sv_score
        self.coef_score = coef_score


def get_latest_isos_keys(bucket, prefix, user_id, race, number_of_isos):
    """Acquiring the most recent isochrones results S3 keys

    Args:
        bucket (string): S3 bucket to look in
        prefix (string): Prefix of the S3 key (typically "live" or "supervised")
        user_id (string): In-game user ID
        race (string): Race ID (Vendée Globe -> 440)
        number_of_isos (int): Number of isochrone keys to retrieve

    Returns:
        list[string]: The keys corresponding to the most recent isochrones
    """
    latest_folder = s3_helper.get_latest_folder(
        bucket, f'{prefix}/{user_id}/{race}/')
    all_files = s3_helper.get_all_files(
        bucket, f'{prefix}/{user_id}/{race}/{latest_folder}')
    all_files_sorted = sorted(
        all_files, key=lambda k: k['LastModified'], reverse=True)
    all_keys = [x['Key'] for x in all_files_sorted]
    print(f'Found {len(all_keys)} isochrones files')

    try:
        if len(all_keys) < number_of_isos:
            print(
                f'Found less isochrones files than the threshold: {number_of_isos}, getting files from yesterday')
            second_latest_folder = s3_helper.get_latest_folder(
                bucket, f'{prefix}/{user_id}/{race}/', start_days_in_past=1)
            all_files = s3_helper.get_all_files(
                bucket, f'{prefix}/{user_id}/{race}/{second_latest_folder}')
            all_files_sorted = sorted(
                all_files, key=lambda k: k['LastModified'], reverse=True)
            all_keys += [x['Key'] for x in all_files_sorted]
            print(f'Now have a total of {len(all_keys)} files')
    except Exception as _:
        print('Error when trying to access yetserday folder. Returning all keys as is')

    if len(all_keys) > number_of_isos:
        all_keys = all_keys[0:number_of_isos]
        print(
            f'Truncated the number of files to only consider the first {number_of_isos} first ones')
    return all_keys


def get_latest_isos(bucket, prefix, user, race, isos_number):
    """Retrieve the most recent isochrone results (file key and file content)

    Args:
        bucket (string): S3 bucket to look in
        prefix (string): Prefix of the S3 key (typically "live" or "supervised")
        user_id (string): In-game user ID
        race (string): Race ID (Vendée Globe -> 440)
        number_of_isos (int): Number of isochrone keys to retrieve

    Returns:
        list[Isochrone]: A list of the custom class Isochrone, containing the most recent isochrone results (file key and file content)
    """
    isos_keys = get_latest_isos_keys(bucket, prefix, user, race, isos_number)
    isos_list = []
    for key in isos_keys:
        waypoints = s3_helper.get_waypoints(bucket, key)
        isos_list.append(Isochrone(key, waypoints))
    return isos_list


def get_x_coords(waypoints):
    """Unpack the X-coordinates of a waypoints list, and adds a padding so that the trajectories can be compared

    Args:
        waypoints (list[float, float]): The waypoints [[x, y], ...]

    Returns:
        list[float]: X-coordinates
    """
    x_coords = []
    shift = 0
    for wp in waypoints:
        x = wp[0]
        if x < 0 and abs(x) > shift:
            shift += abs(x)
            print(f'Negative coordinate, new shift value: {shift}')
        x += shift * 2
        x_coords.append(x)
    return x_coords


def get_y_coords(waypoints):
    """Unpack the Y-coordinates of a waypoints list

    Args:
        waypoints (list[float, float]): The waypoints [[x, y], ...]

    Returns:
        list[float]: Y-coordinates
    """
    y_coords = []
    for wp in waypoints:
        y = wp[1]
        y_coords.append(y)
    return y_coords


def fit_svr(isos_list):
    """Try to fit a SVR model for an intance of the custom class Isochrone

    Args:
        isos_list (list[Isochrone]): a list of the custom class Isochrone

    Returns:
        list[Trajectory]: a list of the custom class Trajectory, which contains the initial isochrone object + the fitted SVR model
    """
    traj_list = []
    init_x = None
    init_shift = 0
    for iso in isos_list:
        x_coords = get_x_coords(iso.waypoints[0:WP_NUMBER])
        y_coords = get_y_coords(iso.waypoints[0:WP_NUMBER])

        first_x = x_coords[0]
        if init_x is None:
            init_x = first_x
        else:
            init_shift = first_x - init_x
        x_coords_shift = [a - init_shift for a in x_coords]

        x = np.array(x_coords_shift).reshape(-1, 1)
        y = np.array(y_coords).reshape(-1, 1)
        regr = svm.SVR()
        regr.fit(x, y.ravel())
        r2 = regr.score(x, y)
        if r2 < R2_MIN:
            print(f'Isochrone: {iso.file_name}')
            print(
                f'R2 {r2} is lower than the threshold {R2_MIN}, not keeping this isochrone...')
        else:
            traj = Trajectory(iso, x, y, regr)
            traj_list.append(traj)
    return traj_list


def calculate_scores(traj_ref, list_traj_to_compare):
    """Compares a reference trajectory with a list of other trajectories, by computing custom metrics linked to the SVR models

    Args:
        traj_ref (Trajectory): The reference trajectory (instance of the custom class Trajectory)
        list_traj_to_compare (list[Trajectory]): The trajectories to compare to

    Returns:
        list[Score]: a list of the custom class Score, which contains the initial object Trajectory (and Isochrone) + the comparison with the reference trajectory results
    """
    list_scores = []

    for traj in list_traj_to_compare:
        print(f'Comparing SVR of isochrone: {traj.isochrone.file_name}')
        sv_score = 0
        coef_score = 0
        common_support = 0
        for i, indice in enumerate(traj_ref.svr.support_):
            if indice in traj.svr.support_:
                svr_index = np.where(np.isclose(traj.svr.support_, indice))
                latest_svr_sv = traj_ref.svr.support_vectors_[i]
                latest_svr_coef = traj_ref.svr.dual_coef_[0][i]
                svr_sv = traj.svr.support_vectors_[svr_index]
                svr_coef = traj.svr.dual_coef_[0][svr_index]
                sv_score += abs(latest_svr_sv - svr_sv)
                coef_score += abs(latest_svr_coef - svr_coef)
                common_support += 1
        if common_support == 0:
            print(
                f'No support indices in common for SVR of isochrone: {traj.isochrone.file_name}')
        else:
            sv_score /= common_support
            coef_score /= common_support
            sv_score = sv_score.tolist()[0][0]
            coef_score = coef_score.tolist()[0]
            list_scores.append(Score(traj, sv_score, coef_score))
        print(
            f'SV score: {round(sv_score, 3)}  |  Coef score: {round(coef_score, 3)}')
    return list_scores


def store_iso(source_bucket, source_key, dest_bucket, dest_key):
    """Store the isochrones result (actually, more of a copy/paste)

    Args:
        source_bucket (string): The bucket where the initial file sits
        source_key (string): The key of the initial file
        dest_bucket (string): The destination bucket
        dest_key (string): The key for the destination file
    """
    s3_resource = boto3.resource('s3')
    content_object = s3_resource.Object(source_bucket, source_key)
    file_content = content_object.get()['Body'].read().decode('utf-8')
    json_content = json.loads(file_content)

    # Upload the .file as-is to S3
    print(f'Uploading file {dest_key} to bucket {dest_bucket}')
    if TESTING:
        print('-----------------------------------------------------')
        print('Testing only, not storing anything to S3')
        print('-----------------------------------------------------')
    else:
        s3object = s3_resource.Object(
            dest_bucket, dest_key)

        s3object.put(
            Body=(bytes(json.dumps(json_content).encode('UTF-8')))
        )
        print(
            f'Succesfully uploaded input file in bucket {dest_bucket}, file name {dest_key}')


def kept_last_isos(user, race):
    """Assess if some isochrones were stored in supervised/ recently (and thus avoid skipping to many of them)

    Args:
        user (string): In-game user ID
        race (string): Race ID (Vendée Globe -> 440)

    Returns:
        bool: True if we consider that "enough" isochrones were stored recently, False otherwise
    """
    last_isos_kept = False
    # Get last x keys in live/...
    live_keys = get_latest_isos_keys(BUCKET_ISOS, PREFIX_ISOS_LIVE,
                                     user, race, MAX_ISO_SKIP + 1)
    # This lambda is trigerred by a storage in the folder live/
    # So the latest file in that folder should not be considered, because there will never be a case where it is already stored in supervised/
    live_keys_filt = live_keys[1:]

    # Get last x keys in supervised/...
    try:
        supervised_keys = get_latest_isos_keys(BUCKET_ISOS, PREFIX_ISOS_SUPERVISED,
                                               user, race, MAX_ISO_SKIP)
    except Exception as _:
        print('An error occured while acquiring the latest files stored in the supervised folder')
        print('Fallback: storing the isochrone')
        return last_isos_kept

    if len(supervised_keys) < MAX_ISO_SKIP:
        print('Looks like there are not enough supervised files stored')
        print('Fallback: storing the isochrone')
        return last_isos_kept

    live_file_name_list = []
    supervised_file_name_list = []
    for key in live_keys_filt:
        live_file_name_list.append(key.split('/')[-1])
    for key in supervised_keys:
        supervised_file_name_list.append(key.split('/')[-1])

    # Compare
    skip_count = 0
    for file in live_file_name_list:
        if file not in supervised_file_name_list:
            print(f'key {file} was NOT stored in supervised/')
            skip_count += 1
        else:
            print(f'key {file} was stored in supervised/')

    if skip_count < MAX_ISO_SKIP:
        last_isos_kept = True
        print(f'{skip_count} isochrones were not stored recently, which is less than the threshold {MAX_ISO_SKIP}')
        print('Continuing with regular execution')
        return last_isos_kept

    print(f'{skip_count} isochrones were not stored recently, which is more than the threshold {MAX_ISO_SKIP}')
    print('Fallback: storing the isochrone')
    return last_isos_kept


def lambda_handler(event, context):
    """This lambda is trigerred by a isochrones result stored with the live/ prefix.
    It aims at making sure this isochrones result is not an "outlier" as compared with the previous ones.
    If it is an outlier, the isochrones result is not re-stored with the supervised/ prefix, hence the lambda predict_heading won't be trigerred.

    Args:
        event (JSON): A S3 PUT event, referencing an isochrones result stored with the live/ prefix
    """
    object_key = event['Records'][0]['s3']['object']['key']
    # No need to pass waypoints to this object
    input_iso = Isochrone(object_key, [])
    print(f'Input file: {input_iso.file_name}')
    user = input_iso.user
    race = input_iso.race
    dest_key = input_iso.dest_key

    # If we haven't kept the last 3 isochrones, keep this one without any calculation (we don't want to diverge too much)
    print('Checking which isochrones were stored recently...')
    if not kept_last_isos(user, race):
        store_iso(BUCKET_ISOS, object_key, BUCKET_ISOS, dest_key)
        return 'Done!'

    print('Acquiring the latest waypoints...')
    isos_list = get_latest_isos(
        BUCKET_ISOS, PREFIX_ISOS_LIVE, user, race, ISOS_NUMBER)
    print('Fitting SVR models...')

    traj_list = fit_svr(isos_list)

    ref_traj = None
    traj_to_compare = []
    for traj in traj_list:
        if traj.isochrone.file_name == input_iso.file_name:
            ref_traj = traj
        else:
            traj_to_compare.append(traj)
    if ref_traj is None:
        print('Something went wrong trying to find the reference trajectory/isochrone, aborting...')
        return 'Error!'

    print(
        f'Comparing {ref_traj.isochrone.file_name} with the other trajectories/isochrones...')
    list_scores = calculate_scores(ref_traj, traj_to_compare)
    if list_scores == []:
        print('Could not compute any similarities between SVR!')
        print('Storing the isochrone anyway')

    else:
        sv_scores = []
        for score in list_scores:
            sv_scores.append(score.sv_score)
        mean_sv_score = np.mean(sv_scores)
        if mean_sv_score > THRESHOLD_SCORE:
            print(
                f'The mean SV score {round(mean_sv_score, 2)} is above threshold {THRESHOLD_SCORE}, not storing this isochornes results...')
            return 'Done!'
        print(
            f'The mean SV score {round(mean_sv_score, 2)} is below threshold {THRESHOLD_SCORE}, storing this isochrones results')
    store_iso(BUCKET_ISOS, object_key, BUCKET_ISOS, dest_key)
    return 'Done!'
