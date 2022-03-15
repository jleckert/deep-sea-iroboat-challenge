from io import StringIO
from datetime import datetime
import base64
import json
import os
from distutils import util
import pandas as pd
import boto3
import s3_helper
from log import logger


BUCKET_BOAT_POSITION = os.environ['BUCKET_BOAT_POSITION']
TESTING = bool(util.strtobool(os.environ['TESTING']))


def lambda_handler(event, context):
    """This lambda function acquires a message from a Kinesis stream
    It processes and stores it to S3

    Args:
        event (base64 string): Kinesis data

    Returns:
        string: Success or not
    """
    s3_resource = boto3.resource('s3')
    for record in event['Records']:
        # Kinesis data is base64 encoded so decode here
        payload = base64.b64decode(record['kinesis']['data'])
        body = json.loads(payload)
        if('tws' in body):
            try:
                process_log_script_message(s3_resource, body)
            except Exception as e:
                logger.error(f'Error with record {body}')
                logger.exception(e)
    return 'Success!'


def process_sails(nbSail):
    sailNames = [0, "Jib", "Spi", "Stay", "LJ", "C0", "HG", "LG", 8, 9,
                 "Auto", "Jib (Auto)", "Spi (Auto)", "Stay (Auto)", "LJ (Auto)", "C0 (Auto)", "HG (Auto)", "LG (Auto)"]
    return sailNames[int(nbSail) % 10]


def formatTime(timestamp):
    return datetime.fromtimestamp(int(timestamp)//1000).strftime("%Y-%m-%d %H:%M:%S")


def format_script_message(row):
    row['sail'] = process_sails(row['sail'])
    row['lastCalcDate'] = formatTime(row['lastCalcDate'])
    for c in ["pos.lat", "pos.lon", "heading", "speed", "twa", "tws"]:
        row[c] = (lambda x: str(x))(row[c])
    return row


def write_fleet_to_s3(s3_resource, record_as_df, user_id, race_id, leg):
    """Store boat status as an individual S3 file

    Args:
        data (Pandas dataframe): data to store
        user_id (string): The user ID in-game
        race_id (string): The race ID in-game
    """
    csv_buffer = StringIO()
    key = f'logs_boat_status/{user_id}/{race_id}/leg_{leg}/user_race_data.csv'
    logger.info(f'Storing info in {key}')
    if TESTING:
        logger.debug('-----------------------------------------------------')
        logger.debug('Testing only, not storing anything to S3')
        logger.debug('-----------------------------------------------------')
    elif s3_helper.key_exists(BUCKET_BOAT_POSITION, key, s3_resource):
        # File exists, load it, append it, write it (Daft Punk style)
        content = s3_helper.read_concat_csv_boat_status(
            BUCKET_BOAT_POSITION, key, s3_resource)
        content_df = pd.DataFrame(content)
        content_df.columns = content_df.iloc[0]
        content_df.drop(content_df.index[0], inplace=True)
        appended_df = content_df.append(record_as_df).drop_duplicates()
        appended_df.to_csv(csv_buffer, header=True, index=False)
    else:
        record_as_df.to_csv(csv_buffer, header=True, index=False)
    s3_resource.Object(BUCKET_BOAT_POSITION, key).put(
        Body=csv_buffer.getvalue())


def process_log_script_message(s3_resource, data):
    """Main logic for processing and storing the boat status

    Args:
        data (Pandas dataframe): the data to handle
    """
    cols = ["lastCalcDate", "pos.lat", "pos.lon", "heading", "speed", "twa", "tws",
            "sail"]
    df = pd.json_normalize(data)
    df_reduced = df[cols]
    df_formatted = df_reduced.apply(format_script_message, axis=1)
    df_formatted.rename(
        columns={"pos.lat": "lat", "pos.lon": "lon"}, inplace=True)
    logger.debug('Storing to S3:')
    logger.debug(dict(zip(cols, df_formatted.values.tolist()[0])))
    user_id = data["userId"]
    race_id = data["raceId"]
    leg = data["leg"]
    write_fleet_to_s3(s3_resource, df_formatted, user_id, race_id, leg)


if __name__ == "__main__":
    with open('./events/process_script_message.json') as json_file:
        data = json.load(json_file)
    lambda_handler(data, None) == 'Success!'
