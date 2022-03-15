import boto3
import datetime as dt
import botocore
import csv
import json


def folder_exists(s3, bucket_name, key):
    """Boolean check if a folder exists in S3 or not. 
    Note that the concept of folder does not really exist in S3, so we check if there are files inside the folder (there can't be empty folders in S3)

    Args:
        s3 (boto3.resource): An instance of boto3 resource
        bucket_name (string): bucket to search in
        key (string): the folder name

    Returns:
        boolean: folder exists (True) or not (False)
    """
    bucket = s3.Bucket(bucket_name)
    objs = list(bucket.objects.filter(Prefix=key))
    if objs != []:
        return True
    else:
        return False


def list_folders(bucket_name, prefix):
    """List folders in a S3 root folder. Note that the concept of folder does not really exist in S3.

    Args:
        bucket_name (string): bucket to search in
        prefix (string): the root folder to search in

    Returns:
        list: sub-folder names as a list of string
    """
    folders = []
    client = boto3.client('s3')
    result = client.list_objects(
        Bucket=bucket_name, Prefix=prefix, Delimiter='/')
    prefixes = result.get('CommonPrefixes')
    if prefixes is None:
        return []
    for o in prefixes:
        folders.append(o.get('Prefix'))
    return folders


def key_exists(bucket, key):
    """Boolean check if a file exists in a bucket

    Args:
        bucket (string): bucket to search in
        key (string): file to search for

    Returns:
        boolean: File exists (True) or not (False) or something went wrong (None)
    """
    s3 = boto3.resource('s3')
    try:
        s3.Object(bucket, key).load()
        return True
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            return False
    print(
        f'Something went wrong trying to find object {key} in bucket {bucket}')
    return None


def get_latest_folder(bucket, prefix, depth_limit=1, suffix='', start_days_in_past=0):
    """Retrieve the latest sub-folder in a root folder, by matching a specific folder naming (using datetime)

    Args:
        bucket (string): bucket to search in
        prefix (string): the root folder
        depth_limit (int, optional): how many days to go back in time. Defaults to 1.
        suffix (str, optional): a suffix to add after the datetime naming: prefix/datetime/suffix. Defaults to ''.

    Returns:
        string: the latest folder
    """
    s3 = boto3.resource('s3')    
    date = dt.datetime.now()
    if start_days_in_past != 0:
        date -= dt.timedelta(days=start_days_in_past)
    root_folder = f'{date.year}{date.month:02d}{date.day:02d}'

    # Checking if today's folder exists
    folder_found = folder_exists(
        s3, bucket, f'{prefix}{root_folder}{suffix}')

    depth = 0
    while not folder_found:
        depth += 1
        if depth > depth_limit:
            print(
                f'Could not retrieve folder for: {prefix}{root_folder}{suffix}, with depth limit: {depth_limit}, skipping it')
            return None
        print(f'Folder does not exist: {prefix}{root_folder}{suffix}')
        date -= dt.timedelta(days=1)
        root_folder = f'{date.year}{date.month:02d}{date.day:02d}'
        folder_found = folder_exists(
            s3, bucket, f'{prefix}{root_folder}{suffix}')
    return root_folder

def get_most_recent_file(bucket, prefix, folder=''):
    """Get most recent file in a given bucket/folder

    Args:
        bucket (string): bucket to search in
        prefix (string): folder to look in
        folder (str, optional): suffix to the foler: prefix/folder. Defaults to ''.

    Returns:
        string: the key of the most recent file (not the data itself!)
    """
    all = get_all_files(bucket, prefix, folder=folder)
    latest = max(all, key=lambda x: x['LastModified'])
    key = latest['Key']
    return key


def get_all_files(bucket, prefix, folder=''):
    s3_client = boto3.client('s3')
    response = s3_client.list_objects_v2(
        Bucket=bucket, Prefix=f'{prefix}{folder}')
    all = response['Contents']
    return all

def read_csv_boat_status(bucket, key):
    """Get boat status data

    Args:
        bucket (string): bucket to search in
        key (string): file to look in

    Returns:
        list: boat data as a list
        lastCalcDate  lat lon heading speed   twa tws twd sail    distanceFromStart   distanceToEnd   credits timeToNextCards aground
    """
    s3 = boto3.resource('s3')
    obj = s3.Object(bucket, key)
    csv_str = obj.get()['Body'].read().decode('utf-8')
    reader = csv.reader(csv_str.split('\n'), delimiter=',')
    data = None
    for row in reader:
        if row != []:
            data = row

    # lastCalcDate  lat lon heading speed   twa tws twd sail    distanceFromStart   distanceToEnd   credits timeToNextCards aground
    return data


def get_waypoints(bucket, object_key):
    s3 = boto3.resource('s3')
    content_object = s3.Object(bucket, object_key)
    file_content = content_object.get()['Body'].read().decode('utf-8')
    print(f'Isochrones file: {bucket}/{object_key}')
    json_content = json.loads(file_content)
    return json_content['waypoints']