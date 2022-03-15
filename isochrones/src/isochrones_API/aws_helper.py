import boto3
import botocore
from botocore.exceptions import ClientError
import os
import datetime as dt
from src.common.log import logger
import json


class AWSHelper:
    """Class used to handle S3 operations
    """

    def __init__(self):
        """Constructor
        """
        super().__init__()
        session = boto3.Session()
        self.s3 = session.client('s3')
        self.s3_resource = session.resource('s3')

    def get_target_files(self, bucket, path_s3, request_id):
        """Donwload wind files from a given bucket/folder

        Args:
            bucket (string): The S3 bucket containing the wind data
            path_s3 (string): the folder containing the wind data
            request_id (string): Unique ID for the Flask request (used for logging + storing the wind data in an unique folder)
        """
        filelist = [f for f in os.listdir(
            f"{request_id}/winds") if f.endswith(".csv")]
        for f in filelist:
            os.remove(os.path.join(f"{request_id}/winds", f))
        n = 6
        c_file = ""
        while True:
            with open(f"{request_id}/winds/{n-6}.csv", "wb") as f:
                c_file = f"{request_id}/winds/{n-6}.csv"
                n_str = str(n)
                while len(n_str) < 3:
                    n_str = "0" + n_str
                target = f"{path_s3}/{n_str}.csv"
                try:
                    self.s3_resource.Bucket(
                        bucket).Object(target).load()
                    self.s3.download_fileobj(
                        bucket, target, f)
                    logger.debug(
                        f"Request ID {request_id}: {c_file} download complete.")
                except botocore.exceptions.ClientError as e:

                    if e.response['Error']['Code'] == "403":
                        logger.error(
                            f"Request ID {request_id}: Could not download file {target}")
                        logger.error(f"Request ID {request_id}: {e}")
                        logger.error(f'Request ID {request_id}: Aborting')
                        exit()
                    if e.response['Error']['Code'] == "404":
                        logger.warning(
                            f"Request ID {request_id}: Could not download file {target}, but this is expected behavior!")
                        logger.warning(f"Request ID {request_id}: {e}")
                        break
            n += 3
        os.remove(c_file)

    def store_result(self, bucket, path_s3, json_data):
        """Store isochrones computation results to S3

        Args:
            bucket (string): The bucket to write to
            path_s3 (string): the file path (key in S3 vocabulary) to write to
            json_data (dictionary): the file content (i.e. isochrones results)
        """
        s3object = self.s3_resource.Object(
            bucket, path_s3)

        s3object.put(
            Body=(bytes(json.dumps(json_data).encode('UTF-8')))
        )
