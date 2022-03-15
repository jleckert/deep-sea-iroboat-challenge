import json
import os
import pika
import s3_helper
from log import logger


def lambda_handler(event, context):
    """This lambda function is trigerred by a new wind data file (CSV).
    It posts the data to each RabbitMQ queues (one queue per team)

    Args:
        event (JSON): S3 Put event (new wind data)
    """
    source_bucket_name = event['Records'][0]['s3']['bucket']['name']
    object_key = event['Records'][0]['s3']['object']['key']
    logger.debug(
        f'Parsing content from file {object_key} in bucket {source_bucket_name}')
    content = s3_helper.csv_to_string(
        source_bucket_name, object_key)
    logger.debug(
        f'Parsed content from file {object_key} in bucket {source_bucket_name}')

    # Create a connection to a RabbitMQ server
    credentials = pika.PlainCredentials(
        os.environ['RABBIT_USER'], os.environ['RABBIT_PWD'])
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(os.environ['RABBIT_HOST'], 5672, '/', credentials))
    channel = connection.channel()

    for team in range(1, 141):
        queue_name = f'wind_team_{team}'
        channel.queue_declare(queue=queue_name)

        # Send a message
        channel.basic_publish(exchange='',
                              routing_key=queue_name,
                              body=content)
        logger.info(f'Pushed message to {queue_name}')
    # Close the connection
    connection.close()

    return 'Done!'


if __name__ == "__main__":
    with open('events/vr_push_wind_rabbit.json') as json_file:
        data = json.load(json_file)
    lambda_handler(data, None)
