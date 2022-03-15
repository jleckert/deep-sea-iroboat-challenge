import os
import pika
from log import logger

CONTINUE_ON_ERROR = False


def callback(ch, method, properties, body):
    logger.debug(f'Received {body}')


def lambda_handler(event, context):
    """This lambda function is trigerred periodically.
    It pulls wind data from RabbitMQ -> simulates the teams behavior
    """
    # Create a connection to a RabbitMQ server
    credentials = pika.PlainCredentials(
        os.environ['RABBIT_USER'], os.environ['RABBIT_PWD'])
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(os.environ['RABBIT_HOST'], 5672, '/', credentials))
    channel = connection.channel()

    for team in range(1, 141):
        queue_name = f'wind_team_{team}'
        try:
            channel.queue_declare(queue=queue_name)
            channel.basic_consume(queue=queue_name,
                                  auto_ack=True,
                                  on_message_callback=callback)
            logger.info(f'Fetched messages from {queue_name}')
        except pika.exceptions.ChannelClosedByBroker as e:
            logger.error(f'Could not connect to queue {queue_name}')
            if CONTINUE_ON_ERROR:
                channel = connection.channel()
            else:
                logger.error('Aborting')
                break

    # Close the connection
    connection.close()

    return 'Done!'


if __name__ == "__main__":
    lambda_handler(None, None)
