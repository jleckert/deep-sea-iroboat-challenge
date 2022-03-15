import logging
import os.path

# create logger
logger = logging.getLogger('')

# reduce log level
logging.getLogger("pika").setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)
logger.setLevel(logging.INFO)

# Only for running locally
# (so the test condition could be something else that differentiates running on lambda vs.
# running locally)
if os.path.isfile(".env"):
    import coloredlogs
    # By default the install() function installs a handler on the root logger,
    # this means that log messages from your code and log messages from the
    # libraries that you use will all show up on the terminal.
    coloredlogs.install(level='INFO')
