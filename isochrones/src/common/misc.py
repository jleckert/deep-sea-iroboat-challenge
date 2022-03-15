from src.common.log import logger


def safe_config_parse(user_input, key, default):
    if key in user_input and user_input[key] != '':
        return user_input[key]
    logger.debug(
        f'An issue occured parsing:\n{user_input}\nThe key \"{key}\" is either not present or the value is empty. Returning the default value: \"{default}\"')
    return default
