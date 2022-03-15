from src import process_script_message as lambda_fn
import json


def test_overall():
    with open('./events/process_script_message.json') as json_file:
        data = json.load(json_file)
    assert lambda_fn.lambda_handler(data, None) == 'Success!'
