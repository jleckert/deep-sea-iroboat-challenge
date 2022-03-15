from src import vr_trigger_isochrones_computation as lambda_fn
import json


def test_overall():
    with open('./events/vr_trigger_isochrones_computation.json') as json_file:
        data = json.load(json_file)
    assert lambda_fn.lambda_handler(data, None) == 'Done!'
