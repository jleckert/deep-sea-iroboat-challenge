from src import vr_trigger_isos_update as lambda_fn
import json


def test_overall():
    with open('./events/vr_trigger_isos_update.json') as json_file:
        data = json.load(json_file)
    assert lambda_fn.lambda_handler(data, None) == 'Done!'


test_overall()
