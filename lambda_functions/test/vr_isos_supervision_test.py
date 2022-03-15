from src import vr_isos_supervision as lambda_fn
import json


def test_overall():
    with open('./events/vr_isos_supervision.json') as json_file:
        data = json.load(json_file)
    assert lambda_fn.lambda_handler(data, None) == 'Done!'


test_overall()
