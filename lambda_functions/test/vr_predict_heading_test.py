from src import vr_predict_heading as lambda_fn
import json


def test_overall():
    with open('./events/vr_predict_heading.json') as json_file:
        data = json.load(json_file)
    assert lambda_fn.lambda_handler(data, None) == 'Success!'
