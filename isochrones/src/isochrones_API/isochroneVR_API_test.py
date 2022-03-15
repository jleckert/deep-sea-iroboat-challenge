from src.isochrones_API.aws_helper import AWSHelper
import datetime as dt
import src.isochrones_API.isochroneVR_API as IsoAPI
from pathlib import Path
from pprint import pprint
from payload import *
import matplotlib

matplotlib.use('TkAgg')


def test_collisions():
    """Call the check collisions method from the API with an hardcoded payload
    """
    json_in = {
        "start": [16.1569439675, -32.6219254749],
        "candidate_ends": [
            [
                -35.238039,
                12.512996
            ],
            [
                23.634847,
                22.216214
            ]
        ]
    }
    result = IsoAPI.check_collisions(json_in)
    pprint(result)


def test_isos(json_in):
    """Call the isochrones API methods with an input taken form the payload.py file

    Args:
        json_in (dictionary): input parameter obtained from payload.py
    """
    matplotlib.use('TkAgg')

    request_id = 'test_api'

    # (X1, X2, Y1, Y2)
    #bounds = (-180, 180, -90, 90)  # Entire globe
    # bounds = (-50, 50, -25, 25)  # Demo
    bounds = (-50, 40, -60, 50)  # Mauricienne
    # bounds = (-90, -60, 10, 50)  # Bermuda
    # bounds = (100, 120, 10, 25)  # China Express
    #bounds = (-20, 0, 30, 50)
    plot_result = True

    Path(f"{request_id}").mkdir(exist_ok=True)
    Path(f"{request_id}/winds").mkdir(exist_ok=True)
    Path(f"{request_id}/winds_processed").mkdir(exist_ok=True)

    aws_helper = AWSHelper()
    iso_api_params = IsoAPI.IsoAPIParams(json_in)
    print(iso_api_params)

    f, we = IsoAPI.update_wind_predictions(
        aws_helper, request_id, iso_api_params, download_raw_wind=False, process_winds=False)

    win_pts = IsoAPI.compute_isos(
        iso_api_params, f, we, request_id, plot_result=plot_result, bounds=bounds)

    results = {
        'start': iso_api_params.start_coords,
        'end': iso_api_params.end_coords_list,
        'waypoints': win_pts[0].tolist(),
        'ETA': win_pts[1].tolist()
    }
    print('-------')
    pprint(results)


if __name__ == "__main__":
    test_collisions()
    test_isos(mauricienne_wp)
