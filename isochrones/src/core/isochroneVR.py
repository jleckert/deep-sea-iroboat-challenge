import logging
import isochroneVR_sphere as sphere
import isochroneVR_isochrone as isochrone
import isochroneVR_vecfields as fields
import isochroneVR_envelopes as envelopes
from dotenv import load_dotenv, find_dotenv
import botocore
import boto3
import datetime as dt
import os
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import random
import time
import matplotlib
from src.common.log import logger
matplotlib.use('TkAgg')


aws_access_key = os.getenv("AWS_ACCESS_KEY")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION")
request_id = 'test_api'

session = boto3.Session(
    region_name="eu-west-2"
)

s3 = session.client('s3')
s3_resource = session.resource('s3')

start_hour = dt.datetime.now().hour % 6

# COMMENT THE BELOW LINE TO GET RANDOM FIELDS, AS IN ACTUAL CHALLENGE
# random.seed(0)

def get_target_files(N):
    filelist = [f for f in os.listdir("winds") if f.endswith(".csv")]
    for f in filelist:
        os.remove(os.path.join("winds", f))

    filelist = [f for f in os.listdir("winds_processed") if f.endswith(".npy")]
    for f in filelist:
        os.remove(os.path.join("winds_processed", f))

    hour = dt.datetime.now().hour
    hour //= 6

    if hour == 0:
        N -= 1

    hour_dict = {
        0: "18",
        1: "00",
        2: "06",
        3: "12",
    }

    n = 6
    c_file = ""
    while True:
        with open(f"winds/{n-6}.csv", "wb") as f:
            c_file = f"winds/{n-6}.csv"
            n_str = str(n)
            while len(n_str) < 3:
                n_str = "0" + n_str
            target = f"winds/csv/{N}/{hour_dict[hour]}/{n_str}.csv"
            try:
                s3_resource.Bucket(
                    'virtual-regatta-wind-data').Object(target).load()
                s3.download_fileobj('virtual-regatta-wind-data', target, f)
                logger.info(f"{c_file} download complete.")
            except botocore.exceptions.ClientError as e:
                logger.warning(target)
                logger.warning(e)
                if e.response['Error']['Code'] == "403":
                    exit()
                if e.response['Error']['Code'] == "404":
                    break
        n += 3
    os.remove(c_file)


PROCESS_C_FILES = False

if PROCESS_C_FILES:
    get_target_files(20200916)

bounds = -180, 180, -90, 90

f = fields.WindField(request_id, process_winds=PROCESS_C_FILES)

wind_data = pd.read_csv("polar.csv", delimiter=";").to_numpy()[:, 1:]
we = envelopes.WindEnvelope(wind_data, sphere.Sphere(6371))

t = 0
# plt.clf()

times = []

o_time = time.time_ns()
c_time = time.time_ns()

iso = None
iso_collide = None

start_lon, start_lat = np.array(-20.), np.array(0)

#goal2 = isochrone.GoalCollision([(145., 145., -38.5, -67.)])
goal2 = isochrone.GoalCollision([(20., 20., -33.5, -70.)])
goal3 = isochrone.GoalCollision([(145., 145., -33.5, -70.)])
goal = isochrone.GoalCollision([(-10., -10., 8, -10.)])
# goal = isochrone.GoalCollision([(145., 145., -38.5, -67.)])

for epoch in range(5):
    iso = isochrone.Isochrone(
        np.pi / 20, we, f, 70, goal, hour_scale=16./(1 + 2*epoch), check_intersect=True)
    iso.propagate_goals([goal2, goal3])
    win_pts = iso.full_compute(start_lon, start_lat, request_id, plot_result=False, constrain_path=iso_collide,
                               constrain_radius=20/(1 + 2*epoch), start_time=start_hour, bounds=(-180, 180, -90, 90))
    iso_collide = win_pts[0]

win_pts
