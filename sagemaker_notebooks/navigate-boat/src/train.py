#!/usr/bin/env python

#Standard library
import argparse
import time
import os
from io import StringIO
import json

# Third Party
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import boto3
import pandas as pd

# import keras modules
import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def train(batch_size, epoch, test_size, random_state, features, target, data_dir,bucket, model):

    s3 = boto3.resource('s3')
    #loading data from s3 bucket
    obj = s3.Object(bucket_name=bucket, key=data_dir+'input.csv')
    data = pd.read_csv(obj.get()['Body'])
    
    # select features and target
    y = data[target]
    x = data[features]
    
    # split test/train sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    
    #download test file to s3 to validate model   
    csv_buffer_x = StringIO()
    x_test.to_csv(csv_buffer_x ,header=True, index=False)
    s3.Object(bucket, data_dir+'x_test.csv').put(Body=csv_buffer_x.getvalue())

    csv_buffer_y = StringIO()
    y_test.to_csv(csv_buffer_y ,header=True, index=False)
    s3.Object(bucket, data_dir+'y_test.csv').put(Body=csv_buffer_y.getvalue())
    
    #normalize dataset
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x_train_scaled = scaler_x.fit_transform(x_train)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1,1))
    # Here it's important to use the same scaler of Training data set for the Testing dataset
    x_test_scaled = scaler_x.transform(x_test)
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1,1))

    #fit model
    model.fit(
        x_train_scaled,
        y_train_scaled,
        batch_size=batch_size,
        epochs=epoch,
        validation_data=(x_test_scaled, y_test_scaled)
    )


def main():
    parser = argparse.ArgumentParser(description="Train keras sequential model for navigation")
    parser.add_argument("--features", type=list, default=['boat_speed', 'target_angle', 'angle_of_attack', 'wind_speed'])
    parser.add_argument("--target", type=str, default='boat_angle')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--test_size", type=float, default=0.10)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--bucket", type=str, default='virtual-regatta-ml')
    parser.add_argument("--data_dir", type=str, default='navigation-boat-ml/data/')
    parser.add_argument("--model_dir", type=str, default='navigation-boat-ml/output')
    
    opt = parser.parse_args()

    model = Sequential()
    model.add(Dense(128, input_dim=len(opt.features), activation='relu', kernel_initializer='normal'))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1))  #kernel_initializer='normal',activation='linear'))
    optimizer = Adam()
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse'])

    # start the training.
    train(opt.batch_size, opt.epoch, opt.test_size, opt.random_state, opt.features, opt.target, opt.data_dir, opt.bucket, model)
    
    model.save('/opt/ml/model/1')


if __name__ == "__main__":
    main()