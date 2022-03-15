"""
This script is a ResNet training script which uses Tensorflow's Keras interface, and provides an example of how to use SageMaker Debugger when you use your own custom container in SageMaker or your own script outside SageMaker.
It has been orchestrated with SageMaker Debugger hooks to allow saving tensors during training.
These hooks have been instrumented to read from a JSON configuration that SageMaker puts in the training container.
Configuration provided to the SageMaker python SDK when creating a job will be passed on to the hook.
This allows you to use the same script with different configurations across different runs.

If you use an official SageMaker Framework container (i.e. AWS Deep Learning Container), you do not have to orchestrate your script as below. Hooks are automatically added in those environments. This experience is called a "zero script change". For more information, see https://github.com/awslabs/sagemaker-debugger/blob/master/docs/sagemaker.md#zero-script-change. An example of the same is provided at https://github.com/awslabs/amazon-sagemaker-examples/sagemaker-debugger/tensorflow2/tensorflow2_zero_code_change.
"""

# Standard Library
import argparse
import random
from os.path import join

# Third Party
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# smdebug modification: Import smdebug support for Tensorflow
import smdebug.tensorflow as smd


def train(batch_size, epoch, model, hook, bucket, data_dir, test_size, random_state):
    
    s3 = boto3.resource('s3')
    #loading data from s3 bucket
    obj = s3.Object(bucket_name=bucket, key=join(data_dir, 'input.csv'))
    data = pd.read_csv(obj.get()['Body'])
    
    # split test/train sets
    rand_split = np.random.rand(len(data))
    train_list = rand_split < 0.8
    val_list = (rand_split >= 0.8) & (rand_split < 0.9)
    test_list = rand_split >= 0.9
    
    data_train = data[train_list]
    data_val = data[val_list]
    data_test = data[test_list]

    train_y = data_train.iloc[:,-1]
    train_X = data_train.iloc[:,:-1]
    
    val_y = data_val.iloc[:,-1]
    val_X = data_val.iloc[:,:-1]

    test_y = data_test.iloc[:,-1]
    test_X = data_test.iloc[:,:-1]
    
    #normalize dataset
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x_train_scaled = scaler_x.fit_transform(train_X)
    y_train_scaled = scaler_y.fit_transform(train_y.values.reshape(-1,1))
    # Here it's important to use the same scaler of Training data set for the Testing dataset
    x_val_scaled = scaler_x.transform(val_X)
    y_val_scaled = scaler_y.transform(val_y.values.reshape(-1,1))

    x_test_scaled = scaler_x.transform(test_X)
    y_test_scaled = scaler_y.transform(test_y.values.reshape(-1,1))
    
    #download test file to s3 to validate model   
    csv_buffer_x = StringIO()
    test_X.to_csv(csv_buffer_x ,header=True, index=False)
    s3.Object(join(bucket,'test', 'test_x.csv')).put(Body=csv_buffer_x.getvalue())

    csv_buffer_y = StringIO()
    test_y.to_csv(csv_buffer_y ,header=True, index=False)
    s3.Object(join(bucket,'test', 'test_y.csv')).put(Body=csv_buffer_y.getvalue())

    # register hook to save the following scalar values
    hook.save_scalar("epoch", epoch)
    hook.save_scalar("batch_size", batch_size)
    hook.save_scalar("train_steps_per_epoch", len(X_train)/batch_size)
    hook.save_scalar("valid_steps_per_epoch", len(X_valid)/batch_size)

    model.fit(x_train_scaled, y_train_scaled,
              batch_size=batch_size,
              epochs=epoch,
              validation_data=(x_val_scaled, y_val_scaled),
              shuffle=True,
              # smdebug modification: Pass the hook as a Keras callback
              callbacks=[hook])


def main():
    parser = argparse.ArgumentParser(description="Train keras model")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--model_dir", type=str, default="./model_keras_resnet")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--test_size", type=float, default=0.10)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--bucket", type=str, default='virtual-regatta-ml')
    parser.add_argument("--data_dir", type=str, default='navigation-boat-ml/data')

    args = parser.parse_args()

    model = Sequential()
    model.add(Dense(128, input_dim=len(opt.features), activation='relu', kernel_initializer='normal'))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1))  #kernel_initializer='normal',activation='linear'))
    optimizer = Adam(learning_rate=args.lr)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse'])

    # smdebug modification:
    # Create hook from the configuration provided through sagemaker python sdk.
    # This configuration is provided in the form of a JSON file.
    # Default JSON configuration file:
    # {
    #     "LocalPath": <path on device where tensors will be saved>
    # }"
    # Alternatively, you could pass custom debugger configuration (using DebuggerHookConfig)
    # through SageMaker Estimator. For more information, https://github.com/aws/sagemaker-python-sdk/blob/master/doc/amazon_sagemaker_debugger.rst
    hook = smd.KerasHook.create_from_json_file()

    # start the training.
    train(args.batch_size, args.epoch, model, hook, bucket, data_dir, test_size, random_state)

if __name__ == "__main__":
    main()
