import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import serial
import time
import Adafruit_IO
from Adafruit_IO import RequestError,Client,Feed
ADAFRUIT_IO_USERNAME='aakash11'
ADAFRUIT_IO_KEY='aio_jRZS08cfjvCtkoTDpFLJjJ5k5Ojx'
aio=Client(ADAFRUIT_IO_USERNAME,ADAFRUIT_IO_KEY)

def load_data(file_path, columns):
    data = pd.read_csv(file_path)
    sensor_data = data[columns].values
    return sensor_data

def normalize_data(data):
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    normalized_data = (data - data_mean) / data_std
    return normalized_data, data_mean, data_std

def build_autoencoder(input_dim):
    encoder = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='relu')
    ])

    decoder = models.Sequential([
        layers.Dense(16, activation='relu', input_shape=(8,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(input_dim, activation='sigmoid')
    ])


    autoencoder = models.Sequential([encoder, decoder])
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def train_autoencoder(model, data, epochs, batch_size):
    model.fit(data, data, epochs=epochs, batch_size=batch_size)

def calculate_threshold(model, data):
    reconstructed_data = model.predict(data)
    mse = np.mean(np.square(data - reconstructed_data), axis=1)
    mean_error = np.mean(mse)
    std_error = np.std(mse)
    threshold = 2*(mean_error + (2 * std_error))  # Adjust the factor (e.g., 2) based on your requirements
    return threshold

def reconstruction(test_instance):
    normalized_test_instance = (test_instance - data_mean) / data_std
    reconstructed_test_instance = autoencoder.predict(normalized_test_instance.reshape(1, -1))
    denormalized_reconstructed_instance = (reconstructed_test_instance * data_std) + data_mean
    reconstruction_errors = np.mean(np.square(normalized_test_instance - reconstructed_test_instance), axis=0)
    reconstructed_values = denormalized_reconstructed_instance.flatten()
    formatted_values = ', '.join(['{:.6f}'.format(val) for val in reconstructed_values])
    formatted_errors = ', '.join(['{:.6f}'.format(err) for err in reconstruction_errors])
    return formatted_values, formatted_errors

file_path = r"C:\Users\aakas\OneDrive\Desktop\Copy of CDP_Dataset(1).csv"
selected_columns = ['outlet_temperature','outlet_pressure','vibration']
epochs = 1000
batch_size = 32
sensor_data = load_data(file_path, selected_columns)
normalized_data, data_mean, data_std = normalize_data(sensor_data)
input_dim = normalized_data.shape[1]
autoencoder = build_autoencoder(input_dim)
train_autoencoder(autoencoder, normalized_data, epochs, batch_size)

threshold = calculate_threshold(autoencoder, normalized_data)
print("Threshold:", threshold)

while True:

    ser = serial.Serial('COM5', 115200)  # Replace 'COMX' with your port
    time.sleep(2)  # Allow time for serial connection to initialize
    sensor_data = str(ser.readline().strip(), 'utf-8', errors='ignore')
    sensor_values = list(map(float, sensor_data.split(',')))  
    sensor_array = np.array(sensor_values)
    test_instance = sensor_array
    reconstructed_values, reconstruction_errors = reconstruction(test_instance)
    print("Sensor value:",sensor_data)
    print("Reconstructed values:", reconstructed_values)
    print("Reconstruction errors:", reconstruction_errors)
    reconstruction_errors = np.array([float(err) for err in reconstruction_errors.split(', ')])
    anomalies = reconstruction_errors > threshold
    anomalous_features = np.where(anomalies)[0]
    if len(anomalous_features) > 0:
        print("Anomalies detected in:")
        for idx in anomalous_features:
            if idx == 0:
                print("Outlet temperature")
                try:
                    test=aio.feeds('test')
                    test1=aio.feeds('temp')
                except RequestError:
                    test_feed=Feed(name='test')
                    test_feed=aio.create_feed(test_feed)
                    test_feed1=Feed(name='temp')
                    test_feed1=aio.create_feed(test_feed1)
                aio.send_data(test.key,"Outlet temperature")
                aio.send_data(test1.key,sensor_values[0])
            elif idx == 1:
                print("Outlet pressure")
                try:
                    test=aio.feeds('test')
                    test1=aio.feeds('temp')
                except RequestError:
                    test_feed=Feed(name='test')
                    test_feed=aio.create_feed(test_feed)
                    test_feed1=Feed(name='temp')
                    test_feed1=aio.create_feed(test_feed1)
                aio.send_data(test.key,"Outlet pressure")
                aio.send_data(test1.key,sensor_values[1])
            else:
                print("Vibration")
                try:
                    test=aio.feeds('test')
                    test1=aio.feeds('temp')
                except RequestError:
                    test_feed=Feed(name='test')
                    test_feed=aio.create_feed(test_feed)
                    test_feed1=Feed(name='temp')
                    test_feed1=aio.create_feed(test_feed1)
                aio.send_data(test.key,"Vibration")
                aio.send_data(test1.key,sensor_values[2])
    else:
        print("No anomalies detected")
    
    ser.close()

    
