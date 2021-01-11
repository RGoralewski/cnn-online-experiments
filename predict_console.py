import paho.mqtt.client as mqtt
import time
import json
import os
import numpy as np
import base64
import struct
import sys
from scipy import signal
import torch

from circular_queue import CircularQueue
from put_emg_gestures_classification.pegc.models import Resnet1D
from put_emg_gestures_classification.pegc import constants
from put_emg_gestures_classification.pegc.training.utils import load_json

BASE64_BYTES_PER_SAMPLE = 4
SCALE_FACTOR_EMG = 4500000/24/(2**23-1)  # uV/count
EMG_FREQUENCY = 1000  # Hz


def filter_signals(s: np.ndarray, fs: int) -> np.ndarray:
    # Highpass filter
    fd = 10  # Hz
    n_fd = fd / (fs / 2)  # normalized frequency
    b, a = signal.butter(1, n_fd, 'highpass')
    hp_filtered_signal = signal.lfilter(b, a, s.T)

    # Notch filter
    notch_filtered_signal = hp_filtered_signal  # cut off the beginning and transpose
    for f0 in [50, 100, 200]:  # 50Hz and 100Hz notch filter
        Q = 5  # quality factor
        b, a = signal.iirnotch(f0, Q, fs)
        notch_filtered_signal = signal.lfilter(b, a, notch_filtered_signal)

    return notch_filtered_signal.T


def base64_to_list_of_channels(encoded_data, bytes_per_sample):
    """Convert base64 string of samples (on each channel) to list of integers
    """
    output = list()
    for i in range(int(len(encoded_data) / bytes_per_sample)):
        # decode base64
        sample_start = i * bytes_per_sample
        decoded_bytes = base64.b64decode(encoded_data[sample_start:sample_start+bytes_per_sample])

        # convert 24-bit signed int to 32-bit signed int
        decoded = struct.unpack('>i', (b'\0' if decoded_bytes[0] < 128 else b'\xff') + decoded_bytes)
        output.append(decoded)
    return output


def on_message(client, userdata, message):
    """Callback function
    """
    global file_created
    global samples_counter
    global gestures_classes
    global model
    global window_length

    global buffer
    global model

    # Decode a message
    message_as_string = str(message.payload.decode("utf-8"))
    message_as_string = ''.join(message_as_string.split())  # remove all white characters

    # Decode the JSON
    packet = json.loads(message_as_string)

    # Read number of channels
    num_channels = packet["channels"]

    # Add a number of samples to the counter
    samples_counter += packet["packets"]

    # Save data in circular buffer
    for encoded_channels in packet["data"]:
        if len(encoded_channels) == num_channels * BASE64_BYTES_PER_SAMPLE:
            channels_list = base64_to_list_of_channels(encoded_channels, BASE64_BYTES_PER_SAMPLE)
            scaled_data = np.asarray(channels_list).T * SCALE_FACTOR_EMG
            # Save decoded data to the buffer
            buffer.push(scaled_data)

    # If data for the window is collected
    if buffer.get_size() >= window_length:
        start = time.time()
        # get data from the buffer
        data_from_buffer = buffer.get_window_of_data(window_length)
        # filtering
        expanded_data = np.concatenate((data_from_buffer,) * 3)  # triple data length for filtering
        filtered_data = filter_signals(expanded_data, EMG_FREQUENCY)

        # Transpose and expand dimension because CNN input=(batch, channels, data_window)
        training_data = np.expand_dims(filtered_data[-200:].T, axis=0)
        # predicting
        with torch.no_grad():
            y_pred = model(torch.Tensor(training_data))

        # print time and loss
        print('****************** '
              f'Filtering and predicting time: {time.time() - start} '
              f'Gesture: {gestures_classes[int(torch.argmax(y_pred))]} PD: {y_pred} '
              '******************')


# MQTT data
broker_data = {
  "broker_address": "192.168.9.100"
}
topic = "sensors/data/emg"

# File created flag
file_created = False

# Circular buffer
buff_cap = 300
buffer = CircularQueue(buff_cap)

# Create specified model.
training_config_file_path = "put_emg_gestures_classification/experiment_scripts/config_template.json"
training_config = load_json(training_config_file_path)
model = Resnet1D(constants.DATASET_FEATURES_SHAPE[0], constants.NB_DATASET_CLASSES,
                 training_config["nb_res_blocks"], training_config["res_block_per_expansion"],
                 training_config["base_feature_maps"])

# Load model and optimizer from checkpoint
model_path = "openbci_third_trained_model.tar"
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])

# Ensure these layers are in evaluation mode
model.eval()

# Window length
window_length = 200  # ms

# Script start time
script_start_time = time.time()

# The time of last sampling rate print
last_sampling_rate_print_time = script_start_time

# List of gestures
gestures_classes = ['idle', 'fist', 'flexion', 'extension', 'pinch_index', 'pinch_middle',
                    'pinch_ring', 'pinch_small']

# Samples counter
samples_counter = 0

# Create a client
print("Creating new subscriber instance")
client = mqtt.Client("emg_sub")

# Type username and password before connecting to broker
# client.username_pw_set(broker_data['username'], broker_data['password'])

# Connect with broker
print("Connecting to broker")
client.connect(broker_data['broker_address'])

# Subscribe to topic
print("Subscribing to topic")
client.subscribe(topic)

# Attach function to callback
client.on_message = on_message

try:
    # Start the loop
    client.loop_forever()

except KeyboardInterrupt:
    # Stop subscriber loop
    client.loop_stop()
