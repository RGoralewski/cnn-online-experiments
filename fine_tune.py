import paho.mqtt.client as mqtt
import time
import json
import os
import numpy as np
import base64
import struct
import sys
import torch
from scipy import signal

from circular_queue import CircularQueue
from put_emg_gestures_classification.pegc.models import Resnet1D
from put_emg_gestures_classification.pegc import constants
from put_emg_gestures_classification.pegc.training.utils import load_json
from pegc.training.lookahead import Lookahead

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
    # ***** GLOBAL VARIABLES NIGHTMARE *****
    global file_created
    global script_start_time
    global last_sampling_rate_print_time
    global samples_counter

    global gestures_classes
    global current_gesture_idx
    global current_gesture_start_time
    global gest_duration
    global time_for_gest_change
    global window_length

    global buffer
    global model
    global optimizer
    global loss_fnc

    global model_path
    # ********* ******* ***********

    # Calculate duration of the current gesture
    curr_gest_dur = time.time() - current_gesture_start_time

    # Change a gesture if time for it elapsed and say it
    if curr_gest_dur > gest_duration + time_for_gest_change:
        current_gesture_idx += 1
        current_gesture_start_time = time.time()
        if current_gesture_idx >= len(gestures_classes):
            # Stop subscriber loop
            print("Recording ends, unsubscribing...")
            client.loop_stop()

            # Save model
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, model_path)

            print("Program ends...")
            sys.exit()

        # Tell user about the gesture
        os.system(f'spd-say "{gestures_classes[current_gesture_idx]}"')

    # Decode a message
    message_as_string = str(message.payload.decode("utf-8"))
    message_as_string = ''.join(message_as_string.split())  # remove all white characters

    # Decode the JSON
    packet = json.loads(message_as_string)

    # Read number of channels
    num_channels = packet["channels"]

    # Add a number of samples to the counter
    samples_counter += packet["packets"]

    # Save data in circular buffer if time for transition elapsed
    if curr_gest_dur > time_for_gest_change:
        for encoded_channels in packet["data"]:
            if len(encoded_channels) == num_channels * BASE64_BYTES_PER_SAMPLE:
                # Decode data
                channels_list = base64_to_list_of_channels(encoded_channels, BASE64_BYTES_PER_SAMPLE)
                scaled_data = np.asarray(channels_list).T * SCALE_FACTOR_EMG
                # Save decoded data to the buffer
                buffer.push(scaled_data)

    # Print the number of samples received
    time_elapsed = time.time() - last_sampling_rate_print_time
    if time_elapsed >= 1.0:

        print("***************************")
        print("Samples received: " + str(samples_counter))
        print("Time from the last print: " + str(time_elapsed))
        print("***************************")

        last_sampling_rate_print_time = time.time()
        samples_counter = 0

    # If data for the window is collected
    if buffer.get_size() >= window_length:

        # get data from the buffer
        data_from_buffer = buffer.get_window_of_data(window_length)
        # filtering
        expanded_data = np.concatenate((data_from_buffer,) * 3)  # triple data length for filtering
        filtered_data = filter_signals(expanded_data, EMG_FREQUENCY)

        # Transpose and expand dimension because CNN input=(batch, channels, data_window)
        training_data = np.expand_dims(filtered_data[-200:].T, axis=0)
        # fine tuning
        y_pred = model(torch.Tensor(training_data))
        y = torch.Tensor([int(gestures_classes[current_gesture_idx] == x) for x in gestures_classes])
        loss = loss_fnc(y_pred, y)
        start = time.time()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stop = time.time()

        # print time and loss
        print('****************** '
              f'Fine tuning time: {stop - start} '
              f'Loss: {loss:.4f} '
              '******************')


# MQTT data
broker_data = {
  "broker_address": "192.168.9.100"
}
topic = "sensors/emg/data"

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

# Optimizer setup.
base_opt = torch.optim.Adam(model.parameters(), lr=training_config["base_lr"])
optimizer = Lookahead(base_opt, k=5, alpha=0.5) if training_config["use_lookahead"] else base_opt

# Loss function
loss_fnc = torch.nn.MultiLabelSoftMarginLoss(reduction='mean')

# Load model and optimizer from checkpoint
model_path = "openbci_third_trained_model.tar"
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Ensure these layers are in training mode
model.train()

# Prepare user for tuning...
os.system(f'spd-say "Fine tuning starts in"')
time.sleep(2)
for n in ('three', 'two', 'one'):
    os.system(f'spd-say "{n}"')
    time.sleep(1)

# Window length
window_length = 200  # ms

# Script start time
script_start_time = time.time()

# The time of last sampling rate print
last_sampling_rate_print_time = script_start_time

# List of gestures
gestures_classes = ['idle', 'fist', 'flexion', 'extension', 'pinch_index', 'pinch_middle',
                    'pinch_ring', 'pinch_small']
gest_duration = 3
time_for_gest_change = 1

# Current gesture
current_gesture_idx = 0
os.system(f'spd-say "{gestures_classes[current_gesture_idx]}"')

# Time of current gesture beginning
current_gesture_start_time = time.time()

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
