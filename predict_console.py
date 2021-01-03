import paho.mqtt.client as mqtt
import time
import json
import os
import numpy as np
import base64
import struct
import sys

BASE64_BYTES_PER_SAMPLE = 4
SCALE_FACTOR_EMG = 4500000/24/(2**23-1)  # uV/count
EMG_FREQUENCY = 1000  # Hz


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
            # TODO save in buffer

    # If data for the window is collected
    # TODO get data from buffer, filter it and predict (print model prediction)


# MQTT data
broker_data = {
  "broker_address": "192.168.9.100"
}
topic = "sensors/emg/data"

# File created flag
file_created = False

# TODO create circular buffer

# TODO load model

# TODO 3, 2, 1, start!

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
