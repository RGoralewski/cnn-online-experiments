import paho.mqtt.client as mqtt
import time
import os
import os.path as osp
import numpy as np
import sys

from emg.emglimbo import EMG_Preprocessor
from emg.emglimbo import EMG_Classifier


def on_message(client, userdata, message):
    """Callback function
    """
    # ***** GLOBAL VARIABLES NIGHTMARE *****
    global gestures_classes
    global current_gesture_idx
    global current_gesture_start_time
    global gest_duration
    global time_for_gest_change

    global preprocessor
    global classifier
    global buffer
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

            # Fine tune the model
            acc_b4, acc_aft = classifier.train(buffer['features'], buffer['labels'])
            print(f"Acc b4: {acc_b4:.2f}, Acc aft: {acc_aft:.2f}")

            print("Program ends...")
            sys.exit()

        # Tell user about the gesture
        os.system(f'spd-say "{gestures_classes[current_gesture_idx]}"')

    # Decode a message
    message_as_string = str(message.payload.decode("utf-8"))
    message_as_string = ''.join(message_as_string.split())  # remove all white characters

    # Preprocess the message if time for gesture change elapsed
    if curr_gest_dur >= time_for_gest_change:
        samples = preprocessor.process(message_as_string)
        for sample in samples:
            preprocessor.append_to_buffer(sample)

            # If there's a window of data collected (including stride) than get the features vector
            if preprocessor.check_buffered_data_size():
                features = preprocessor.get_features()  # returns 1D vector with features

                # Label
                label = gestures_classes[current_gesture_idx]

                # Append to the buffer
                buffer['features'].append(features)
                buffer['labels'].append(label)


# MQTT data
broker_data = {
  "broker_address": "192.168.9.119"
}
topic = "sensors/data/emg"

# Create classifier model
model_path = "emg/model.tar"
classifier = EMG_Classifier(model_path, fake=False)

# Preprocessor
window = 300000
stride = 200000
preprocessor = EMG_Preprocessor(window, stride)

# Lists to collect features and labels
buffer = {'features': [], 'labels': []}

# Prepare user for tuning...
os.system(f'spd-say "Fine tuning starts in"')
time.sleep(2)
for n in ('three', 'two', 'one'):
    os.system(f'spd-say "{n}"')
    time.sleep(1)

# List of gestures
gestures_classes = ['idle', 'fist', 'flexion', 'extension', 'pinch_thumb-index',
                    'pinch_thumb-middle', 'pinch_thumb-ring', 'pinch_thumb-small']
gest_duration = 3.5
time_for_gest_change = 1

# Current gesture
current_gesture_idx = 0
os.system(f'spd-say "{gestures_classes[current_gesture_idx]}"')

# Time of current gesture beginning
current_gesture_start_time = time.time()

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
