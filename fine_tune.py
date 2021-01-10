import paho.mqtt.client as mqtt
import time
import os
import numpy as np
import sys
import torch

# TODO remove put_emg_gestures submodule and add only pegc submodule

from put_emg_gestures_classification.pegc.models import Resnet1D
from put_emg_gestures_classification.pegc import constants
from put_emg_gestures_classification.pegc.training.utils import load_json
from put_emg_gestures_classification.pegc.training.lookahead import Lookahead

from emg.emglimbo import EMG_Preprocessor
from emg.emglimbo import constants


def on_message(client, userdata, message):
    """Callback function
    """
    # ***** GLOBAL VARIABLES NIGHTMARE *****
    global file_created
    global script_start_time

    global gestures_classes
    global current_gesture_idx
    global current_gesture_start_time
    global gest_duration
    global time_for_gest_change

    global preprocessor
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


            # TODO do fine tuning in the end -> epochs, test and validation dataset, etc.


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

    # Preprocess the message if time for gesture change elapsed
    if curr_gest_dur >= time_for_gest_change:
        preprocessor.preprocess(message_as_string)

    # If there's a window of data collected (including stride) than get the features vector
    if preprocessor.check_buffered_data_size():
        features = preprocessor.get_features()  # returns 1D vector with features
        features_2d = features.reshape((-1, constants.CHANNELS))
        
        # 8th channel is not connected to any electrode - copy 7th channel to it
        features_2d[:, 7] = features_2d[:, 6]

        # Transpose and expand dimension because CNN input=(batch, channels, data_window)
        training_data = np.expand_dims(features_2d.T, axis=0)

        # One hot label
        label = np.array([int(gestures_classes[current_gesture_idx] == x) for x in gestures_classes])

        # Append to the buffer
        buffer['features'].append(training_data)
        buffer['labels'].append(label)


# MQTT data
broker_data = {
  "broker_address": "192.168.9.100"
}
topic = "sensors/emg/data"

# File created flag
file_created = False

# Preprocessor
buff_cap = 300
window = 200000
stride = 200000
preprocessor = EMG_Preprocessor(window, stride, buff_cap)

# Lists to collect features and labels
buffer = {'features': [], 'labels': []}

# Create specified model.
training_config_file_path = "config_template.json"
training_config = load_json(training_config_file_path)
model = Resnet1D(constants.DATASET_FEATURES_SHAPE[0], constants.NB_DATASET_CLASSES,
                 training_config["nb_res_blocks"], training_config["res_block_per_expansion"],
                 training_config["base_feature_maps"])

# Optimizer setup.
base_opt = torch.optim.Adam(model.parameters(), lr=0.1*training_config["base_lr"])
optimizer = Lookahead(base_opt, k=5, alpha=0.5) if training_config["use_lookahead"] else base_opt

# Loss function
loss_fnc = torch.nn.MultiLabelSoftMarginLoss(reduction='mean')

# Load model and optimizer from checkpoint
model_path = "openbci_third_trained_model.tar"
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# TODO freeze middle layers

# Ensure these layers are in training mode
model.train()

# Prepare user for tuning...
os.system(f'spd-say "Fine tuning starts in"')
time.sleep(2)
for n in ('three', 'two', 'one'):
    os.system(f'spd-say "{n}"')
    time.sleep(1)

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
