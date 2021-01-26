import paho.mqtt.client as mqtt
import time
import os

import numpy as np
import sys



def on_message(client, userdata, message):
    """Callback function
    """
    # ***** GLOBAL VARIABLES NIGHTMARE *****
    global gestures_classes
    global current_gesture_idx
    global current_gesture_start_time
    global gest_duration
    global time_for_gest_change
    # ********* ******* ***********

    # Calculate duration of the current gesture
    curr_gest_dur = time.time() - current_gesture_start_time

    # Change a gesture if time for it elapsed and say it
    if curr_gest_dur > gest_duration:
        current_gesture_idx += 1
        current_gesture_start_time = time.time()
        if current_gesture_idx >= len(gestures_classes):


            # Stop subscriber loop
            print("Recording ends, unsubscribing...")
            client.loop_stop()

            print("Program ends...")
            sys.exit()

        # Tell user about the gesture
        os.system(f'spd-say "{gestures_classes[current_gesture_idx]}"')

    # Decode a message
    message_as_string = str(message.payload.decode("utf-8"))
    message_as_string = ''.join(message_as_string.split())  # remove all white characters




# MQTT data
broker_data = {
  "broker_address": "192.168.9.119"
}
topic = "sensors/data/emg"

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


# Prepare user for tuning...
os.system(f'spd-say "Fine tuning starts in"')
time.sleep(2)
for n in ('three', 'two', 'one'):
    os.system(f'spd-say "{n}"')
    time.sleep(1)

# List of gestures
gestures_classes = ['idle', 'fist', 'flexion', 'extension', 'pinch_thumb-index',
                    'pinch_thumb-middle', 'pinch_thumb-ring', 'pinch_thumb-small'] * 2
gest_duration = 3.5
time_for_gest_change = 1

# Current gesture
current_gesture_idx = 0
os.system(f'spd-say "{gestures_classes[current_gesture_idx]}"')

# Time of current gesture beginning
current_gesture_start_time = time.time()

try:
    # Start the loop
    client.loop_forever()

except KeyboardInterrupt:
    # Stop subscriber loop
    client.loop_stop()
