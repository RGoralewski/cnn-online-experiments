import paho.mqtt.client as mqtt
import time
import json


def on_message(client, userdata, message):
    """Callback function
    """
    global file_created
    global samples_counter
    global last_sampling_rate_print_time

    # Decode a message
    message_as_string = str(message.payload.decode("utf-8"))
    message_as_string = ''.join(message_as_string.split())  # remove all white characters

    # Decode the JSON
    packet = json.loads(message_as_string)

    # Add a number of samples to the counter
    samples_counter += packet["packets"]

    # Save json to the file
    with open('jsons_from_wifi_shield.txt', 'a') as f:
        f.write(message_as_string)
        f.write('\n')

    # Print the number of samples received
    time_elapsed = time.time() - last_sampling_rate_print_time
    if time_elapsed >= 1.0:

        print("***************************")
        print("Samples received: " + str(samples_counter))
        print("Time from the last print: " + str(time_elapsed))
        print("***************************")

        last_sampling_rate_print_time = time.time()
        samples_counter = 0


# MQTT data
broker_data = {
  "broker_address": "192.168.9.100"
}
topic = "sensors/data/emg"

# File created flag
file_created = False

# Script start time
script_start_time = time.time()

# The time of last sampling rate print
last_sampling_rate_print_time = script_start_time

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
