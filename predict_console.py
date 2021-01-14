import paho.mqtt.client as mqtt
import operator
import numpy as np

from emg.emglimbo import EMG_Preprocessor
from emg.emglimbo import EMG_Classifier


def on_message(client, userdata, message):
    """Callback function
    """
    global samples_counter

    global preprocessor
    global classifier

    # Decode a message
    message_as_string = str(message.payload.decode("utf-8"))
    message_as_string = ''.join(message_as_string.split())  # remove all white characters

    # Process message
    packets = preprocessor.process(message_as_string)

    # Add to buffer checking if there's enough data to predict
    for packet in packets:
        preprocessor.append_to_buffer(packet)
        if preprocessor.check_buffered_data_size():
            features = preprocessor.get_features()

            # Predict a gesture and print
            prob_dist = classifier.classify(features)

            print(f"***********\n"
                  f"Gesture = {max(prob_dist.items(), key=operator.itemgetter(1))[0]}\n"
                  f"PD = {prob_dist}\n"
                  "***********")


# MQTT data
broker_data = {
  "broker_address": "192.168.9.100"
}
topic = "sensors/data/emg"

# File created flag
file_created = False

# Preprocessor
window = 200000
stride = 200000
preprocessor = EMG_Preprocessor(window, stride)

# Create classifier model
model_path = "openbci_third_trained_model.tar"
classifier = EMG_Classifier(model_path, fake=False)

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
