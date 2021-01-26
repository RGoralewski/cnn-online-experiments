import paho.mqtt.client as mqtt
import time

import tellopy
import collections


def handler(event, sender, data, **args):
    global drone
    drone = sender
    if event is drone.EVENT_FLIGHT_DATA:
        print(data)


def stop_drone():
    global drone
    drone.forward(0)
    drone.up(0)
    drone.down(0)
    drone.counter_clockwise(0)
    drone.clockwise(0)


def on_message(client, userdata, message):
    """Callback function
    """
    global memory
    global in_the_air
    global drone

    # Decode a message
    message_as_string = str(message.payload.decode("utf-8"))
    message_as_string = ''.join(message_as_string.split())  # remove all white characters

    print(message_as_string)

    '''
    # Save in memory
    memory.rotate(-1)
    memory[-1] = gesture

    # Take off / land
    if memory.count('pinch thumb-small') == 5:
        if not in_the_air:
            drone.takeoff()
            in_the_air = True
        else:
            drone.land()
            in_the_air = False

    # Palm landing
    if memory.count('pinch thumb-ring') == 5:
        if in_the_air:
            drone.palm_land()
            in_the_air = False

    # Move
    dynamism = 30
    if memory[-1] == memory[-2]:
        if memory[-1] == 'fist':
            drone.forward(dynamism)
        elif memory[-1] == 'pinch thumb-index':
            drone.up(dynamism)
        elif memory[-1] == 'pinch thumb-middle':
            drone.down(dynamism)
        elif memory[-1] == 'flexion':
            drone.counter_clockwise(dynamism)
        elif memory[-1] == 'extension':
            drone.clockwise(dynamism)
        else:
            stop_drone()
    '''


# MQTT data
broker_data = {
  "broker_address": "192.168.9.119"
}
topic = "ClassificationResults"


# Create drone
drone = tellopy.Tello()
#drone.connect()
#drone.wait_for_connection(60.0)
#drone.subscribe(drone.EVENT_FLIGHT_DATA, handler=handler)

# Create memory of last 5 gestures
memory = collections.deque(['idle'] * 5)

# In the air flag
in_the_air = False

# Create a client
print("Creating new subscriber instance")
client = mqtt.Client("emg_sub")

# Type username and password before connecting to broker
# client.username_pw_set(broker_data['username'], broker_data['password'])

# Connect with broker
print("Connecting to broker")
client.connect(broker_data['broker_address'])

# Turn on the interface
print("Publishing interfaces/use...")
client.publish('interfaces/use', '{"interface":"emg"}')
time.sleep(5)

# Turn on data streaming from sensor
print("Publishing command/controlSensor...")
client.publish('command/controlSensor', '{"command":"start"}')
time.sleep(5)

# Turn on the predict mode
print("Publishing command/useMode...")
client.publish('command/useMode', 'predict')
time.sleep(5)

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

    # Disconnect with drone
    drone.quit()
