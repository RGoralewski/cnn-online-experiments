import paho.mqtt.client as mqtt
import time
import os
import os.path as osp
import numpy as np
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from put_emg_gestures_classification.pegc.models import Resnet1D
from put_emg_gestures_classification.pegc import constants
from put_emg_gestures_classification.pegc.training import _epoch_train, _validate
from put_emg_gestures_classification.pegc.training.utils import load_json, \
    initialize_random_seeds, save_json
from put_emg_gestures_classification.pegc.training.lookahead import Lookahead
from put_emg_gestures_classification.pegc.generators import PUTEEGGesturesDataset
from put_emg_gestures_classification.pegc.training.clr import CyclicLR

from emg.emglimbo import constants as emgconst
from emg.emglimbo import EMG_Preprocessor

import matplotlib.pyplot as plt


def load_model(model_path: str, model: nn.Module) -> tuple:
    # Load model and optimizer from checkpoint
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def fine_tune(model_path: str, data: dict, config_file_path: str, val_split_size: float = 0.15,
              test_split_size: float = 0.1) -> None:

    # Read config from file
    fine_tuning_config = load_json(config_file_path)
    device = torch.device('cuda') if torch.cuda.is_available() and not fine_tuning_config["force_cpu"] \
        else torch.device('cpu')
    initialize_random_seeds(constants.RANDOM_SEED)

    # Load data
    features = np.array(data["features"])
    labels = np.array(data["labels"])

    assert features.shape[0] == labels.shape[0]

    # Plot every channel to see if the signals are correct
    plt_f = np.concatenate(np.swapaxes(features, 1, 2)).T
    for i in range(8):
        plt.plot(plt_f[i])
        plt.show()

    # Shuffle windows
    features, labels = unison_shuffled_copies(features, labels)

    # Divide data to train, test and validate splits
    val_size = int(val_split_size * features.shape[0])
    test_size = int(test_split_size * features.shape[0])

    X_train = features[val_size:-test_size]
    y_train = labels[val_size:-test_size]

    X_val = features[:val_size]
    y_val = labels[:val_size]

    X_test = features[-test_size:]
    y_test = labels[-test_size:]

    # Load data to generators
    train_dataset = PUTEEGGesturesDataset(X_train, y_train)
    val_dataset = PUTEEGGesturesDataset(X_val, y_val)
    test_dataset = PUTEEGGesturesDataset(X_test, y_test)
    train_gen = DataLoader(train_dataset,
                           batch_size=fine_tuning_config["batch_size"],
                           shuffle=fine_tuning_config["shuffle"])
    val_gen = DataLoader(val_dataset,
                         batch_size=fine_tuning_config["batch_size"],
                         shuffle=fine_tuning_config["shuffle"])
    test_gen = DataLoader(test_dataset,
                          batch_size=fine_tuning_config["batch_size"],
                          shuffle=fine_tuning_config["shuffle"])
    # Note: this data is quite simple, no additional workers will be required for loading/processing.

    # Create specified model.
    model = Resnet1D(constants.DATASET_FEATURES_SHAPE[0], constants.NB_DATASET_CLASSES,
                     fine_tuning_config["nb_res_blocks"], fine_tuning_config["res_block_per_expansion"],
                     fine_tuning_config["base_feature_maps"])

    # Load model
    model = load_model(model_path, model)

    # Freeze all layers except two first (convolutional and batch normalization) and the last (fully connected)
    for name, param in model.named_parameters():
        if ('res_blocks.0.' in name) or ('dense' in name):
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Optimizer setup.
    base_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), fine_tuning_config["base_lr"])
    optimizer = Lookahead(base_opt, k=5, alpha=0.5) if fine_tuning_config["use_lookahead"] else base_opt

    # Loss function
    class_shares = np.sum(y_train, axis=0) / len(y_train)
    class_weights = 1 / class_shares
    class_weights = class_weights / np.sum(class_weights)  # normalize to 0-1 range
    loss_fnc = torch.nn.MultiLabelSoftMarginLoss(reduction='mean',
                                                 weight=torch.tensor(class_weights, dtype=torch.float32).to(device))

    # LR schedulers setup.
    epochs_per_half_clr_cycle = 4
    clr = CyclicLR(optimizer, base_lr=fine_tuning_config["base_lr"], max_lr=fine_tuning_config["max_lr"],
                   step_size_up=len(train_gen) * epochs_per_half_clr_cycle,
                   mode='triangular2', cycle_momentum=False)
    schedulers = [clr]

    # Temporarily - for saving stats
    metrics = []
    results_dir_path = f"stats_{int(time.time())}"
    os.makedirs(results_dir_path, exist_ok=True)

    # Fine tuning
    start = time.time()
    for ep in range(1, fine_tuning_config["epochs"] + 1):
        epoch_stats = _epoch_train(model, train_gen, device, optimizer, loss_fnc, ep, fine_tuning_config["use_mixup"],
                                   fine_tuning_config["alpha"], schedulers)
        val_stats = _validate(model, loss_fnc, val_gen, device)
        epoch_stats.update(val_stats)

        print(f'\nEpoch {ep} train loss: {epoch_stats["loss"]:.4f}, '
              f'val loss: {epoch_stats["val_loss"]:.5f}, val_acc: {epoch_stats["val_acc"]:.4f}')

        metrics.append(epoch_stats)

    # Check results on final test/holdout set:
    test_set_stats = _validate(model, loss_fnc, test_gen, device)
    print(f'\nFinal evaluation on test set: '
          f'test loss: {test_set_stats["val_loss"]:.5f}, test_acc: {test_set_stats["val_acc"]:.4f}')

    # Save metrics/last network/optimizer state
    save_json(osp.join(results_dir_path, 'test_set_stats.json'), test_set_stats)
    save_json(osp.join(results_dir_path, 'training_losses_and_metrics.json'), {'epochs_stats': metrics})

    stop = time.time()

    # print time and loss
    print('****************** '
          f'Fine tuning time: {stop - start} '
          '******************')

    # Save model
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, model_path)


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
    global buffer
    global model_path
    global config_file_path
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
            fine_tune(model_path, buffer, config_file_path)

            print("Program ends...")
            sys.exit()

        # Tell user about the gesture
        os.system(f'spd-say "{gestures_classes[current_gesture_idx]}"')

    # Decode a message
    message_as_string = str(message.payload.decode("utf-8"))
    message_as_string = ''.join(message_as_string.split())  # remove all white characters

    # Preprocess the message if time for gesture change elapsed
    if curr_gest_dur >= time_for_gest_change:
        samples = preprocessor.preprocess(message_as_string)
        for sample in samples:
            preprocessor.append_to_buffer(sample)

            # If there's a window of data collected (including stride) than get the features vector
            if preprocessor.check_buffered_data_size():
                features = preprocessor.get_features()  # returns 1D vector with features
                features_2d = features.reshape((-1, emgconst.CHANNELS))

                # 8th channel is not connected to any electrode - copy 7th channel to it
                features_2d[:, 7] = features_2d[:, 6]

                # Transpose because CNN input=(batch, channels, data_window)
                training_data = features_2d.T

                # One hot label
                label = np.array([int(gestures_classes[current_gesture_idx] == x) for x in gestures_classes])

                # Append to the buffer
                buffer['features'].append(training_data)
                buffer['labels'].append(label)


# MQTT data
broker_data = {
  "broker_address": "192.168.9.100"
}
topic = "sensors/data/emg"

# Paths to model and config file
model_path = "openbci_third_trained_model.tar"
config_file_path = "config_fine_tuning.json"

# Preprocessor
buff_cap = 300
window = 200000
stride = 200000
preprocessor = EMG_Preprocessor(window, stride, buff_cap)

# Lists to collect features and labels
buffer = {'features': [], 'labels': []}

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
gest_duration = 3.5
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
