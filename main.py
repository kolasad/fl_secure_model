import os
from datetime import datetime

import numpy as np
import random
from sklearn.metrics import accuracy_score

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from keras.datasets import mnist

from models import MLPModel

LOGS_DIR = 'results'


def load_dataset(dataset_dir: str):
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError("Dataset directory not found.")

    images = []
    labels = []

    # Assuming the dataset directory contains subdirectories for each class
    class_folders = sorted(os.listdir(dataset_dir))

    for class_id, class_folder in enumerate(class_folders):
        class_path = os.path.join(dataset_dir, class_folder)
        if os.path.isdir(class_path):
            for image_filename in os.listdir(class_path):
                image_path = os.path.join(class_path, image_filename)
                images.append(image_path)
                labels.append(class_id)

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    return (X_train, y_train), (X_test, y_test)


def batch_data(data_shard, bs=32):
    """Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object"""
    # seperate shard into data and labels lists
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)


def scale_model_weights(weight, scalar):
    """function for scaling a models weights"""
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final


def sum_scaled_weights(scaled_weight_list):
    """Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights"""
    avg_grad = list()
    # get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)

    return avg_grad


def test_model(X_test, Y_test, model, comm_round, log_filename='main.csv', directory='default'):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    logits = model.predict(X_test)
    loss = cce(Y_test, logits)
    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    log(log_filename, f'{comm_round},{round(acc * 100, 2)},{loss}', directory)
    return acc, loss


def acc_filter(weights=None, accuracies=None):
    avg_acc = sum(accuracies)/len(accuracies)
    scaled_local_weight_list_filtered = [
        x for n, x in enumerate(weights) if accuracies[n] > avg_acc
    ]
    # scale the model weights and add to list
    scaling_factor = 1 / len(scaled_local_weight_list_filtered)
    scaled_local_weight_list_filtered = [
        scale_model_weights(x, scaling_factor) for x in scaled_local_weight_list_filtered
    ]
    return scaled_local_weight_list_filtered


def apply_filter(filter_method, **kwargs):
    return filter_method(kwargs)


def log(filename, results, directory):
    with open(f'{LOGS_DIR}/{directory}/{filename}', 'a') as file:
        file.write(results)
        file.write('\n')


def create_clients(image_list, label_list, initial='clients', clients_num=5, shuffle_clients=0):
    # create a list of client names
    client_names = ['{}_{}'.format(initial, i + 1) for i in range(clients_num)]
    data = list(zip(image_list, label_list))
    random.shuffle(data)
    # shard data and place at each client
    size = len(data) // clients_num
    shards = [data[i:i + size] for i in range(0, size * clients_num, size)]
    # number of clients must equal number of shards
    assert (len(shards) == len(client_names))

    clients_ = dict()
    for i in range(len(client_names)):
        clients_[client_names[i]] = shards[i]

    if shuffle_clients > 0:  # randomly shuffle clients data
        for i in range(shuffle_clients):
            images = [x[0] for x in shards[i]]
            labels = [x[1] for x in shards[i]]
            random.shuffle(images)
            random.shuffle(labels)
            corrupted_shard = list(zip(images, labels))
            clients_[client_names[i]] = corrupted_shard

    return clients_


def start_train_test(
    rounds=50,
    learning_rate=0.01,
    model=MLPModel,
    clients_number=5,
    model_name: str = None,
    client_single_round_epochs_num=1,
    corrupt_data_clients_num=0,
    dataset: str = None,
    detection_method=acc_filter,

):
    # initial setup
    today = model_name if model_name else datetime.today()
    os.makedirs(f'{LOGS_DIR}/{today}')
    if not dataset:
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = [x/255 for x in X_train.reshape(60000, 784)]  # normalized train images
        new_train = []
        for label in y_train:
            empty = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            empty[label] = 1
            new_train.append(empty)

        y_train = new_train

        X_test = [x/255 for x in X_test.reshape(10000, 784)]  # normalized test images
        new_train = []
        for label in y_test:
            empty = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            empty[label] = 1
            new_train.append(empty)

        y_test = new_train
    else:
        (X_train, y_train), (X_test, y_test) = load_dataset(dataset)
        X_train = [x / 255 for x in X_train.reshape(60000, 784)]  # normalized train images
        new_train = []
        for label in y_train:
            empty = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            empty[label] = 1
            new_train.append(empty)

        y_train = new_train

        X_test = [x / 255 for x in X_test.reshape(10000, 784)]  # normalized test images
        new_train = []
        for label in y_test:
            empty = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            empty[label] = 1
            new_train.append(empty)

        y_test = new_train

    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    optimizer = SGD(
        lr=learning_rate,
        decay=learning_rate / rounds,
        momentum=0.9
    )

    # initialize global model
    smlp_global = model()
    global_model = smlp_global.build(784, 10)

    clients = create_clients(
        X_train,
        y_train,
        initial='client',
        clients_num=clients_number,
        shuffle_clients=corrupt_data_clients_num
    )

    # process and batch the training data for each client
    clients_batched = dict()
    for (client_name, data) in clients.items():
        clients_batched[client_name] = batch_data(data)

    # process and batch the test set
    test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))
    ###################

    # commence global training loop
    for comm_round in range(rounds):

        # get the global model's weights - will serve as the initial weights for all local models
        global_weights = global_model.get_weights()

        # initial list to collect local model weights after scalling
        scaled_local_weight_list = list()
        accuracies = []
        losses = []

        # randomize client data - using keys
        client_names = list(clients_batched.keys())
        random.shuffle(client_names)

        # loop through each client and create new local model
        for client in client_names:
            smlp_local = model
            local_model = smlp_local.build(784, 10)
            local_model.compile(
                loss=loss,
                optimizer=optimizer,
                metrics=metrics
            )

            # set local model weight to the weight of the global model
            local_model.set_weights(global_weights)

            # fit local model with client's data
            local_model.fit(clients_batched[client], epochs=client_single_round_epochs_num, verbose=0)

            for (X_test, Y_test) in test_batched:
                local_acc, local_loss = test_model(
                    X_test, Y_test, local_model, comm_round, log_filename=f'{client}.csv', directory=today
                )
                accuracies.append(local_acc)
                losses.append(local_loss)

            scaled_local_weight_list.append(local_model.get_weights())

            # clear session to free memory after each communication round
            K.clear_session()

        scaled_local_weight_list = apply_filter(
            detection_method,
            accuracies=accuracies,
            losses=losses,
            client_names=client_names,
            scaled_local_weight_list=scaled_local_weight_list,
            X_test=X_test,
            Y_test=Y_test
        )

        scaling_factor = 1 / len(scaled_local_weight_list)
        scaled_local_weight_list_filtered = [
            scale_model_weights(x, scaling_factor) for x in scaled_local_weight_list
        ]

        # to get the average over all the local model, we simply take the sum of the scaled weights
        average_weights = sum_scaled_weights(scaled_local_weight_list_filtered)

        # update global model
        global_model.set_weights(average_weights)

        # test global model and print out metrics after each communications round
        for (X_test, Y_test) in test_batched:
            acc_global, loss_global = test_model(
                X_test, Y_test, global_model, comm_round, directory=today
            )
            print(acc_global, loss_global)
