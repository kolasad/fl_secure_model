import os
from datetime import datetime

import numpy as np
import random
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

from clients import create_clients
from filters import acc_filter
from models import MLPModel
from datasets_utils import load_dataset
from weights_utils import sum_scaled_weights, scale_model_weights

LOGS_DIR = os.path.join(os.curdir, 'results')


def test_model(X_test, Y_test, model, comm_round, log_filename='main.csv', directory='default'):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    logits = model.predict(X_test)
    loss = cce(Y_test, logits)
    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    log(log_filename, f'{comm_round},{round(acc * 100, 2)},{loss}', directory)
    return acc, loss


def apply_filter(filter_method, **kwargs):
    return filter_method(kwargs)


def log(filename, results, directory):
    with open(f'{LOGS_DIR}/{directory}/{filename}', 'a') as file:
        file.write(results)
        file.write('\n')


def start_train_test(
    rounds=50,
    learning_rate=0.01,
    model=MLPModel,
    clients_number=5,
    model_name: str = None,
    client_single_round_epochs_num=1,
    corrupt_data_clients_num=0,
    dataset: str = 'mnist',
    detection_method=acc_filter,

):
    # initial setup
    today = model_name if model_name else datetime.today()
    os.makedirs(f'{LOGS_DIR}/{today}', exist_ok=True)
    if dataset == 'mnist':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        X_train = [x/255 for x in X_train.reshape(len(X_train), X_train[0].flatten().shape[0])]  # normalized train images
        new_train = []
        for label in y_train:
            empty = np.zeros(max(y_train) + 1)
            empty[label] = 1
            new_train.append(empty)

        y_train = new_train

        X_test = [x/255 for x in X_test.reshape(len(y_test), y_test[0].flatten().shape[0])]  # normalized test images
        new_train = []
        for label in y_test:
            empty = np.zeros(max(y_test) + 1)
            empty[label] = 1
            new_train.append(empty)

        y_test = new_train
    else:
        (X_train, y_train), (X_test, y_test) = load_dataset(dataset)
        X_train = [x/255 for x in X_train.reshape(len(X_train), X_train[0].flatten().shape[0])]  # normalized train images  # normalized train images
        new_train = []
        for label in y_train:
            empty = np.zeros(max(y_train) + 1)
            empty[label] = 1
            new_train.append(empty)

        y_train = new_train

        X_test = [x/255 for x in X_test.reshape(len(y_test), y_test[0].flatten().shape[0])]  # normalized test images
        new_train = []
        for label in y_test:
            empty = np.zeros(max(y_test) + 1)
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
