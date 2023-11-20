import os

import tensorflow as tf
from sklearn.model_selection import train_test_split


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