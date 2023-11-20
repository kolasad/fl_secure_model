import random


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
