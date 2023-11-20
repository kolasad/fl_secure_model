# Federated Learning Secure Model Repository
## Introduction
The Federated Learning Secure Model Repository (FLSMR) is an innovative software solution designed to enhance the security and integrity of machine learning models in federated learning scenarios. FLSMR employs algorithms to identify and filter out malicious contributions from clients, ensuring the reliability of the shared model ecosystem.

## Motivation
Federated learning allows collaborative model training across distributed data sources. However, the decentralized nature of this approach presents challenges related to data privacy, model security, and adversarial attacks. FLSMR addresses these concerns by providing a secure repository for sharing models among nodes, preserving data privacy, and preventing malicious contributions.

## Usage
Using Python 3.9.13 install the required dependencies by running:
```
cd src
pip install -r requirements.txt
```
Inside the `src` directory. Run the `run.py` script to initiate the federated learning process with the following customizable parameters:
* rounds: Number of communication rounds.
* learning_rate: Learning rate for optimization.
* model: Choose the model architecture (default: SimpleMLP).
* clients_number: Number of clients participating in the process.
* model_name: Name of the model (optional).
* client_single_round_epochs_num: Number of epochs per client per communication round.
* corrupt_data_clients_num: Number of clients with shuffled/corrupted data.
* dataset: Use a custom dataset if available.
* detection_method: Malicious clients detection method.

Process evaluates the model's accuracy and loss after each communication round saving resulting logs from each client and a global model.

## Usage examples

### Basic usage with default parameters
```
cd src
python run.py
```


### Usage with custom model and dataset
```
start_train_test(
    model=CustomModel,  # custom model class
    dataset="custom_dataset_directory",  # Path to the custom dataset
    learning_rate=0.001,
    rounds=30,
    clients_number=100,  # number of Federated Learning participants in each round
    client_single_round_epochs_num=1,  # number of epochs by each client
    corrupt_data_clients_num=2,  # number of corrupted clients
)
```

### Usage with custom detection method 
Define a custom detection method as
```
def custom_detection(weights=None, accuracies=None, **kwargs):
    # Implement your detection logic here
    # You can use accuracies and weights to filter out malicious clients
    return scaled_weights_list_filtered
```
start training with the provided method
```
start_train_test(
    model=MLPModel,
    learning_rate=0.001,
    rounds=50,
    detection_method=custom_detection
)
```

# License
The Federated Learning Secure Model Repository is provided under the MIT License.
