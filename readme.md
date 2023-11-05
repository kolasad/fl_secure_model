# Federated Learning Secure Model Repository
## Introduction
The Federated Learning Secure Model Repository (FLSMR) is an innovative software solution designed to enhance the security and integrity of machine learning models in federated learning scenarios. FLSMR employs algorithms to identify and filter out malicious contributions from clients, ensuring the reliability of the shared model ecosystem.

## Motivation
Federated learning allows collaborative model training across distributed data sources. However, the decentralized nature of this approach presents challenges related to data privacy, model security, and adversarial attacks. FLSMR addresses these concerns by providing a secure repository for sharing models among nodes, preserving data privacy, and preventing malicious contributions.

## Usage
Install the required dependencies by running:
```
pip install -r requirements.txt
```
Run the `run.py` script to initiate the federated learning process with the following customizable parameters:
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

### readme 2

The Federated Learning Secure Model Repository is a versatile framework designed to ensure the security and integrity of machine learning models generated through federated learning. It employs advanced algorithms for real-time detection of malicious contributions and offers a comprehensive set of functionalities to enable effective model auditing and collaborative model training in a secure environment.

Table of Contents
Overview
Software Functionalities
Getting Started
Usage
Contributing
License
Overview
Federated Learning is a distributed machine learning paradigm where multiple clients collaboratively train a shared global model while keeping their data localized. The Federated Learning Secure Model Repository addresses the challenge of malicious contributions from clients that may attempt to undermine the integrity of the shared model. This repository offers innovative solutions to detect and mitigate malicious contributions, enhancing the trustworthiness of the collaborative learning process.

Software Functionalities
The Federated Learning Secure Model Repository provides a set of key functionalities:

Data Loading: Supports dataset loading from local disks and integration with popular dataset frameworks like TensorFlow datasets.
Client Creation: Creates individual client instances with shuffled and potentially corrupted data, simulating real-world variations in data quality.
Model Building: Includes MLP (Multi-Layer Perceptron) and CNN (Convolutional Neural Network) model architectures, with provisions for integrating custom models.
Malicious Contribution Detection: Employs an accuracy-based filter and weights comparison detection to identify and filter malicious contributions.
Communication Rounds: Manages multiple communication rounds, updating the global model based on client contributions.
Logging and Reporting: Logs accuracy, loss, and other metrics for analysis, enabling monitoring and reporting of model performance.
Getting Started
To get started with the Federated Learning Secure Model Repository, follow these steps:

Install the required dependencies as listed in the code file.
Configure the parameters according to your specific use case.
Run the provided sample code to witness the functionalities of the repository in action.
Usage
The repository's functionalities can be customized and extended based on your requirements. Here's how you can leverage the provided functionalities:

Data Loading: Load your dataset or integrate with existing datasets to simulate realistic client instances.
Client Creation: Modify the creation of client instances and data shuffling to replicate real-world scenarios.
Model Building: Use the provided MLP and CNN models or integrate your custom model architectures.
Malicious Contribution Detection: Experiment with detection methods, including accuracy-based filters, to identify and filter malicious contributions.
Communication Rounds: Observe how the global model updates based on client contributions over multiple communication rounds.
Logging and Reporting: Analyze the logged metrics to monitor and assess the performance of the shared model.
Contributing
We welcome contributions to enhance and extend the Federated Learning Secure Model Repository. If you'd like to contribute, please follow these steps:

Fork the repository.
Create a new branch for your feature or improvement.
Make your changes and ensure they adhere to the coding standards.
Submit a pull request, describing the changes you've made.
License
The Federated Learning Secure Model Repository is provided under the MIT License,




## Usage examples

### Basic usage with default parameters
```
start_train_test(model=MLPModel)
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
