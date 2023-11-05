from fl_secure_model.models import MLPModel
from main import start_train_test

start_train_test(
    rounds=50,
    learning_rate=0.01,
    model=MLPModel,
    clients_number=100,
    model_name="test",
    client_single_round_epochs_num=1,
    corrupt_data_clients_num=1,
    dataset="mnist"
)
