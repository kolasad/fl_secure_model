from filters import acc_filter
from models import MLPModel
from main import start_train_test

import argparse

parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--rounds", type=int, default=50, help="Number of rounds")
parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate")
parser.add_argument("--clients-number", type=int, default=100, help="Number of clients")
parser.add_argument("--model-name", default="test", help="Name of the model used for logging")
parser.add_argument("--client-single-round-epochs-num", type=int, default=1, help="Number of epochs per round for a single client")
parser.add_argument("--corrupt-data-clients-num", type=int, default=1, help="Number of clients with corrupt data")
parser.add_argument("--dataset", default="mnist", help="Dataset name or path")

args = parser.parse_args()
config = vars(args)

start_train_test(
    model=MLPModel,
    detection_method=acc_filter,
    **config,
)
