import warnings
import flwr as fl
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import utils

if __name__ == "__main__":
    # Load persona 1 data (impulsivemax)
    (X_train, y_train), (X_test, y_test) = utils.load_data_client5()

    # Split train set into 5 partitions and randomly choose one for this round
    partition_id = np.random.choice(5)
    X_train, y_train = utils.partition(X_train, y_train, 5)[partition_id]

    # Create a Logistic Regression model
    model = LogisticRegression(
        solver='saga',
        penalty='l2',
        max_iter=1,      # local epoch count
        warm_start=True, # retain previous weights on .fit()
    )

    # Initialize model parameters so Flower can request them
    utils.set_initial_params(model)

    # Define Flower client
    class PersonaClient(fl.client.NumPyClient):
        def get_parameters(self):  # type: ignore
            return utils.get_model_parameters(model)

        def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            # Suppress warnings from low-max_iter
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            print(f"Training finished for round {config['rnd']}")
            return utils.get_model_parameters(model), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            preds = model.predict_proba(X_test)
            loss = log_loss(y_test, preds, labels=[0, 1])
            accuracy = model.score(X_test, y_test)
            return loss, len(X_test), {"accuracy": accuracy}

    # Start Flower client and connect to server
    fl.client.start_numpy_client(
        server_address="localhost:5040",  # replace with your server address if different
        client=PersonaClient()
    )
