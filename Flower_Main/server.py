import flwr as fl
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import utils
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerConfig

# ---------------- Helper callbacks ----------------

def fit_round(rnd: int):
    return {"rnd": rnd}


def get_eval_fn(model: LogisticRegression):
    """Aggregate all persona test splits into one evaluation set."""

    loaders = [
        utils.load_data_client1,
        utils.load_data_client2,
        utils.load_data_client3,
        utils.load_data_client4,
        utils.load_data_client5,
        utils.load_data_client6,
        utils.load_data_client7,
        utils.load_data_client8,
        utils.load_data_client9,
        utils.load_data_client10,
        utils.load_data_client11,
    ]

    X_tests, y_tests = [], []
    for fn in loaders:
        _, (Xt, yt) = fn()
        X_tests.append(Xt)
        y_tests.append(yt)

    X_test = np.vstack(X_tests)
    y_test = np.concatenate(y_tests)

    def evaluate(server_round: int, parameters, _config):
        utils.set_model_params(model, parameters)
        proba = model.predict_proba(X_test)
        loss = float(log_loss(y_test, proba, labels=[0, 1]))
        accuracy = float(model.score(X_test, y_test))
        if server_round == 10:  # save once at the final round
            pd.DataFrame({"prob_0": proba[:, 0], "prob_1": proba[:, 1], "true": y_test}).to_csv(
                "prediction_results.csv", index=False
            )
        return loss, {"accuracy": accuracy}

    return evaluate


if __name__ == "__main__":
    model = LogisticRegression(solver="saga", penalty="l2", max_iter=1, warm_start=True)
    utils.set_initial_params(model)
    initial_parameters = ndarrays_to_parameters(utils.get_model_parameters(model))

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,          # train on all connected clients each round
        fraction_evaluate=0.0,     # we do server‑side evaluation only
        min_fit_clients=11,
        min_available_clients=11,
        evaluate_fn=get_eval_fn(model),
        on_fit_config_fn=fit_round,
        initial_parameters=initial_parameters,
    )

    fl.server.start_server(
        server_address="localhost:5040",
        config=ServerConfig(num_rounds=10),
        strategy=strategy,
    )



# import flwr as fl
# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import log_loss

# import utils
# from flwr.common import ndarrays_to_parameters
# from flwr.server import ServerConfig

# # ---------- Helper functions ----------

# def fit_round(rnd: int):
#     """Pass the current round number to each client."""
#     return {"rnd": rnd}

# #round_metrics = [] 
# def get_eval_fn(model: LogisticRegression):
#     """Return an evaluation function (server-side)."""

#     # Combine the two persona test splits so the global model is assessed on both
#     _, (X_test1, y_test1) = utils.load_data_client1()
#     _, (X_test2, y_test2) = utils.load_data_client2()
#     X_test = np.vstack([X_test1, X_test2])
#     y_test = np.concatenate([y_test1, y_test2])

#     def evaluate(server_round: int, parameters, config):  # <-- 3‑arg signature!
#         # Update local copy of the model with aggregated weights
#         utils.set_model_params(model, parameters)

#         # Predict probabilities and compute metrics
#         proba = model.predict_proba(X_test)
#         loss = float(log_loss(y_test, proba, labels=[0, 1]))  # Flower expects float
#         accuracy = float(model.score(X_test, y_test))

#         # Optional: save predictions for manual inspection
#         pd.DataFrame({"prob_0": proba[:, 0], "prob_1": proba[:, 1], "true": y_test}).to_csv(
#             "prediction_results.csv", index=False
#         # store for later analysis ➋
#        # round_metrics.append(
#         #    {"round": server_round, "loss": loss, "accuracy": accuracy})    
#         )

#         return loss, {"accuracy": accuracy}

#     return evaluate

# # ---------- Main ----------

# if __name__ == "__main__":
#     # Base (global) model definition
#     model = LogisticRegression(solver="saga", penalty="l2", max_iter=1, warm_start=True)
#     utils.set_initial_params(model)

#     # Turn zero‑weights into Flower parameters
#     initial_parameters = ndarrays_to_parameters(utils.get_model_parameters(model))

#     # FedAvg strategy with our evaluation logic
#     strategy = fl.server.strategy.FedAvg(
#         min_available_clients=2,
#         evaluate_fn=get_eval_fn(model),
#         on_fit_config_fn=fit_round,
#         initial_parameters=initial_parameters,
#     )

#     # Launch server (still deprecated API, but functional)
#     fl.server.start_server(
#         server_address="localhost:5040",
#         config=ServerConfig(num_rounds=10),
#         strategy=strategy,
#     )
#     # ➌  save after the blocking call returns
#     # import pandas as pd
#     # pd.DataFrame(round_metrics).to_csv("training_metrics.csv", index=False)
#     # print("Round‑wise metrics saved to training_metrics.csv")
