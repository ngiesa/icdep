from modeling_layer.train_functions import load_train_test_dl, train_mlp_model, train_tree_model
from ray import tune
import numpy as np
from modeling_layer.models.multilayer_perceptron import MLP
from configs import config_mlp, config_trees
import pandas as pd


class HyperTuner:
    def __init__(self, current_date):
        self.df_hyperparam_res = pd.DataFrame()
        self.current_date = current_date
        self.counter = 0

    def hyperparameter_optimization(self, config, loss_type="bce", data=None, ml_type="mlp", l1_norm=False):
        # reassigning data
        gamma, lamda = 0, 0
        X_train = data["X_train"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        y_test = data["y_test"]
        if ml_type == "mlp":
            # load data
            train_df, test_df = load_train_test_dl(
                config["batch_size"], X_train, X_test, y_train, y_test)
            # setting hyperparameters
            nodes = config["layer"] * [config["node"]]
            if loss_type == "focal":
                gamma = config["gamma"]
            if l1_norm:
                lamda = config["lamda"]
        # calculate positive weight
        pos_weight = list(y_train).count(0) / list(y_train).count(1)
        # create model
        for iteration in range(0, 1):
            if ml_type == "mlp":
                model = MLP(
                    n_inputs=X_train.shape[1], n_nodes=nodes, activation=config["activation"])
                # iteration over selected configuration
                training_loss, validation_loss, _ = train_mlp_model(
                    train_df=train_df, test_df=test_df, model=model, learning_rate=config[
                        "learning_rate"],
                    pos_weight=pos_weight, loss_type=loss_type,
                    gamma=gamma, l1_regularization=l1_norm, lamda=lamda
                )
            if ml_type == "tree":
                training_loss, validation_loss, _, _, _ = train_tree_model(
                    data, config, pos_weight
                )
            # Feed the score back back to Tune.
            tune.report(
                mean_training_loss=np.mean(training_loss),
                mean_validation_loss=np.mean(validation_loss),
            )

    def execute_hyperparameter_optimization(self, loss_type, data, ml_type="mlp", l1_norm=False):

        # configure hyperparemeter search space
        config = {}
        if ml_type == "mlp":
            config = config_mlp
            if loss_type == "focal":
                config = {
                    **config,
                    "gamma": tune.choice([1, 2, 4])
                }
            if l1_norm:
                config = {
                    **config,
                    "lamda": tune.choice([1e-2, 1e-3, 1e-4])
                }

        elif ml_type == "tree":
            config = config_trees

        # execute the hyperparameter search
        analysis = tune.run(
            tune.with_parameters(
                self.hyperparameter_optimization, loss_type=loss_type, data=data, ml_type=ml_type, l1_norm=l1_norm),
            config=config,
            local_dir="/home/giesan/wd/icdep/workspace/data/metrics/hyperparameter_search",
            checkpoint_at_end=True,
            verbose=False,
        )
        best_config = analysis.get_best_config(
            metric="mean_validation_loss", mode="min"
        )
        print("Best config: ", best_config)
        # Get a dataframe for analyzing trial results.
        self.df_hyperparam_res = self.df_hyperparam_res.append(
            analysis.results_df
        )
        self.df_hyperparam_res.to_csv(
            "./data/metrics/performance/training_procedure/ray_tuner_{}_{}_revised.csv".format(
                ml_type, self.current_date
            )
        )

        return best_config
