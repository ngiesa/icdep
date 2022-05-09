from torch.optim import Adam
import pandas as pd
from functools import reduce
from torch import cuda, from_numpy
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from ray import tune
from modeling_layer.models.multilayer_perceptron import MLP


def train_model(
    train_df,
    test_df,
    model,
    learning_rate,
    pos_weight,
    loss_type,
    patience=5,
    max_epoch=None,
):
    # store training and validation losses
    training_loss, validation_loss = [], []
    # define the optimization
    optimizer = Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )  # TODO regularization???
    # enumerate epochs until early stopping
    if not max_epoch:
        max_epoch = 1000
    for epoch in range(1000):
        # enumerate mini batches
        training_loss_epoch = 0
        for i, (inputs, targets) in enumerate(train_df):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output yhat
            pred = model(inputs).reshape(-1)
            # calculate the loss
            loss = model.calc_loss(pred, targets, pos_weight, loss_type)
            # append loss to epoch variable
            training_loss_epoch = training_loss_epoch + loss.detach().numpy()
            # backpropagating loss
            loss.backward()
            # update model weights
            optimizer.step()
        # append epoch loss to loss training loss list
        training_loss.append(training_loss_epoch / i)
        # varify the model on validation data
        inputs = test_df.dataset.tensors[0]
        targets = test_df.dataset.tensors[1]
        # make predictions
        pred = model(inputs).reshape(-1)
        # calculate the loss
        loss = model.calc_loss(pred, targets, pos_weight, loss_type)
        # append loss to validation loss list
        validation_loss.append(loss.detach().numpy())
        stop_early = model.early_stop(
            curr_validation_loss=validation_loss, patience=patience
        )
        if stop_early | (epoch == max_epoch):
            return training_loss, validation_loss, model

    return training_loss, validation_loss, model


def load_train_test_dl(batch_size, X_train, X_test, y_train, y_test):
    device = "cuda" if cuda.is_available() else "cpu"
    # make values numeric
    X_train = X_train.apply(pd.to_numeric).fillna(0)
    X_test = X_test.apply(pd.to_numeric).fillna(0)
    # create tensors
    x_train_tensor = from_numpy(np.array(X_train)).float().to(device)
    y_train_tensor = from_numpy(np.array(y_train)).float().to(device)
    x_test_tensor = from_numpy(np.array(X_test)).float().to(device)
    y_test_tensor = from_numpy(np.array(y_test)).float().to(device)
    # create train and test dls
    train_df = DataLoader(
        TensorDataset(x_train_tensor, y_train_tensor),
        batch_size=batch_size,
        shuffle=True,
    )
    test_df = DataLoader(TensorDataset(x_test_tensor, y_test_tensor))
    # return them
    return train_df, test_df


def hyperparameter_optimization(config, data=None):
    # reassigning data
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    # setting hyperparameters
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    nodes = config["layer"] * [config["node"]]
    loss_type = config["loss_type"]
    # calculate positive weight
    pos_weight = list(y_train).count(0) / list(y_train).count(1)
    train_df, test_df = load_train_test_dl(batch_size, X_train, X_test, y_train, y_test)
    # create model
    for iteration in range(0, 1):
        model = MLP(n_inputs=X_train.shape[1], n_nodes=nodes)
        # iteration over selected configuration
        training_loss, validation_loss, _ = train_model(
            train_df, test_df, model, learning_rate, pos_weight, loss_type
        )
        # Feed the score back back to Tune.
        tune.report(
            mean_training_loss=np.mean(training_loss),
            mean_validation_loss=np.mean(validation_loss),
        )
