from torch.optim import Adam
import pandas as pd
from torch import cuda, from_numpy
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import numpy as np
from sklearn.metrics import log_loss


def train_mlp_model(
    train_df,
    test_df,
    model,
    learning_rate,
    pos_weight,
    loss_type="bce",
    gamma=0,
    l1_regularization=False,
    l2_regularization=False,
    patience=10,
    max_epoch=500,
    lamda=1e-4
):
    """
    implements the mlp training process via early stopping, optimization and regilarization functions
            Parameters:
                        train_df (DataLoader): training data, test_df (DataLoader), model (nn.Modul),
                        learning_rate (float): step size for optimizer in gradient decent, pos_weight (minority class weight),
                        loss_type (string) either "bce" or "focal", gamma (int): hyperparameter for focal loss,
                        patience (int): max epoch with val loss increase for early stopping, max_epoch (int): max number of training epochs,
                        l1_regularization (bool): flag for l1 norm, l2_regularization (bool): flag for l2_norm,
                        lamda (float): hyperparameter for regularization terms
            Returns:
                    training_loss (float), validation_loss (float), model (nn.Modul)
    """
    # store training and validation losses
    training_loss, validation_loss = [], []
    # configure l2 norm if required
    weight_decay = 0
    if l2_regularization:
        weight_decay = lamda
    # define the optimization
    optimizer = Adam(
        # weight_decay
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    # enumerate epochs until early stopping
    for epoch in range(max_epoch):
        # enumerate mini batches
        training_loss_epoch = 0
        for i, (inputs, targets) in enumerate(train_df):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output yhat
            pred = model(inputs).reshape(-1)
            # calculate the loss
            loss = model.calc_loss(pred, targets, pos_weight, loss_type, gamma)
            # apply L1 norm applied on first layer for feature selection process
            if l1_regularization:
                l1_norm = sum(abs(p).sum()
                              for p in model.layers[0].parameters())
                loss = loss + lamda * l1_norm
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
        loss = model.calc_loss(pred, targets, pos_weight, loss_type, gamma)
        # append loss to validation loss list
        validation_loss.append(loss.detach().numpy())
        stop_early = model.early_stop(
            curr_validation_loss=validation_loss, patience=patience
        )
        if stop_early | (epoch == max_epoch):
            return training_loss, validation_loss, model
    return training_loss, validation_loss, model


def load_train_test_dl(batch_size, X_train, X_test, y_train, y_test):
    """
    creates torch specific data element for training
            Parameters:
                    batch_size (int): size of batches, X_train (array like or df): features in train set, X_test (array like or df): features in test set,
                    y_train (array like or df): labels in training set, y_test (array like or df): labels in testing set
            Returns:
                    train_df (DataLoader), test_df (DataLoader)
    """
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


def train_tree_model(data: dict, config: dict, pos_weight: float):
    """
    training function for tree based ml models (random forest or boosted trees), used by hyperparameter search algorithm
            Parameters:
                    data (dict): dict holding X_train, X_test, y_train, y_test as keys, 
                    config (dict): dict with hyperparameters to test against, pos_weight (float): weight for minority class
            Returns:
                    training_loss (float), validation_loss (float), model (sklear model), 
                    predictions_test (array): probabilities predicted by model on test set,
                    predictions_train (array): probabilities predicted by model on train set,
    """
    # configure random forest model
    if config["tree_type"] == "random_forest":
        model = RandomForestClassifier(n_estimators=config["n_estimators"],
                                       max_depth=config["max_depth"],
                                       min_samples_split=config["min_sample_split"],
                                       random_state=0, class_weight="balanced")
        model.fit(np.array(data["X_train"]), np.array(data["y_train"]))
    # configure boosted trees and weight samples accordingly
    if config["tree_type"] == "boosted_trees":
        weights = np.array(data["y_train"]) * pos_weight
        weights[weights == 0.0] = 1
        model = GradientBoostingClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_split=config["min_sample_split"],
            n_iter_no_change=10,
            tol=0.01
        )
        model.fit(np.array(data["X_train"]), np.array(
            data["y_train"]), sample_weight=weights)
    # make predictions on train and test sets and report loss
    predictions_test = model.predict(np.array(data["X_test"]))
    predictions_train = model.predict(np.array(data["X_train"]))
    validation_loss = log_loss(y_true=np.array(
        data["y_test"]), y_pred=np.array(predictions_test))
    training_loss = log_loss(y_true=np.array(
        data["y_train"]), y_pred=np.array(predictions_train))
    return training_loss, validation_loss, model, predictions_test, predictions_train
