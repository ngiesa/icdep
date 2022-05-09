from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from numpy.random import seed
from pandas.core.frame import DataFrame
import random
from keras.metrics import AUC
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import tensorflow_addons as tfa

from focal_loss import BinaryFocalLoss


class SequentialModel:

    def __init__(self,
                 df_X_train: DataFrame = None,
                 activation: str = "sigmoid",
                 optimizer="Adam",
                 drop_out: float = 0.0,
                 l1_reg_factor: float = 0.01,
                 hidden_nodes: list = [],
                 gamma: int = 2,
                 loss: str = "focal"):
        self.df_X_train = df_X_train
        self.activation = activation
        self.optimizer = optimizer
        self.hidden_nodes = hidden_nodes
        self.drop_out = drop_out
        self.l1_reg_factor = l1_reg_factor
        self.loss = loss
        self.gamma = gamma
        self.model = self.__create_model()

    def __create_model(self):

        # enable accelerated linear algebra
        tf.config.optimizer.set_jit(True)

        # set random seed for reproducibility
        random.seed(42)

        # sequential model
        model = Sequential()

        # specifying the input shape of the NN
        model.add(tf.keras.Input(shape=(
            self.df_X_train.shape[1],)))

        # create additional hidden layers
        for node in self.hidden_nodes:
            model.add(Dense(node,
                            activation=self.activation,
                            activity_regularizer=tf.keras.regularizers.L1(
                                self.l1_reg_factor)
                            ))

        # add dropout, default is none
        model.add(Dropout(self.drop_out))

        # create output layer for binary output
        model.add(Dense(1,
                        activation="sigmoid",
                        activity_regularizer=tf.keras.regularizers.L1(
                            self.l1_reg_factor)
                        ))

        metrics = [tf.keras.metrics.AUC(curve="ROC"),
                   tf.keras.metrics.AUC(curve="PR"),
                   tf.keras.metrics.Precision(),
                   tf.keras.metrics.Recall(),
                   tf.keras.metrics.SpecificityAtSensitivity(0.8),
                   tf.keras.metrics.SensitivityAtSpecificity(0.8)]

        # crossentropy is focal loss with gamma equal to zero
        if self.loss == "crossentropy":
            self.gamma = 0

        # define focal loss, with gamma = 0 it becomes cross entropy
        loss = BinaryFocalLoss(gamma=self.gamma)

        # compile the model
        model.compile(optimizer=self.optimizer,
                      loss=loss,
                      metrics=metrics)

        return model
