from torch.nn import Module, ModuleList, Linear, ReLU, Sigmoid
from torch.nn import BCEWithLogitsLoss
from torch import Tensor
import torch.nn.functional as F
import torch

# weighted loss function


def custom_weighted_bce_loss_logits(targets, pos_weight):
    # multiply with target tensor
    w = targets * pos_weight
    # replace zeros by ones
    w[w == 0] = 1
    # return weighted BCE loss
    return BCEWithLogitsLoss(weight=w)


def sigmoid_focal_loss(
    inputs: Tensor,
    targets: Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "mean",
) -> Tensor:
    # definition of sigmoid focal loss accroding to facebook ai
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


# model definition module is the root class for all neural networks
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs, n_nodes, activation="relu"):
        super(MLP, self).__init__()
        # add list of layers
        self.layers = ModuleList()
        # init the concurrent input nodes
        next_input = n_inputs
        # fill list with hidden layers
        for nodes in n_nodes:
            self.layers.append(Linear(next_input, nodes))
            next_input = nodes
            if activation == "relu":
                self.layers.append(ReLU())
            if activation == "sig":
                self.layers.append(Sigmoid())
        # assign final output layer
        self.layers.append(Linear(next_input, 1))

    # forward propagation
    def forward(self, X):
        # concatenate data through layers
        for layer in self.layers:
            X = layer(X)
        # return result
        return X

    # calculate the loss
    def calc_loss(self, pred, targets, pos_weight, loss_type, gamma):
        if loss_type == "focal":
            loss = sigmoid_focal_loss(
                alpha=pos_weight / 100, gamma=gamma, inputs=pred, targets=targets
            )
        if loss_type == "bce":
            criterion = custom_weighted_bce_loss_logits(targets, pos_weight)
            loss = criterion(pred, targets)
        return loss

    # implementation of early stopping method patience as number of eapochs to wait till validation loss incerases
    def early_stop(self, curr_validation_loss: list = [], patience: int = 10):
        # condition that values are comparable wait until patience size is met
        if len(curr_validation_loss) < patience:
            return False
        # get first and last validation loss value
        first = curr_validation_loss[-patience]
        last = curr_validation_loss[-1]
        # if whithin the epoch comparison first value lower last value stop
        if first < last:
            print("best validation loss:", last)
            return True
        return False
