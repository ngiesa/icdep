from torch.nn import Module, ModuleList, Linear, ReLU
from torch.nn import BCELoss, BCEWithLogitsLoss
from torch import Tensor
from torch import long, exp

# weighted loss function
def custom_weighted_bce_loss_logits(targets, pos_weight):
    # multiply with target tensor
    w = targets * pos_weight
    # replace zeros by ones
    w[w == 0] = 1
    # return weighted BCE loss
    return BCEWithLogitsLoss(weight=w)


# focal loss function
def focal_loss_logits(alpha=0.09, gamma=1, pred=Tensor(), targets=Tensor()):
    alpha = Tensor([alpha, 1 - alpha])
    loss = BCEWithLogitsLoss(reduction="none")
    BCE_loss = loss(pred, targets)
    targets = targets.type(long)
    at = alpha.gather(0, targets.data.view(-1))
    pt = exp(-BCE_loss)
    F_loss = at * (1 - pt) ** gamma * BCE_loss
    return F_loss.mean()


# model definition module is the root class for all neural networks
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs, n_nodes):
        super(MLP, self).__init__()
        # add list of layers
        self.layers = ModuleList()
        # init the concurrent input nodes
        next_input = n_inputs
        # fill list with hidden layers
        for nodes in n_nodes:
            self.layers.append(Linear(next_input, nodes))
            next_input = nodes
            self.layers.append(ReLU())
        # assign final output layer
        self.layers.append(Linear(next_input, 1))
        # self.layers.append(Sigmoid()) #

    # forward propagation
    def forward(self, X):
        # concatenate data through layers
        for layer in self.layers:
            X = layer(X)
        # return result
        return X

    # calculate the loss
    def calc_loss(self, pred, targets, pos_weight, loss_type):
        if loss_type == "focal":
            loss = focal_loss_logits(
                alpha=pos_weight / 100, gamma=2, pred=pred, targets=targets
            )
        if loss_type == "bce":
            criterion = custom_weighted_bce_loss_logits(targets, pos_weight)
            # criterion = custom_weighted_BCELoss(targets, pos_weight)
            loss = criterion(pred, targets)
        return loss

    # implementation of early stopping method patience as number of eapochs to wait till validation loss incerases/ decreases
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

