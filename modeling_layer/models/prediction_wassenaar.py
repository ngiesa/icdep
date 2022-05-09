# Prediction model
# Wassenaar, A., van den Boogaard, M. H. W. A., van Achterberg, T., Slooter, A. J. C., Kuiper, M. A., Hoogendoorn, M. E., ... & Pickkers, P. (2015).
# Multinational development and validation of an early prediction model for delirium in ICU patients. Intensive care medicine, 41(6), 1048-1056.

import numpy as np


class WassenaarPredictor:
    def __init__(self):
        self.inter = -3.907
        self.est_age = 0.025
        self.est_his_cog = 0.878
        self.est_alc = 0.505
        self.est_urg = 0.621
        self.est_map = -0.006
        self.est_cort = 0.283
        self.est_resp = 0.982
        self.est_bun = 0.018

    # adm cat 1 -> surgery, 2 -> medical, 3 -> trauma, 4 -> neuro/surgery
    def __get_est_adm_cat(self, cat):
        if cat == 1:
            return 0
        elif cat == 2:
            return 0.37
        elif cat == 3:
            return 1.219
        else:
            return 0.504

    def predict_outcome(self, X):
        X = np.array(X)
        W = np.array(
            (
                self.est_age,
                self.est_his_cog,
                self.est_alc,
                self.__get_est_adm_cat(X[3]),
                self.est_urg,
                self.est_map,
                self.est_cort,
                self.est_resp,
                self.est_bun,
            )
        )
        Z = np.dot(W.T, X.T) + self.inter
        p = 1 / (1 + np.exp(-Z))
        return p
