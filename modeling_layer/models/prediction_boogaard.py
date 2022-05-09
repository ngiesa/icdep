# Prediction model
# van den Boogaard, M. H. W. A., Schoonhoven, L., Maseda, E., Plowright, C., Jones, C., Luetz, A., ... & Pickkers, P. (2014).
# Recalibration of the delirium prediction model for ICU patients (PRE-DELIRIC): a multinational observational study. Intensive care medicine, 40(3), 361-369.
# using the new values of linear predictors .. maybe splitting in recal and normal models
# urea must be highest value in mm/L

import numpy as np


class BoogaardPredictor:

    def __init__(self):
        self.inter = (-6.3131)
        self.est_age = 0.0387
        self.est_apache = 0.0575
        self.est_infect = 1.0509
        self.est_meta_acid = 0.2918
        self.est_seda = 1.3932
        self.est_urea = 0.0298
        self.est_urg = 0.4004

    # coma 0 -> no, 1 -> drug, 2 -> other, 3 -> combination
    def __get_est_coma(self, coma):
        if coma == 1:
            return 0.5458
        elif coma == 2:
            return 2.2695
        elif coma == 3:
            return 2.8283
        else:
            return 0

    # adm cat 1 -> surgery, 2 -> medical, 3 -> trauma, 4 -> neuro/surgery
    def __get_est_adm_cat(self, cat):
        if cat == 2:
            return 0.3061
        elif cat == 3:
            return 1.1253
        elif cat == 4:
            return 1.3793
        else:
            return 0

    # enter the commulated morphine dosis in mg
    def __get_est_morph(self, morph):
        if (morph >= 0.01) and (morph <= 7.1):
            return 0.4078
        elif (morph >= 7.2) and (morph <= 18.6):
            return 0.1323
        elif (morph > 18.6):
            return 0.5110
        else:
            return 0

    def predict_outcome(self, X):
        X = np.array(X)
        W = np.array((self.est_age,
                      self.est_apache,
                      self.__get_est_coma(X[2]),
                      self.__get_est_adm_cat(X[3]),
                      self.est_infect,
                      self.est_meta_acid,
                      self.__get_est_morph(X[6]),
                      self.est_seda,
                      self.est_urea,
                      self.est_urg
                      ))
        Z = np.dot(W.T, X.T)+self.inter
        p = 1/(1+np.exp(-Z))
        return p
