from os import pread
import pandas as pd
from pandas.core.frame import DataFrame
from pandas.io.pytables import incompatibility_doc
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from modeling_layer.models.multilayer_perceptron import MLP
from modeling_layer.train_functions import (
    hyperparameter_optimization,
    load_train_test_dl,
    train_model,
)
from statistic_layer.feature_selector import FeatureSelector
from statistic_layer.descriptive_statistic import DescriptiveStatistic
from datetime import datetime
from extraction_layer.support_classes.js_converter import JSConverter
from preprocessing_layer.data_manager import DataManager
import numpy as np
from ray import tune
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    roc_curve,
    roc_auc_score,
    f1_score,
)
import gc


class ModelTrainer:
    def __init__(self) -> None:
        self.counter = 0
        self.num_cols = []
        self.model_name = ""
        self.miss_frac = 0
        self.num_thres = 0
        self.cat_thres = 0
        self.fs = FeatureSelector()
        self.dm = DataManager()
        self.result_experiments = pd.DataFrame()
        self.df_hyperparam_res = pd.DataFrame()
        self.current_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.model_variants = {
            "model_1": {
                "tl": [1],
                "drop_50": [False, True],
                "oversample": [False],  # [True]
            },
            "model_2": {
                "tl": [2],
                "drop_50": [False, True],
                "oversample": [False],  # [True]
            },
            "model_3": {
                "tl": [3],
                "drop_50": [False, True],
                "oversample": [False],  # [True]
            },
            "model_4": {
                "tl": [1, 2],
                "drop_50": [False, True],
                "oversample": [False],  # [True]
            },
            "model_5": {
                "tl": [2, 3],
                "drop_50": [False, True],
                "oversample": [False],  # [True]
            },
            "model_6": {
                "tl": [1, 2, 3],
                "drop_50": [False, True],
                "oversample": [False],  # [True]
            },
        }
        self.no_feats = [
            "c_pat_id",
            "c_case_id",
            "c_an_start_ts",
            "c_op_id",
            "c_target",
            "c_time_consistent",
        ]

    def get_metrics(self, targets, pred, thres):  # TODO thresholds
        df_target = pd.DataFrame({"targets": targets, "pred": pred}).dropna()
        targets = df_target["targets"]
        pred = df_target["pred"]
        fpr, tpr, thresholds = roc_curve(targets, pred)
        df_eval = pd.DataFrame(
            {
                "fpr": fpr,
                "tpr": tpr,
                "tnr": [1 - x for x in fpr],
                "tnr_tpr": tpr + [1 - x for x in fpr],
                "tres": thresholds,
            }
        )
        # threshold where sum tpr tnr is max
        df_max = df_eval[df_eval.tnr_tpr == max(df_eval.tnr_tpr)]
        max_thres = df_max.tres.iloc[0]
        # calculations for precision / recall
        yhat = [0 if x < max_thres else 1 for x in pred]
        # evaluate on precision, recall, specificity
        precision, recall, thresholds = precision_recall_curve(targets, pred)
        # get results where sum maximizes
        sens_spec = pd.DataFrame(
            {"spec": [1 - x for x in list(fpr)], "sens": list(tpr)}
        )
        sens_spec = sens_spec.assign(sum_spec_sens=sens_spec.spec + sens_spec.sens)
        sens_spec = sens_spec[sens_spec.sum_spec_sens == max(sens_spec.sum_spec_sens)]
        prec_rec = pd.DataFrame({"prec": list(precision), "rec": list(recall)})
        # remove edge cases
        prec_rec_05 = prec_rec[(prec_rec.rec >= 0.5)]
        prec_rec_05 = prec_rec_05[prec_rec_05.prec == max(prec_rec_05.prec)]
        prec_rec_07 = prec_rec[(prec_rec.rec >= 0.7)]
        prec_rec_07 = prec_rec_07[prec_rec_07.prec == max(prec_rec_07.prec)]
        prec_rec_08 = prec_rec[(prec_rec.rec >= 0.8)]
        prec_rec_08 = prec_rec_08[prec_rec_08.prec == max(prec_rec_08.prec)]
        prec_rec_09 = prec_rec[(prec_rec.rec >= 0.9)]
        prec_rec_09 = prec_rec_09[prec_rec_09.prec == max(prec_rec_09.prec)]
        f_score = f1_score(targets, yhat)
        print("AUC-PR: %0.3f" % auc(recall, precision))
        print("AUC-ROC:  %0.3f" % roc_auc_score(targets, pred))
        print("Sensitivity: %0.3f" % sens_spec.iloc[0]["sens"])
        print("Specificity: %0.3f" % sens_spec.iloc[0]["spec"])
        print("Precision: %0.3f" % prec_rec_07.iloc[0]["prec"])
        print("F1-Score: %0.3f" % f_score)
        return {
            "auc-pr": auc(recall, precision),
            "auc-roc": roc_auc_score(targets, pred),
            "sens": sens_spec.iloc[0]["sens"],
            "spec": sens_spec.iloc[0]["spec"],
            "prec_07": prec_rec_07.iloc[0]["prec"],
            "prec_08": prec_rec_08.iloc[0]["prec"],
            "prec_09": prec_rec_09.iloc[0]["prec"],
            "prec_05": prec_rec_05.iloc[0]["prec"],
            # "spec_07": sens_spec[sens_spec.sens >= 0.7].iloc[0]["spec"],
            # "spec_08": sens_spec[sens_spec.sens >= 0.8].iloc[0]["spec"],
            # "spec_09": sens_spec[sens_spec.sens >= 0.9].iloc[0]["spec"],
            # "spec_05": sens_spec[sens_spec.sens >= 0.5].iloc[0]["spec"],
            "f1": f_score,
        }

    def __oversample(self, X_train, y_train):
        X_train["c_target"] = y_train
        X_train = self.dm.perform_random_oversampling(df=X_train, incidence_rate=0.50)
        y_train = X_train["c_target"]
        X_train = X_train.drop(columns=["c_target"])
        return X_train, y_train

    def __prepare_datasets(self, df_num, df_cat, model):

        num_features = list(df_num.drop(columns=self.no_feats).columns)
        cat_features = list(df_cat.drop(columns=self.no_feats).columns)

        for drop_50 in model["drop_50"]:
            if drop_50:
                df_num = df_num.assign(nan_count=df_num.isnull().sum(axis=1))
                df_num = df_num.assign(
                    nan_perc=df_num.nan_count
                    / (len(df_num.columns) - len(self.no_feats))
                )
                df_num = df_num[df_num.nan_perc < 0.5].drop(
                    columns=["nan_perc", "nan_count"]
                )

            df_num = self.dm.locf_within_stay(df=df_num)
            df_num, col_miss = self.dm.add_missing_indicator(df_num)
            # merge cat and num data
            df = df_num.merge(df_cat, on=self.no_feats)
            # drop any occuring duplicates
            df = df.drop(
                columns=[
                    "c_pat_id",
                    "c_case_id",
                    "c_an_start_ts",
                    "c_op_id",
                    "c_time_consistent",
                ]
            ).drop_duplicates()

            for oversampling in model["oversample"]:
                # getting X and Y
                X = df.drop(columns=["c_target"])
                y = df["c_target"]
                indcidence_rate = list(y).count(1) / len(list(y))
                y_pos = list(y).count(1)
                y_neg = list(y).count(0)

                # apply 3 fold cross validation and 3 times different best found hyperparameter sets
                from sklearn.model_selection import StratifiedKFold, KFold

                skf = StratifiedKFold(n_splits=3)
                cv = 0
                for train_index, test_index in skf.split(X, y):
                    cv = cv + 1
                    # create the corresponding fold sets
                    X_train = X.iloc[train_index]
                    y_train = y.iloc[train_index]
                    X_test = X.iloc[test_index]
                    y_test = y.iloc[test_index]

                    # oversampling the training set only
                    if oversampling:
                        X_train, y_train = self.__oversample(X_train, y_train)

                    # standarize the data
                    X_train, metr_dict = self.dm.standardize_data(
                        df=X_train, num_cols=self.num_cols
                    )
                    X_test, _ = self.dm.standardize_data(
                        df=X_test, metr_dict=metr_dict, num_cols=self.num_cols
                    )
                    X_train = X_train.fillna(0)
                    X_test = X_test.fillna(0)

                    # assign data to instance
                    data = {
                        "X_train": X_train,
                        "X_test": X_test,
                        "y_train": y_train,
                        "y_test": y_test,
                    }

                    # execute the hyperparameter search
                    analysis = tune.run(
                        tune.with_parameters(hyperparameter_optimization, data=data),
                        config={
                            "batch_size": tune.choice([64, 128, 256]),
                            "learning_rate": tune.choice([1e-3, 1e-4, 1e-5]),
                            "loss_type": tune.choice(["bce"]),
                            "node": tune.grid_search([8, 16, 32, 64, 128]),
                            "layer": tune.grid_search([4, 8, 12]),
                        },
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
                        "./data/metrics/performance/training_procedure/ray_tuner_{}.csv".format(
                            self.current_date
                        )
                    )

                    # retrain model with best configuration and store metrics on all folds
                    # training and application of model configuration
                    cv_validation = 0
                    for train_index, test_index in skf.split(X, y):
                        cv_validation = cv_validation + 1
                        # create the corresponding fold sets
                        X_train = X.iloc[train_index]
                        y_train = y.iloc[train_index]
                        X_test = X.iloc[test_index]
                        y_test = y.iloc[test_index]

                        # oversampling the training set only
                        if oversampling:
                            X_train, y_train = self.__oversample(X_train, y_train)

                        # standarize the data
                        X_train, metr_dict = self.dm.standardize_data(
                            df=X_train, num_cols=self.num_cols
                        )
                        X_test, _ = self.dm.standardize_data(
                            df=X_test, metr_dict=metr_dict, num_cols=self.num_cols
                        )
                        X_train = X_train.fillna(0)
                        X_test = X_test.fillna(0)

                        # assign data to instance
                        data = {
                            "X_train": X_train,
                            "X_test": X_test,
                            "y_train": y_train,
                            "y_test": y_test,
                        }
                        pos_weight = list(y_train).count(0) / list(y_train).count(1)
                        train_df, test_df = load_train_test_dl(
                            best_config["batch_size"], X_train, X_test, y_train, y_test
                        )
                        # create model
                        nodes = best_config["layer"] * [best_config["node"]]
                        model = MLP(n_inputs=X_train.shape[1], n_nodes=nodes)
                        # retrain model and calculate metrics with selected hyperparameters
                        training_loss, validation_loss, model = train_model(
                            train_df,
                            test_df,
                            model,
                            best_config["learning_rate"],
                            pos_weight,
                            best_config["loss_type"],
                        )

                        # apply for training data
                        targets = train_df.dataset.tensors[1]
                        inputs = train_df.dataset.tensors[0]
                        pred = model(inputs).reshape(-1)
                        metrics_dict_train = self.get_metrics(
                            targets=targets.detach().numpy(),
                            pred=pred.detach().numpy(),
                            thres=0.5,
                        )
                        # apply for testing data
                        targets = test_df.dataset.tensors[1]
                        inputs = test_df.dataset.tensors[0]
                        pred = model(inputs).reshape(-1)
                        metrics_dict_test = self.get_metrics(
                            targets=targets.detach().numpy(),
                            pred=pred.detach().numpy(),
                            thres=0.5,
                        )
                        self.result_experiments = self.result_experiments.append(
                            pd.DataFrame(
                                {
                                    "cv_hyperparam": [cv],
                                    "cv_validation": [cv_validation],
                                    "model": [self.model_name],
                                    "incidence-rate": [indcidence_rate],
                                    "y_pos": [y_pos],
                                    "y_neg": [y_neg],
                                    "drop_50": [drop_50],
                                    "oversample": [oversampling],
                                    "loss-train": [np.mean(training_loss)],
                                    "loss-val": [np.mean(validation_loss)],
                                    "auc-roc-train": [metrics_dict_train["auc-roc"]],
                                    "auc-roc-val": [metrics_dict_test["auc-roc"]],
                                    "auc-pr-train": [metrics_dict_train["auc-pr"]],
                                    "auc-pr-val": [metrics_dict_test["auc-pr"]],
                                    "sens-train": [metrics_dict_train["sens"]],
                                    "sens-val": [metrics_dict_test["sens"]],
                                    "spec-train": [metrics_dict_train["spec"]],
                                    "spec-val": [metrics_dict_test["spec"]],
                                    "prec-train": [metrics_dict_train["prec"]],
                                    "prec-val": [metrics_dict_test["prec"]],
                                    "miss_frac": [self.miss_frac],
                                    "num_thres": [self.num_thres],
                                    "cat_thres": [self.cat_thres],
                                    "batch_size": [best_config["batch_size"]],
                                    "learning_rate": [best_config["learning_rate"]],
                                    "node": [best_config["node"]],
                                    "layer": [best_config["layer"]],
                                    "num_feats": [num_features],
                                    "cat_feats": [cat_features],
                                }
                            ),
                            ignore_index=True,
                        )

                        self.result_experiments.to_csv(
                            "./data/metrics/performance/training_procedure/results_{}.csv".format(
                                self.current_date
                            )
                        )

    def apply_experimental_methods(self, model_list):

        for model in list(self.model_variants.keys()):
            if model not in model_list:
                continue
            # fraction of cases which have more than 50% missing features
            numeric_50miss_fractions = [0.2, 0.4, 0.6]  # TODO
            # numeric AUC threshold
            numeric_auc_threshold = [0.1, 0.05]
            # minimum availability of patients having values in relation to all other
            availability_threshold = 0.05
            # categorical threshold for odds ratio
            categoric_or_threshold = [1.5, 0.5]
            for miss_frac in numeric_50miss_fractions:
                for cat_thres in categoric_or_threshold:
                    for num_thres in numeric_auc_threshold:
                        try:
                            (
                                df_num,
                                df_cat,
                                num_cols,
                                _,
                            ) = self.fs.apply_feature_selection_algorithm(
                                tl_list=self.model_variants[model]["tl"],
                                missing_fraction=miss_frac,
                                numeric_threshold=num_thres,
                                categoric_threshold=cat_thres,
                                availability_threshold=availability_threshold,
                            )
                            self.model_name = model
                            self.miss_frac = miss_frac
                            self.num_thres = num_thres
                            self.cat_thres = cat_thres
                            self.num_cols = num_cols
                            self.__prepare_datasets(
                                df_num, df_cat, self.model_variants[model]
                            )
                            gc.collect()
                        except Exception as e:
                            print("Exception")
                            print(e)
                            continue
