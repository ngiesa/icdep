import pandas as pd
from modeling_layer.hypertuner import HyperTuner
from modeling_layer.metrics import get_metrics, get_metrics_df
from modeling_layer.models.multilayer_perceptron import MLP
from modeling_layer.train_functions import (
    load_train_test_dl,
    train_mlp_model,
    train_tree_model,
)
import traceback
from configs import no_feats, model_variants
from statistic_layer.feature_selector import FeatureSelector
from datetime import datetime
from preprocessing_layer.data_manager import DataManager
import gc
from sklearn.model_selection import StratifiedKFold


class ModelTrainer:
    def __init__(self) -> None:
        self.counter = 0
        self.num_cols = []
        self.model_name = ""
        self.miss_frac = 0
        self.num_thres = 0
        self.cat_thres = 0
        self.no_feats = no_feats
        self.model_variants = model_variants
        self.fs = FeatureSelector()
        self.dm = DataManager()
        self.result_experiments = pd.DataFrame()
        self.current_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    def merge_num_cat_features(self, df_num, df_cat):
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
        return df

    def __oversample(self, X_train, y_train):
        """
        perform random oversampling of training data if set true
                Parameters:
                        X_train (df): features per surgery, y_train (arr): GT class labels
                Returns:
                        X_train, y_train oversampled training data
        """
        X_train["c_target"] = y_train
        X_train = self.dm.perform_random_oversampling(
            df=X_train, incidence_rate=0.50)
        y_train = X_train["c_target"]
        X_train = X_train.drop(columns=["c_target"])
        return X_train, y_train

    def __execute_cv_training(self, df_num, df_cat, model_variant, l1_norm=False):
        """
        executes cross validation pipeline with preselected features and stores results
                Parameters:
                        df_num (df): dataframe holding numeric features,
                        df_cat (df): dataframe holding categoric features,
                        model_variant (dict): the model config as dict
                Returns:
                        None
        """

        num_features = list(df_num.drop(columns=self.no_feats).columns)
        cat_features = list(df_cat.drop(columns=self.no_feats).columns)

        ht = HyperTuner(current_date=self.current_date)

        for loss_type in model_variant["loss_type"]:

            if loss_type == "bce":
                continue  # TODO

            for drop_50 in model_variant["drop_50"]:
                if drop_50:
                    # did not yield any promising results
                    df_num = df_num.assign(
                        nan_count=df_num.isnull().sum(axis=1))
                    df_num = df_num.assign(
                        nan_perc=df_num.nan_count
                        / (len(df_num.columns) - len(self.no_feats))
                    )
                    df_num = df_num[df_num.nan_perc < 0.5].drop(
                        columns=["nan_perc", "nan_count"]
                    )
                # merge cat and num data
                df = self.merge_num_cat_features(df_num, df_cat)

                for oversampling in model_variant["oversample"]:
                    # getting X and Y
                    X = df.drop(columns=["c_target"])
                    y = df["c_target"]
                    y_pos = list(y).count(1)
                    y_neg = list(y).count(0)

                    incidence_rate = list(y).count(1) / len(list(y))

                    skf = StratifiedKFold(n_splits=3)
                    cv = 0
                    for train_index, test_index in skf.split(X, y):
                        cv = cv + 1
                        # create the corresponding fold sets
                        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
                        X_test, y_test = X.iloc[test_index], y.iloc[test_index]

                        # oversampling the training set only
                        if oversampling:
                            X_train, y_train = self.__oversample(
                                X_train, y_train)

                        # standarize the data
                        X_train, metr_dict = self.dm.standardize_data(
                            df=X_train, num_cols=self.num_cols
                        )
                        X_test, _ = self.dm.standardize_data(
                            df=X_test, metr_dict=metr_dict, num_cols=self.num_cols
                        )
                        X_train, X_test = X_train.fillna(0), X_test.fillna(0)

                        # assign data to instance
                        data = {
                            "X_train": X_train,
                            "X_test": X_test,
                            "y_train": y_train,
                            "y_test": y_test,
                        }

                        for ml_type in ["mlp", "tree"]:

                            is_mlp = True

                            if ml_type == "mlp":
                                # execute hyperparameter tuning with mlp
                                best_config = ht.execute_hyperparameter_optimization(
                                    ml_type=ml_type, loss_type=loss_type, data=data, l1_norm=l1_norm
                                )
                            elif (ml_type == "tree") and (loss_type == "bce"):
                                is_mlp = False
                                # execute hyperparameter tuning with tree and not for two losses
                                best_config = ht.execute_hyperparameter_optimization(
                                    ml_type=ml_type, data=data, loss_type=loss_type
                                )
                            else:
                                # if loss is focal and ml method tree, continue
                                continue

                            # retrain model with best configuration and store metrics on all folds
                            # training and application of model configuration
                            cv_validation = 0
                            for train_index, test_index in skf.split(X, y):
                                cv_validation = cv_validation + 1
                                # create the corresponding fold sets
                                X_train, y_train = X.iloc[train_index], y.iloc[train_index]
                                X_test, y_test = X.iloc[test_index], y.iloc[test_index]

                                # oversampling the training set only
                                if oversampling:
                                    X_train, y_train = self.__oversample(
                                        X_train, y_train
                                    )

                                # standarize the data
                                X_train, metr_dict = self.dm.standardize_data(
                                    df=X_train, num_cols=self.num_cols
                                )
                                X_test, _ = self.dm.standardize_data(
                                    df=X_test,
                                    metr_dict=metr_dict,
                                    num_cols=self.num_cols,
                                )
                                X_train, X_test = X_train.fillna(
                                    0), X_test.fillna(0)

                                # assign data to instance
                                data = {
                                    "X_train": X_train,
                                    "X_test": X_test,
                                    "y_train": y_train,
                                    "y_test": y_test,
                                }
                                pos_weight = list(y_train).count(0) / list(
                                    y_train
                                ).count(1)

                                if is_mlp:
                                    # retrain mlp with best hyperparameters from search
                                    train_df, test_df = load_train_test_dl(
                                        best_config["batch_size"],
                                        X_train,
                                        X_test,
                                        y_train,
                                        y_test,
                                    )
                                    # create model
                                    nodes = best_config["layer"] * \
                                        [best_config["node"]]
                                    model = MLP(
                                        n_inputs=X_train.shape[1], n_nodes=nodes, activation=best_config["activation"]
                                    )
                                    # retrain model and calculate metrics with selected hyperparameters
                                    gamma, lamda = 0, 0
                                    if loss_type == "focal":
                                        gamma = best_config["gamma"]
                                    if l1_norm:
                                        lamda = best_config["lamda"]
                                    (
                                        training_loss,
                                        validation_loss,
                                        model,
                                    ) = train_mlp_model(
                                        train_df=train_df,
                                        test_df=test_df,
                                        model=model,
                                        learning_rate=best_config["learning_rate"],
                                        pos_weight=pos_weight,
                                        loss_type=loss_type,
                                        gamma=gamma,
                                        l1_regularization=l1_norm,
                                        lamda=lamda
                                    )
                                    # apply for training data
                                    targets_train = train_df.dataset.tensors[1]
                                    inputs_train = train_df.dataset.tensors[0]
                                    pred_train = (
                                        model(inputs_train)
                                        .reshape(-1)
                                    )
                                    # apply for testing data
                                    targets_test = test_df.dataset.tensors[1]
                                    inputs_test = test_df.dataset.tensors[0]
                                    pred_test = (
                                        model(inputs_test)
                                        .reshape(-1)
                                    )
                                    metrics_dict_train = get_metrics(
                                        targets_train.detach().numpy(), pred_train.detach().numpy())
                                    metrics_dict_test = get_metrics(
                                        targets_test.detach().numpy(), pred_test.detach().numpy())

                                # train tree based methods as well
                                if not is_mlp:

                                    (training_loss, validation_loss, model,
                                        pred_test, pred_train
                                     ) = train_tree_model(
                                        config=best_config,
                                        data=data,
                                        pos_weight=pos_weight,
                                    )
                                    metrics_dict_train = get_metrics(
                                        data["y_train"], pred_train)
                                    metrics_dict_test = get_metrics(
                                        data["y_test"], pred_test)

                                self.result_experiments = (
                                    self.result_experiments.append(
                                        get_metrics_df(metrics_dict_train, metrics_dict_test, cv, cv_validation, incidence_rate, self.model_name, y_pos, y_neg,
                                                       loss_type, drop_50, oversampling, training_loss, validation_loss, self.miss_frac, self.num_thres, self.cat_thres,
                                                       best_config, num_features, cat_features, list(X.columns), len(X.columns), ml_type),
                                        ignore_index=True,
                                    )
                                )

                                self.result_experiments.to_csv(
                                    "./data/metrics/performance/training_procedure/results_{}_revised.csv".format(
                                        self.current_date
                                    )
                                )

                                gc.collect()

    def apply_experimental_methods(self, model_list):
        """
        executes experiments with cross validation technique performed on training set including
        1. feature selection with different thresholds, 2. ML model selection, 3. hyperparameter optimization with ray
        stores results in separate files and writes them on disc
                Parameters:
                        model_list (list): list with model names that need to be tested
                Returns:
                        None
        """
        for model in list(self.model_variants.keys()):
            if model not in model_list:
                continue
            # fraction of cases which have more than 50% missing features
            numeric_50miss_fractions = [0.4, 0.6, 0.8]
            # numeric AUC threshold
            numeric_auc_threshold = [0.1, 0.05]
            # minimum availability of patients having values in relation to all other
            availability_threshold = 0.1
            # categorical threshold for odds ratio
            categoric_or_threshold = [0.5, 1.5]
            # performing grid seach with these model variants
            for miss_frac in numeric_50miss_fractions:
                for cat_thres in categoric_or_threshold:
                    for num_thres in numeric_auc_threshold:
                        # perform evaluaiton with trees and mlps
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
                            print("model: " + model)
                            print("miss_frac: " + str(miss_frac))
                            print("cat_thres: " + str(cat_thres))
                            print("num_thres: " + str(num_thres))
                            self.model_name = model
                            self.miss_frac = miss_frac
                            self.num_thres = num_thres
                            self.cat_thres = cat_thres
                            self.num_cols = num_cols
                            self.__execute_cv_training(
                                df_num, df_cat, self.model_variants[model]
                            )
                            gc.collect()
                        except Exception as e:
                            print("Exception")
                            print(e)
                            print(traceback.format_exc())
                            continue
            # select all features and perform cross_validation with L1 norm for feature selection
            try:
                miss_frac, num_thres, cat_thres, availability_threshold = 1, 0.0, 0.0, 0.0
                (df_num, df_cat, num_cols, _,
                 ) = self.fs.apply_feature_selection_algorithm(
                    tl_list=self.model_variants[model]["tl"],
                    missing_fraction=miss_frac,
                    numeric_threshold=num_thres,
                    categoric_threshold=cat_thres,
                    availability_threshold=availability_threshold)
                self.model_name = model
                self.miss_frac = miss_frac
                self.num_thres = num_thres
                self.cat_thres = cat_thres
                self.num_cols = num_cols
                self.__execute_cv_training(
                    df_num, df_cat, self.model_variants[model], l1_norm=True)
            except Exception as e:
                print("Exception in L1 norm feature selection")
                print(e)
                print(traceback.format_exc())
                continue
