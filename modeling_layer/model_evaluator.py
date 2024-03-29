from xml.sax.handler import all_features
import pandas as pd
import pickle
import json
from modeling_layer.comparison_models import ComparisonModels
from modeling_layer.metrics import get_metrics
from modeling_layer.model_trainer import ModelTrainer
from plotting_layer.auc_plotter import AUCPlotter
from plotting_layer.metric_bars import MetricBar
from preprocessing_layer.data_manager import DataManager
from statistic_layer.feature_selector import FeatureSelector
from modeling_layer.train_functions import load_train_test_dl, train_mlp_model
from modeling_layer.models.multilayer_perceptron import MLP
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from plotting_layer.linesyles import linestyle_tuple
from configs import model_timelines, model_variants


class ModelEvaluator:
    def __init__(self) -> None:
        self.fs = FeatureSelector()
        self.mt = ModelTrainer()
        self.dm = DataManager()
        self.boot_strap_iterations = 1000

    def retreiving_train_and_test_sets_per_tl(self, tl: str = "", num_feats: list = [], cat_feats: list = []):
        '''
        loads train and test data from initial split, if no num_feats and cat_feats are specified, all data is loaded
        '''
        df_num_train, df_cat_train = self.fs.get_tl_datasets(tl, "train")
        df_num_test, df_cat_test = self.fs.get_tl_datasets(tl, "test")
        # adding missing indicator and locf within stay
        df_num_train = self.dm.locf_within_stay(df=df_num_train)
        df_num_train, col_miss = self.dm.add_missing_indicator(df_num_train)
        df_num_test = self.dm.locf_within_stay(df=df_num_test)
        df_num_test, col_miss = self.dm.add_missing_indicator(df_num_test)
        # merging to get df train and test sets
        df_train = df_num_train.merge(
            df_cat_train, on=self.mt.no_feats)
        df_test = df_num_test.merge(df_cat_test, on=self.mt.no_feats)
        # getting targets and feature space
        y_train = df_train.c_target
        y_test = df_test.c_target
        # missing indicator varibale
        if num_feats and cat_feats:
            X_train = df_train.drop(columns=self.mt.no_feats)[
                num_feats + cat_feats + col_miss]
            X_test = df_test.drop(columns=self.mt.no_feats)[
                num_feats + cat_feats + col_miss]
        else:
            X_train = df_train.drop(columns=self.mt.no_feats)
            X_test = df_test.drop(columns=self.mt.no_feats)
        # standarize the data
        X_train, metr_dict = self.dm.standardize_data(
            df=X_train, num_cols=num_feats
        )
        X_test, _ = self.dm.standardize_data(
            df=X_test, metr_dict=metr_dict, num_cols=num_feats,
        )
        # fill with 0 representing mean imputation
        X_train, X_test = X_train.fillna(0), X_test.fillna(0)
        return X_train, X_test, y_train, y_test

    def retrain_and_evaluate_models(self, evaluate_baselines=True):
        '''
        retrains best in 3-fold CV evaluated models, evaluates them on final test data and plots results with baselines
        '''

        for j in range(0, 3):

            # retrieving the best models with configurations
            df_model_conf = pd.read_csv(
                "./data/metrics/performance/training_procedure/best_models_revised_05_09_22.csv",
                index_col=0,
            ).sort_values(["model"])

            # define plot figure for all models per time line
            f_roc, ax_rocs = plt.subplots(
                3,
                2,
                figsize=(15, 20),
                constrained_layout=True,
                sharex=True,
                sharey=True,
            )
            f_pr, ax_prs = plt.subplots(
                3,
                2,
                figsize=(15, 20),
                constrained_layout=True,
                sharex=True,
                sharey=True,
            )

            # define plot for all models per timeline in AUC and PR
            f_auprc, ax_auprc = plt.subplots(
                2,
                2,
                figsize=(15, 12),
                constrained_layout=True,
                sharex=True,
                sharey=True,
            )

            # storing metric dicts
            metrics_dict_train_list = []
            metrics_dict_test_list = []

            df_val_res = pd.DataFrame()
            # init reference to comparison models
            cp = ComparisonModels()

            # boot strap iterations
            boot_strap_iterations = self.boot_strap_iterations

            # iterate through models and configuratiuons
            for i, model_name in enumerate(list(df_model_conf["model"])):

                title_dict = model_timelines[i]
                ax_title = "{} - {} ({})".format(
                    title_dict["from"],
                    title_dict["to"],
                    title_dict["m"].replace("M", "T"),
                )
                # get the axis where graphs are plotted on
                ax_pr = ax_prs[(i) % 3][int((i + 1) / (3.1))]
                ax_pr.set_title(ax_title)
                ax_roc = ax_rocs[(i) % 3][int((i + 1) / (3.1))]
                ax_roc.set_title(ax_title)
                print("Model variant: " + str(i))
                # get confoguration parameters from evaluation file
                current_eval = df_model_conf[df_model_conf.model == model_name]
                config_dict = json.loads(current_eval.config_dict.iloc[0])
                # read hyperparameter
                is_focal_loss = "gamma" in list(config_dict.keys())
                is_all_features = current_eval.miss_frac.iloc[0] == 1
                is_lamda = "lamda" in list(config_dict.keys())
                loss_type, lamda = "bce", 0
                if is_focal_loss:
                    loss_type = "focal"
                # get corresponding data
                tl = model_variants[model_name]["tl"]
                num_feats, cat_feats = current_eval.num_feats.iloc[0], current_eval.cat_feats.iloc[0]
                num_feats = [x.replace("'", "").strip()
                             for x in num_feats[1:-1].split(",")]
                cat_feats = [x.replace("'", "").strip()
                             for x in cat_feats[1:-1].split(",")]
                # load all available data per time phase for logistic regression
                X_train, X_test, y_train, y_test = self.retreiving_train_and_test_sets_per_tl(
                    tl=tl)
                # fitting LR classifier
                clf = LogisticRegression(random_state=0, class_weight="balanced", max_iter=2000).fit(
                    X_train, y_train
                )
                # make predictions with weighted logistic regression
                prob = clf.predict_proba(np.nan_to_num(X_train))
                prob = [x[1] for x in prob]
                metrics_dict_train = get_metrics(
                    targets=y_train, pred=prob,
                )
                metrics_dict_train["model"] = model_name + "_lr"
                metrics_dict_train["type"] = "train"
                metrics_dict_train["nr"] = str(i + 1)
                prob = clf.predict_proba(np.nan_to_num(X_test))
                prob = [x[1] for x in prob]
                metrics_dict_test = get_metrics(
                    targets=y_test, pred=prob,
                )
                metrics_dict_test["model"] = model_name + "_lr"
                metrics_dict_test["type"] = "test"
                metrics_dict_test["nr"] = str(i + 1)
                df_val_res = df_val_res.append(
                    pd.DataFrame(metrics_dict_train, index=[0]), ignore_index=True
                )
                df_val_res = df_val_res.append(
                    pd.DataFrame(metrics_dict_test, index=[0]), ignore_index=True
                )
                # apply bootstrapping on test data with logistic regression
                df_val_res = self.dm.perform_bootstrap_lr(
                    X_test,
                    y_test,
                    clf,
                    boot_strap_iterations,
                    df_val_res,
                    model_name,
                    i,
                    self.mt,
                )
                auc_plot = AUCPlotter()
                auc_plot.plot_roc_auc(
                    targets=y_test,
                    probabilitites=prob,
                    model_name="{}_lr_auroc:{}".format(
                        model_timelines[i]["m"],
                        metrics_dict_test["auc-roc"].round(3),
                    ),
                    ax=ax_roc,
                    draw_random=True,
                    draw_ticks=False,
                    linestyle="-.",
                )
                if i == 5:
                    auc_plot.plot_roc_auc(
                        targets=y_test,
                        probabilitites=prob,
                        model_name="{}_lr_auroc:{}".format(
                            model_timelines[i]["m"],
                            metrics_dict_test["auc-roc"].round(3),
                        ),
                        ax=ax_auprc[1][0],
                        draw_random=True,
                        draw_ticks=True,
                        linestyle="solid",
                        title="AUROC for time phase combination T123",
                    )
                auc_plot.plot_auc_pre_rec(
                    targets=y_test,
                    probabilitites=prob,
                    model_name="{}_lr_auprc:{}".format(
                        model_timelines[i]["m"],
                        metrics_dict_test["auc-pr"].round(3),
                    ),
                    ax=ax_pr,
                    draw_random=True,
                    draw_ticks=False,
                    linestyle="-.",
                )
                if i == 5:
                    auc_plot.plot_auc_pre_rec(
                        targets=y_test,
                        probabilitites=prob,
                        model_name="{}_lr_auprc:{}".format(
                            model_timelines[i]["m"],
                            metrics_dict_test["auc-pr"].round(3),
                        ),
                        ax=ax_auprc[1][1],
                        draw_random=True,
                        draw_ticks=True,
                        linestyle="solid",
                        title="AUPRC for time phase combination T123",
                    )

                # load all available data per time phase according to feature selection parameters
                X_train, X_test, y_train, y_test = self.retreiving_train_and_test_sets_per_tl(
                    tl=tl,
                    num_feats=num_feats,
                    cat_feats=cat_feats)
                # loading data in tensor format
                train_df, test_df = load_train_test_dl(
                    int(config_dict["batch_size"]
                        ), X_train, X_test, y_train, y_test
                )
                # get pos weight
                pos_weight = list(y_train).count(0) / list(y_train).count(1)
                # set gamma
                gamma = 0
                if is_focal_loss:
                    gamma = config_dict["gamma"]
                if is_lamda:
                    lamda = config_dict["lamda"]
                # create model
                nodes = config_dict["layer"] * [config_dict["node"]]
                model = MLP(
                    n_inputs=X_train.shape[1], n_nodes=nodes, activation=config_dict["activation"])

                # retrain model and calculate metrics with selected hyperparameters
                training_loss, validation_loss, model = train_mlp_model(
                    train_df=train_df,
                    test_df=test_df,
                    model=model,
                    learning_rate=config_dict["learning_rate"],
                    pos_weight=pos_weight,
                    loss_type=loss_type,
                    l1_regularization=is_all_features,
                    gamma=gamma,
                    lamda=lamda
                )
                # pickle model
                pickle.dump(
                    model, open(
                        "./modeling_layer/saved_models/sav_" + model_name, "wb")
                )
                # apply for training data
                targets = train_df.dataset.tensors[1]
                inputs = train_df.dataset.tensors[0]
                pred = model(inputs).reshape(-1)
                metrics_dict_train = get_metrics(
                    targets=targets.detach().numpy(),
                    pred=pred.detach().numpy()
                )
                # apply for testing data
                targets = test_df.dataset.tensors[1]
                inputs = test_df.dataset.tensors[0]
                pred = model(inputs).reshape(-1)
                metrics_dict_test = get_metrics(
                    targets=targets.detach().numpy(),
                    pred=pred.detach().numpy(),

                )
                f, ax = plt.subplots(figsize=(8, 5))
                ax.plot(training_loss)
                ax.plot(validation_loss)
                f.savefig(
                    "./plots/interpreted/best_models/eval_graphs/loss_{}.png".format(
                        model_name
                    )
                )
                aucp = AUCPlotter()
                aucp.plot_roc_auc(
                    targets.detach().numpy(),
                    pred.detach().numpy(),
                    "{}_mlp_auroc:{}".format(
                        model_timelines[i]["m"],
                        metrics_dict_test["auc-roc"].round(3),
                    ),
                    ax_roc,
                    False,
                    False,
                    "Time phase combination {}".format(
                        model_timelines[i]["m"].replace("M", "T")
                    ),
                )
                auc_plot.plot_roc_auc(
                    targets=targets.detach().numpy(),
                    probabilitites=pred.detach().numpy(),
                    model_name="{}_mlp_auroc:{}".format(
                        model_timelines[i]["m"],
                        metrics_dict_test["auc-roc"].round(3),
                    ),
                    ax=ax_auprc[0][0],
                    draw_random=(i == 0),
                    draw_ticks=True,
                    linestyle="solid",
                    title="AUROC per time phase combination T1-T123",
                )
                if i == 5:
                    auc_plot.plot_roc_auc(
                        targets=targets.detach().numpy(),
                        probabilitites=pred.detach().numpy(),
                        model_name="{}_mlp_auroc:{}".format(
                            model_timelines[i]["m"],
                            metrics_dict_test["auc-roc"].round(3),
                        ),
                        ax=ax_auprc[1][0],
                        draw_random=False,
                        draw_ticks=False,
                        linestyle="solid",
                    )
                aucp.plot_auc_pre_rec(
                    targets.detach().numpy(),
                    pred.detach().numpy(),
                    "{}_mlp_auprc:{}".format(
                        model_timelines[i]["m"],
                        metrics_dict_test["auc-pr"].round(3),
                    ),
                    ax_pr,
                    False,
                    False,
                    "Time phase combination {}".format(
                        model_timelines[i]["m"].replace("M", "T")
                    ),
                )
                auc_plot.plot_auc_pre_rec(
                    targets=targets.detach().numpy(),
                    probabilitites=pred.detach().numpy(),
                    model_name="{}_mlp_auprc:{}".format(
                        model_timelines[i]["m"],
                        metrics_dict_test["auc-pr"].round(3),
                    ),
                    ax=ax_auprc[0][1],
                    draw_random=(i == 0),
                    draw_ticks=True,
                    linestyle="solid",
                    title="AUPRC per time phase combination T1-T123",
                )
                if i == 5:
                    auc_plot.plot_auc_pre_rec(
                        targets=targets.detach().numpy(),
                        probabilitites=pred.detach().numpy(),
                        model_name="{}_mlp_auprc:{}".format(
                            model_timelines[i]["m"],
                            metrics_dict_test["auc-pr"].round(3),
                        ),
                        ax=ax_auprc[1][1],
                        draw_random=False,
                        draw_ticks=False,
                        linestyle="solid",
                    )

                metrics_dict_train_list.append(metrics_dict_train)
                metrics_dict_test_list.append(metrics_dict_test)

                metrics_dict_train["model"] = model_name + "_mlp"
                metrics_dict_train["type"] = "train"
                metrics_dict_train["nr"] = str(i + 1)
                metrics_dict_test["model"] = model_name + "_mlp"
                metrics_dict_test["type"] = "test"
                metrics_dict_test["nr"] = str(i + 1)

                df_val_res = df_val_res.append(
                    pd.DataFrame(metrics_dict_train, index=[0]), ignore_index=True
                )
                df_val_res = df_val_res.append(
                    pd.DataFrame(metrics_dict_test, index=[0]), ignore_index=True
                )

                # apply bootstrapping on test data
                df_val_res = self.dm.perform_bootstrap_lr(
                    inputs,
                    targets,
                    model,
                    boot_strap_iterations,
                    df_val_res,
                    model_name,
                    i,
                    self.mt,
                    True,
                )

                df_val_res.to_csv(
                    "./data/metrics/performance/final_test_set/eval_models.csv"
                )

                if evaluate_baselines:

                    print("Evaluating Boogard for i = " + str(i+1))

                    (
                        ax_roc,
                        ax_pr,
                        ax_auprc,
                        test_dict_list,
                        df_val_res,
                    ) = cp.evaluate_boogard_model(
                        ax_pr=ax_pr,
                        ax_roc=ax_roc,
                        ax_roc_pr=ax_auprc,
                        model_nr=(i + 1),
                        df_val_res=df_val_res,
                    )

                    print("Evaluating Wassenaar for i = " + str(i+1))

                    # evaluate the comparison model for the corresponding time perspective
                    (
                        ax_roc,
                        ax_pr,
                        ax_auprc,
                        test_dict_list,
                        df_val_res,
                    ) = cp.evaluate_wassenaar_model(
                        ax_pr=ax_pr,
                        ax_roc=ax_roc,
                        ax_roc_pr=ax_auprc,
                        model_nr=(i + 1),
                        df_val_res=df_val_res,
                    )
                    for test_dict in test_dict_list:
                        df_val_res = df_val_res.append(
                            pd.DataFrame(test_dict, index=[0]), ignore_index=True
                        )

                print("Finished model ", str(i + 1))

                # store the bootstrapped test results
            df_val_res.to_csv(
                "./data/metrics/performance/final_test_set/eval_models.csv"
            )

            ax_auprc[0][0].grid()
            ax_auprc[0][1].grid()
            f_auprc.savefig(
                "./plots/interpreted/best_models/eval_graphs/auroc_pr_summary_{}_revised.png".format(
                    str(j)
                )
            )

            f_roc.supxlabel("Sensitivity")
            f_roc.supylabel("1-Specificity")
            f_pr.supxlabel("Recall")
            f_pr.supylabel("Precision")
            f_roc.savefig(
                "./plots/interpreted/best_models/eval_graphs/auroc_summary_{}_revised.png".format(
                    str(j)
                )
            )
            f_pr.savefig(
                "./plots/interpreted/best_models/eval_graphs/auprc_summary_{}_revised.png".format(
                    str(j)
                )
            )
