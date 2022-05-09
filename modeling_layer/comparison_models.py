from numpy import True_
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gc
from extraction_layer.support_classes.js_converter import JSConverter
from preprocessing_layer.data_manager import DataManager
from preprocessing_layer.data_manager import DataManager
from preprocessing_layer.time_manager import TimeManager
from preprocessing_layer.preprocessor import Preprocessor
from pandas.core.frame import DataFrame
from plotting_layer.auc_plotter import AUCPlotter
from sklearn.linear_model import LogisticRegression
from modeling_layer.train_functions import load_train_test_dl, train_model
from modeling_layer.models.multilayer_perceptron import MLP
from modeling_layer.model_trainer import ModelTrainer
from modeling_layer.models.prediction_boogaard import BoogaardPredictor
from modeling_layer.models.prediction_boogaard_rec import BoogaardPredictorRecalibrated
from modeling_layer.models.prediction_wassenaar import WassenaarPredictor


class ComparisonModels:
    def __init__(self) -> None:
        self.dm = DataManager()
        self.mt = ModelTrainer()
        self.ax_roc = []
        self.f_roc = []
        self.bootstrap_rounds = 100

    def retrain_lr_and_mlp(
        self,
        X_train,
        X_test,
        df_merge_test,
        df_merge_train,
        ax_roc,
        ax_pr,
        ax_roc_pr,
        num_cols,
        author,
        model_nr=1,
        linestyle="-",
        df_val_res={},
    ):
        metrics_dict_all = []
        auc_plot = AUCPlotter()
        X_train, metr_dict = self.dm.standardize_data(df=X_train, num_cols=num_cols)
        X_test, _ = self.dm.standardize_data(
            df=X_test, num_cols=num_cols, metr_dict=metr_dict
        )
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
        y_train = list(df_merge_train.c_target)
        y_test = list(df_merge_test.c_target)
        # try out simple logistic regression first
        clf = LogisticRegression(random_state=0, class_weight="balanced").fit(
            X_train, y_train
        )
        # apply on train data
        prob = clf.predict_proba(np.nan_to_num(X_train))
        prob = [x[1] for x in prob]
        metrics_dict_train = self.mt.get_metrics(targets=y_train, pred=prob, thres=0.5,)
        metrics_dict_train["model"] = author + "_lr"
        metrics_dict_train["type"] = "train"
        metrics_dict_train["nr"] = model_nr
        metrics_dict_all.append(metrics_dict_train)
        # apply on test data
        prob = clf.predict_proba(np.nan_to_num(X_test))
        prob = [x[1] for x in prob]
        metrics_dict_test = self.mt.get_metrics(targets=y_test, pred=prob, thres=0.5,)
        metrics_dict_test["model"] = author + "_lr"
        metrics_dict_test["type"] = "test"
        metrics_dict_test["nr"] = model_nr
        metrics_dict_all.append(metrics_dict_test)
        df_val_res = self.dm.perform_bootstrap_lr(
            X_test,
            y_test,
            clf,
            self.bootstrap_rounds,
            df_val_res,
            author,
            model_nr,
            self.mt,
            False,
        )
        auc_plot.plot_roc_auc(
            targets=y_test,
            probabilitites=prob,
            model_name="{}_lr_auroc:{}".format(
                author, metrics_dict_test["auc-roc"].round(3)
            ),
            ax=ax_roc,
            draw_random=False,
            draw_ticks=False,
            linestyle=linestyle,
        )
        auc_plot.plot_auc_pre_rec(
            targets=y_test,
            probabilitites=prob,
            model_name="{}_lr_auprc:{}".format(
                author, metrics_dict_test["auc-pr"].round(3)
            ),
            ax=ax_pr,
            draw_random=False,
            draw_ticks=False,
            linestyle=linestyle,
        )
        if model_nr == 6:
            auc_plot.plot_roc_auc(
                targets=y_test,
                probabilitites=prob,
                model_name="{}_lr_auroc:{}".format(
                    author, metrics_dict_test["auc-roc"].round(3)
                ),
                ax=ax_roc_pr[1][0],
                draw_random=False,
                draw_ticks=False,
                linestyle=linestyle,
            )
            auc_plot.plot_auc_pre_rec(
                targets=y_test,
                probabilitites=prob,
                model_name="{}_lr_auprc:{}".format(
                    author, metrics_dict_test["auc-pr"].round(3)
                ),
                ax=ax_roc_pr[1][1],
                draw_random=False,
                draw_ticks=False,
                linestyle=linestyle,
            )
        # apply MLP
        train_df, test_df = load_train_test_dl(
            int(128), X_train, X_test, y_train, y_test
        )
        pos_weight = list(y_train).count(0) / list(y_train).count(1)
        nodes = 4 * [128]
        model = MLP(n_inputs=X_train.shape[1], n_nodes=nodes)
        # retrain model and calculate metrics with selected hyperparameters
        gc.collect()
        training_loss, validation_loss, model = train_model(
            train_df, test_df, model, 1e-3, pos_weight, "bce", 5, 50
        )
        targets = train_df.dataset.tensors[1]
        inputs = train_df.dataset.tensors[0]
        pred = model(inputs).reshape(-1)
        df_target = pd.DataFrame(
            {"targets": targets.detach().numpy(), "pred": pred.detach().numpy()}
        ).dropna()
        targets = df_target["targets"]
        pred = df_target["pred"]
        metrics_dict_train = self.mt.get_metrics(targets=targets, pred=pred, thres=0.5,)
        model_name = author + "_mlp"
        metrics_dict_train["model"] = model_name
        metrics_dict_train["type"] = "train"
        metrics_dict_train["nr"] = model_nr
        metrics_dict_all.append(metrics_dict_train)
        # apply for testing data
        targets = test_df.dataset.tensors[1]
        inputs = test_df.dataset.tensors[0]
        if len(inputs) != len(targets):
            raise ValueError("inputs and tagets are not equal")
        df_val_res = self.dm.perform_bootstrap_lr(
            inputs,
            targets,
            model,
            self.bootstrap_rounds,
            df_val_res,
            author,
            model_nr,
            self.mt,
            True,
        )
        pred = model(inputs).reshape(-1)
        df_target = pd.DataFrame(
            {"targets": targets.detach().numpy(), "pred": pred.detach().numpy()}
        ).dropna()
        targets = df_target["targets"]
        pred = df_target["pred"]
        metrics_dict_test = self.mt.get_metrics(targets=targets, pred=pred, thres=0.5)
        metrics_dict_test["model"] = model_name
        metrics_dict_test["type"] = "test"
        metrics_dict_test["nr"] = model_nr
        metrics_dict_all.append(metrics_dict_test)
        for d in metrics_dict_all:
            df_val_res = df_val_res.append(
                pd.DataFrame(d, index=[0]), ignore_index=True
            )

        df_val_res.to_csv("./data/metrics/performance/final_test_set/eval_models.csv")

        auc_plot.plot_roc_auc(
            targets=targets,
            probabilitites=pred,
            model_name="{}_mlp_auroc:{}".format(
                author, metrics_dict_test["auc-roc"].round(3)
            ),
            ax=ax_roc,
            draw_random=False,
            draw_ticks=False,
            linestyle=linestyle,
        )
        auc_plot.plot_auc_pre_rec(
            targets=targets,
            probabilitites=pred,
            model_name="{}_mlp_auprc:{}".format(
                author, metrics_dict_test["auc-pr"].round(3)
            ),
            ax=ax_pr,
            draw_random=False,
            draw_ticks=False,
            linestyle=linestyle,
        )
        if model_nr == 6:
            auc_plot.plot_roc_auc(
                targets=targets,
                probabilitites=pred,
                model_name="{}_mlp_auroc:{}".format(
                    author, metrics_dict_test["auc-roc"].round(3)
                ),
                ax=ax_roc_pr[1][0],
                draw_random=False,
                draw_ticks=False,
                linestyle=linestyle,
            )
            auc_plot.plot_auc_pre_rec(
                targets=targets,
                probabilitites=pred,
                model_name="{}_mlp_auprc:{}".format(
                    author, metrics_dict_test["auc-pr"].round(3)
                ),
                ax=ax_roc_pr[1][1],
                draw_random=False,
                draw_ticks=False,
                linestyle=linestyle,
            )

        return ax_roc, ax_pr, ax_roc_pr, df_val_res

    def evaluate_boogard_model(self, ax_roc, ax_pr, ax_roc_pr, model_nr, df_val_res):
        # apply for training data
        # first evaluate the boogard models
        features_boogaard = [
            "age",
            "amount_morphine",
            "emergency_admission",
            "apache_ii",
            "rass",
            "drug_class_antibiotics",
            "drug_class_sedative",
            "infection_diagnosis",
            "urea_nitrogen_in_blood",
            "ph_in_blood",
            "bicarbonat_in_blood",
        ]
        data_holder = self.dm.load_features(features_boogaard, model_nr)
        df = data_holder["apache_ii"]
        # merge time master table with value df
        df = self.dm.get_first_value(df)
        # reassigning apache score to data list
        data_holder["apache_ii"] = df
        # labeling coma category with rass scoring
        df = data_holder["rass"]
        # sample data in 8h blocks according to authors description
        df = self.dm.sample_mean_per_time_unit(df=df, time_unit="8H")
        # create variable c_value where 1 stands for 8h below -4 and 0 for above
        df = df.assign(c_value=[1 if x <= -4 else 0 for x in list(df["c_value"])])
        # assigning rass
        data_holder["rass_8h"] = self.dm.get_first_value(
            df[["c_case_id", "c_start_ts", "c_value"]]
        )
        # label rass positives with use of medication sedatives
        df = data_holder["drug_class_sedative"].merge(
            data_holder["rass_8h"],
            on=["c_case_id"],
            how="inner",
            suffixes=["", "_rass"],
        )
        # define conditions on which coma is defined by authors
        df = df[
            (df.c_start_ts_rass > df.c_start_ts)
            & (df.c_start_ts_rass < df.c_end_ts)
            & (df.c_value_rass == 1)
        ]
        # select parts to append to
        df = df[["c_value", "c_case_id", "c_op_id"]]
        # reassigning values to rass for coma definition
        df_rass = data_holder["rass_8h"]
        df_rass = df_rass.assign(
            c_value=[2 if x == 1 else 0 for x in list(data_holder["rass_8h"].c_value)]
        )
        df_rass = df_rass[["c_value", "c_case_id", "c_op_id"]]
        # appending and selecting the maximum category
        df = df_rass.append(df).drop_duplicates()
        df = df.groupby(["c_op_id", "c_case_id"])["c_value"].max().reset_index()
        # reassigning data
        data_holder["coma"] = df
        # combining features to define infection like defined by the authors with antibiotics intake
        df = (
            self.dm.get_first_value(data_holder["drug_class_antibiotics"])
            .merge(
                self.dm.get_first_value(data_holder["infection_diagnosis"]),
                on=["c_case_id", "c_op_id"],
                how="outer",
                suffixes=["_ant", "_dia"],
            )
            .fillna(0)
        )
        # calc sum of bowth binary variables
        df = df.assign(c_value_sum=df.c_value_ant + df.c_value_dia)
        # create variable with 1, 0 in case at least one is positive
        df = df.assign(c_value=[1 if x > 0 else 0 for x in list(df["c_value_sum"])])
        # assigning to infection in data holder
        data_holder["infection"] = df[
            ["c_case_id", "c_op_id", "c_value"]
        ].drop_duplicates()
        # sample ph values as well so that they can be compared
        df = data_holder["ph_in_blood"]
        # sample data in 8h blocks according to authors description
        df = self.dm.sample_mean_per_time_unit(df=df, time_unit="8H")
        # create variable c_value with 1 for <7.35 and 0 for a level above
        df = df.assign(c_value=[1 if x < 7.35 else 0 for x in list(df["c_value"])])
        # reassigning to data holder
        data_holder["ph_in_blood_8h"] = df
        # sample bicarbonate values as well so that they can be compared
        df = data_holder["bicarbonat_in_blood"]
        # filter all data which was measured in mmol/L
        df = df[df.c_unit.str.lower() == "mmol/l"]
        # sample data in 8h blocks according to authors description
        df = self.dm.sample_mean_per_time_unit(df=df, time_unit="8H")
        # create variable c_value with 1 for <24 mmol/l and 0 for a level above
        df = df.assign(c_value=[1 if x < 24 else 0 for x in list(df["c_value"])])
        # reassigning to data holder
        data_holder["bicarbonat_in_blood_8h"] = df
        # combining and annotating data ph and bicarbonate
        df = pd.merge(
            data_holder["bicarbonat_in_blood_8h"],
            data_holder["ph_in_blood_8h"],
            on=["c_case_id", "c_op_id", "c_start_ts"],
            suffixes=["_bic", "_ph"],
        )
        df = df.assign(c_value_sum=df.c_value_bic + df.c_value_ph)
        df = df.assign(c_value=[1 if x == 2 else 0 for x in list(df.c_value_sum)])
        # reassigning data
        data_holder["metabolic_acidosis"] = (
            df[["c_case_id", "c_op_id", "c_value"]].drop_duplicates().fillna(0)
        )
        # getting the accumulated morphine
        df = data_holder["amount_morphine"]
        # aggregate c_value per case and operation
        df = self.dm.get_cumulated_value(df)
        # reassign df
        data_holder["amount_morphine"] = df
        # highest value of urea in blood before pod
        df = data_holder["urea_nitrogen_in_blood"]
        # converting units from mg/dl in mmol/L
        df = df.assign(c_value=[x / 18 for x in list(df.c_value)])
        # get highest value in df
        df = self.dm.get_highest_value(df)
        # storing it in data holder
        data_holder["urea_nitrogen_in_blood"] = df
        # merge dfs either inner or outer depending on required on missing
        df_merge = (
            DataManager()
            .master_df[["c_op_id", "c_case_id", "c_pat_id", "c_target"]]
            .merge(
                data_holder["age"][["c_case_id", "c_value"]],
                on="c_case_id",
                how="left",
                suffixes=["", "_age"],
            )
            .merge(
                data_holder["apache_ii"][["c_op_id", "c_value"]],
                on="c_op_id",
                how="left",
                suffixes=["", "_apache"],
            )
            .merge(
                data_holder["coma"][["c_op_id", "c_value"]],
                on="c_op_id",
                how="left",
                suffixes=["", "_coma"],
            )
            .assign(c_value_adm=2)
            .merge(
                data_holder["infection"][["c_op_id", "c_value"]],
                on="c_op_id",
                how="left",
                suffixes=["", "_inf"],
            )
            .merge(
                data_holder["metabolic_acidosis"][["c_op_id", "c_value"]],
                on="c_op_id",
                how="left",
                suffixes=["", "_acid"],
            )
            .merge(
                data_holder["amount_morphine"][["c_op_id", "c_value"]],
                on="c_op_id",
                how="left",
                suffixes=["", "_morph"],
            )
            .merge(
                data_holder["drug_class_sedative"][["c_case_id", "c_value"]],
                on="c_case_id",
                how="left",
                suffixes=["", "_sed"],
            )
            .merge(
                data_holder["emergency_admission"][["c_case_id", "c_value"]],
                on="c_case_id",
                how="left",
                suffixes=["", "_emg"],
            )
            .merge(
                data_holder["urea_nitrogen_in_blood"][["c_op_id", "c_value"]],
                on="c_op_id",
                how="left",
                suffixes=["", "_urea"],
            )
            .drop_duplicates()
        )
        # apply median imputation technique
        for col in list(
            df_merge.drop(
                columns=["c_op_id", "c_case_id", "c_pat_id", "c_target"]
            ).columns
        ):
            df_merge[col] = df_merge[col].fillna((df_merge[col].median()))
        # opening test and train sets
        jsc = JSConverter()
        js = jsc.read_js_file("./data/meta/cohort/train_test_split")
        df_merge_test = df_merge[
            (df_merge.c_pat_id.isin(js["test_pos"]))
            | (df_merge.c_pat_id.isin(js["test_neg"]))
        ]
        df_merge_train = df_merge[
            (df_merge.c_pat_id.isin(js["train_pos"]))
            | (df_merge.c_pat_id.isin(js["train_neg"]))
        ]
        # getting feature space
        cols = [
            "c_value",
            "c_value_apache",
            "c_value_coma",
            "c_value_adm",
            "c_value_inf",
            "c_value_acid",
            "c_value_morph",
            "c_value_sed",
            "c_value_emg",
            "c_value_urea",
        ]
        X_train = df_merge_train[cols]
        X_test = df_merge_test[cols]
        # applying the prediction model
        metrics_dict_all = []

        pred_boogaard_recalibrated = BoogaardPredictorRecalibrated()
        pred = X_test.apply(
            lambda x: pred_boogaard_recalibrated.predict_outcome(x), axis=1
        )
        metrics_dict_test = self.mt.get_metrics(
            targets=list(df_merge_test.c_target), pred=list(pred), thres=0.5,
        )
        metrics_dict_test["model"] = "boogard_rec"
        metrics_dict_test["type"] = "test"
        metrics_dict_test["nr"] = model_nr
        metrics_dict_all.append(metrics_dict_test)

        df_val_res = self.dm.perform_bootstrap_lr(
            X_test,
            list(df_merge_test.c_target),
            pred_boogaard_recalibrated,
            self.bootstrap_rounds,
            df_val_res,
            "boogard_rec",
            model_nr,
            self.mt,
            False,
            True,
        )

        # plotting the auc
        auc_plot = AUCPlotter()
        auc_plot.plot_roc_auc(
            targets=list(df_merge_test.c_target),
            probabilitites=list(pred),
            model_name="Boogaard_pretr_rec_auroc:{}".format(
                metrics_dict_test["auc-roc"].round(3)
            ),
            ax=ax_roc,
            draw_random=False,
            draw_ticks=False,
            linestyle=":",
        )
        auc_plot.plot_auc_pre_rec(
            targets=list(df_merge_test.c_target),
            probabilitites=list(pred),
            model_name="Boogaard_pretr_rec_auprc:{}".format(
                metrics_dict_test["auc-pr"].round(3)
            ),
            ax=ax_pr,
            draw_random=False,
            draw_ticks=False,
            linestyle=":",
        )
        if model_nr == 6:
            auc_plot.plot_roc_auc(
                targets=list(df_merge_test.c_target),
                probabilitites=list(pred),
                model_name="Boogaard_pretr_rec_auroc:{}".format(
                    metrics_dict_test["auc-roc"].round(3)
                ),
                ax=ax_roc_pr[1][0],
                draw_random=False,
                draw_ticks=False,
                linestyle=":",
            )
            auc_plot.plot_auc_pre_rec(
                targets=list(df_merge_test.c_target),
                probabilitites=list(pred),
                model_name="Boogaard_pretr_rec_auprc:{}".format(
                    metrics_dict_test["auc-pr"].round(3)
                ),
                ax=ax_roc_pr[1][1],
                draw_random=False,
                draw_ticks=False,
                linestyle=":",
            )
        pred_boogaard = BoogaardPredictor()
        # make the actual prediction
        df_merge_test["p_model"] = X_test.apply(
            lambda x: pred_boogaard.predict_outcome(x), axis=1
        )
        metrics_dict_test = self.mt.get_metrics(
            targets=list(df_merge_test.c_target),
            pred=list(df_merge_test.p_model),
            thres=0.5,
        )
        metrics_dict_test["model"] = "boogard"
        metrics_dict_test["type"] = "test"
        metrics_dict_test["nr"] = model_nr

        df_val_res = self.dm.perform_bootstrap_lr(
            X_test,
            list(df_merge_test.c_target),
            pred_boogaard_recalibrated,
            self.bootstrap_rounds,
            df_val_res,
            "boogard",
            model_nr,
            self.mt,
            False,
            True,
        )

        metrics_dict_all.append(metrics_dict_test)
        auc_plot.plot_roc_auc(
            targets=list(df_merge_test.c_target),
            probabilitites=list(df_merge_test.p_model),
            model_name="Boogaard_pretr_auroc:{}".format(
                metrics_dict_test["auc-roc"].round(3)
            ),
            ax=ax_roc,
            draw_random=False,
            draw_ticks=False,
            linestyle=":",
        )
        auc_plot.plot_auc_pre_rec(
            targets=list(df_merge_test.c_target),
            probabilitites=list(df_merge_test.p_model),
            model_name="Boogaard_pretr_auprc:{}".format(
                metrics_dict_test["auc-pr"].round(3)
            ),
            ax=ax_pr,
            draw_random=False,
            draw_ticks=False,
            linestyle=":",
        )
        if model_nr == 6:
            auc_plot.plot_roc_auc(
                targets=list(df_merge_test.c_target),
                probabilitites=list(df_merge_test.p_model),
                model_name="Boogaard_pretr_auroc:{}".format(
                    metrics_dict_test["auc-roc"].round(3)
                ),
                ax=ax_roc_pr[1][0],
                draw_random=False,
                draw_ticks=False,
                linestyle=":",
            )
            auc_plot.plot_auc_pre_rec(
                targets=list(df_merge_test.c_target),
                probabilitites=list(df_merge_test.p_model),
                model_name="Boogaard_pretr_auprc:{}".format(
                    metrics_dict_test["auc-pr"].round(3)
                ),
                ax=ax_roc_pr[1][1],
                draw_random=False,
                draw_ticks=False,
                linestyle=":",
            )
        num_cols = [
            "c_value",
            "c_value_apache",
            "c_value_morph",
            "c_value_coma",
            "c_value_adm",
            "c_value_urea",
        ]
        ax_roc, ax_pr, ax_roc_pr, df_val_res = self.retrain_lr_and_mlp(
            X_train,
            X_test,
            df_merge_test,
            df_merge_train,
            ax_roc,
            ax_pr,
            ax_roc_pr,
            num_cols,
            "Boogard",
            model_nr,
            ":",
            df_val_res,
        )
        for test_dict in metrics_dict_all:
            df_val_res = df_val_res.append(
                pd.DataFrame(test_dict, index=[0]), ignore_index=True
            )
        return ax_roc, ax_pr, ax_roc_pr, metrics_dict_all, df_val_res

    def evaluate_wassenaar_model(self, ax_roc, ax_pr, ax_roc_pr, model_nr, df_val_res):
        features_wassenaar = [
            "age",
            "history_of_cognitive_impairment",
            "drinking_history",
            "emergency_admission",
            "nibp_mean",
            "drug_class_corticosteroids",
            "respiratory_failure",
            "urea_nitrogen_in_blood",
        ]
        metrics_dict_all = []
        data_holder = self.dm.load_features(features_wassenaar, model_nr)
        # get first entry measured before pod
        data_holder["nibp_mean"] = self.dm.get_first_value(data_holder["nibp_mean"])
        # convert blood urea nitrogen from mg/dl to mmol/l
        df_bun = data_holder["urea_nitrogen_in_blood"]
        df_bun = df_bun.assign(
            blood_urea_nitrogen=[(float(x) * 0.170) for x in df_bun.c_value]
        )
        # select first known entry for blood urea nitrogen and reassign
        data_holder["urea_nitrogen_in_blood"] = self.dm.get_first_value(df_bun)
        # merge dfs either inner or outer depending on required on missing
        df_merge = (
            DataManager()
            .master_df[["c_op_id", "c_case_id", "c_pat_id", "c_target"]]
            .merge(
                data_holder["age"][["c_case_id", "c_value"]],
                on="c_case_id",
                how="left",
                suffixes=["", "_age"],
            )
            .merge(
                data_holder["drinking_history"][["c_case_id", "c_value"]],
                on="c_case_id",
                how="left",
                suffixes=["", "_drink"],
            )
            .merge(
                data_holder["emergency_admission"][["c_case_id", "c_value"]],
                on="c_case_id",
                how="left",
                suffixes=["", "_emg"],
            )
            .assign(c_value_adm=0)
            .merge(
                data_holder["history_of_cognitive_impairment"][
                    ["c_case_id", "c_value"]
                ],
                on="c_case_id",
                how="left",
                suffixes=["", "_cog"],
            )
            .merge(
                data_holder["drug_class_corticosteroids"][["c_case_id", "c_value"]],
                on="c_case_id",
                how="left",
                suffixes=["", "_cort"],
            )
            .merge(
                data_holder["urea_nitrogen_in_blood"][["c_op_id", "c_value"]],
                on="c_op_id",
                how="left",
                suffixes=["", "_bun"],
            )
            .merge(
                data_holder["nibp_mean"][["c_op_id", "c_value"]],
                on="c_op_id",
                how="left",
                suffixes=["", "_map"],
            )
            .merge(
                data_holder["respiratory_failure"][["c_case_id", "c_value"]],
                on="c_case_id",
                how="left",
                suffixes=["", "_resp"],
            )
            .drop_duplicates()
        )
        # apply median imputation technique
        for col in list(
            df_merge.drop(
                columns=["c_op_id", "c_case_id", "c_pat_id", "c_target"]
            ).columns
        ):
            df_merge[col] = df_merge[col].fillna((df_merge[col].median()))
        jsc = JSConverter()
        js = jsc.read_js_file("./data/meta/cohort/train_test_split")
        df_merge_test = df_merge[
            (df_merge.c_pat_id.isin(js["test_pos"]))
            | (df_merge.c_pat_id.isin(js["test_neg"]))
        ]
        df_merge_train = df_merge[
            (df_merge.c_pat_id.isin(js["train_pos"]))
            | (df_merge.c_pat_id.isin(js["train_neg"]))
        ]
        cols = [
            "c_value",
            "c_value_cog",
            "c_value_drink",
            "c_value_adm",
            "c_value_emg",
            "c_value_map",
            "c_value_cort",
            "c_value_resp",
            "c_value_bun",
        ]
        X_train = df_merge_train[cols]
        X_test = df_merge_test[cols]
        predictor = WassenaarPredictor()
        pred = X_test.apply(lambda x: predictor.predict_outcome(x), axis=1)
        metrics_dict_test = self.mt.get_metrics(
            targets=list(df_merge_test.c_target), pred=list(pred), thres=0.5
        )
        metrics_dict_test["model"] = "wassenaar"
        metrics_dict_test["type"] = "test"
        metrics_dict_test["nr"] = model_nr

        df_val_res = self.dm.perform_bootstrap_lr(
            X_test,
            df_merge_test.c_target,
            predictor,
            self.bootstrap_rounds,
            df_val_res,
            "wassenaar",
            model_nr,
            self.mt,
            False,
            True,
        )

        metrics_dict_all.append(metrics_dict_test)
        for test_dict in metrics_dict_all:
            df_val_res = df_val_res.append(
                pd.DataFrame(test_dict, index=[0]), ignore_index=True
            )
        # plotting the auc
        auc_plot = AUCPlotter()
        auc_plot.plot_roc_auc(
            targets=list(df_merge_test.c_target),
            probabilitites=list(pred),
            model_name="Wassenaar_pretr_auroc:{}".format(
                metrics_dict_test["auc-roc"].round(3)
            ),
            ax=ax_roc,
            draw_random=False,
            draw_ticks=False,
            linestyle="-.",
        )
        auc_plot.plot_auc_pre_rec(
            targets=list(df_merge_test.c_target),
            probabilitites=list(pred),
            model_name="Wassenaar_pretr_auprc:{}".format(
                metrics_dict_test["auc-pr"].round(3)
            ),
            ax=ax_pr,
            draw_random=False,
            draw_ticks=False,
            linestyle="-.",
        )
        if model_nr == 6:
            auc_plot.plot_roc_auc(
                targets=list(df_merge_test.c_target),
                probabilitites=list(pred),
                model_name="Wassenaar_pretr_auroc:{}".format(
                    metrics_dict_test["auc-roc"].round(3)
                ),
                ax=ax_roc_pr[1][0],
                draw_random=False,
                draw_ticks=False,
                linestyle="-.",
            )
            auc_plot.plot_auc_pre_rec(
                targets=list(df_merge_test.c_target),
                probabilitites=list(pred),
                model_name="Wassenaar_pretr_auprc:{}".format(
                    metrics_dict_test["auc-pr"].round(3)
                ),
                ax=ax_roc_pr[1][1],
                draw_random=False,
                draw_ticks=False,
                linestyle="-.",
            )
        num_cols = ["c_value", "c_value_map", "c_value_resp", "c_value_bun"]
        ax_roc, ax_pr, ax_roc_pr, df_val_res = self.retrain_lr_and_mlp(
            X_train,
            X_test,
            df_merge_test,
            df_merge_train,
            ax_roc,
            ax_pr,
            ax_roc_pr,
            num_cols,
            "Wassenaar",
            model_nr,
            "-.",
            df_val_res,
        )

        return ax_roc, ax_pr, ax_roc_pr, metrics_dict_all, df_val_res

    def evaluate_comp_models(self):
        f_roc, ax_roc = plt.subplots(figsize=(10, 8))
        f_pr, ax_pr = plt.subplots(figsize=(10, 8))
        # ax_roc, ax_pr = self.evaluate_boogard_model(ax_roc, ax_pr, 2)
        ax_roc, ax_pr = self.evaluate_wassenaar_model(ax_roc, ax_pr, 2)
        f_roc.savefig(
            "./plots/interpreted/best_models/eval_graphs/comp_auroc_summary.png"
        )
        f_pr.savefig(
            "./plots/interpreted/best_models/eval_graphs/comp_auprc_summary.png"
        )

