import pandas as pd
import numpy as np
import json
from configs import grouping_columns, validation_columns, model_variants
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    roc_curve,
    roc_auc_score,
    f1_score,
)


def get_metrics(targets, pred):
    '''
    calculates performance metrics for model evaluation while training
            Parameters:
                    targets (arr): array of target GT class labels,
                    pred (arr): predictions in form of probabilities
            Returns:
                    metrics (dict)
    '''
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
    sens_spec = sens_spec.assign(
        sum_spec_sens=sens_spec.spec + sens_spec.sens)
    sens_spec = sens_spec[sens_spec.sum_spec_sens ==
                          max(sens_spec.sum_spec_sens)]
    prec_rec = pd.DataFrame({"prec": list(precision), "rec": list(recall)})
    prec_rec_07 = prec_rec[(prec_rec.rec >= 0.7)]
    prec_rec_07 = prec_rec_07[prec_rec_07.prec == max(prec_rec_07.prec)]
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
        "prec": prec_rec_07.iloc[0]["prec"],
        "f1": f_score,
    }


def store_best_models(storing_date: str = "2022_08_30_14_00_00", ml_type="tree"):
    '''
    evaluating of training results and yielding of best performed model per cv
    '''
    cv_results = pd.read_csv(
        "./data/metrics/performance/training_procedure/results_{}_{}_revised.csv".format(ml_type, storing_date), index_col=0)
    # ensure that ml type is really set in results
    cv_results = cv_results[cv_results["ml_type"] == ml_type]
    # split according to losses because scales can not be compared
    loss_types = list(cv_results.loss_type.drop_duplicates())
    for loss in loss_types:
        cv = cv_results[cv_results.loss_type == loss]
        # get mean evaluation metrics accross inner cv folds
        cv = cv.groupby(grouping_columns)[
            validation_columns].mean().reset_index()
        df_res = []
        # iterate through model types
        for model in list(model_variants.keys()):
            # get minimum validation loss according to groups
            cv_model = cv[cv.model == model]
            cv_model = cv_model.assign(roc_pr_sum=cv_model["auc-roc-val"] +
                                       cv_model["auc-pr-val"])
            df_res.append(cv_model[cv_model["loss-val"]
                          == min(cv_model["loss-val"])])
        pd.concat(df_res).to_csv(
            "./data/metrics/performance/training_procedure/best_{}_{}_models_revised.csv".format(ml_type, loss))


def get_metrics_df(metrics_dict_train, metrics_dict_test, cv, cv_validation, incidence_rate, model_name, y_pos, y_neg,
                   loss_type, drop_50, oversampling, training_loss, validation_loss, miss_frac, num_thres, cat_thres,
                   best_config, num_features, cat_features, X, X_len, ml_type):
    '''
    takes parameters and creates df for storing results
            Returns:
                    merics df (df)
    '''
    return pd.DataFrame(
        {
            "cv_hyperparam": [cv],
            "cv_validation": [cv_validation],
            "model": [model_name],
            "ml_type": [ml_type],
            "incidence-rate": [incidence_rate],
            "y_pos": [y_pos],
            "y_neg": [y_neg],
            "loss_type": [loss_type],
            "drop_50": [drop_50],
            "oversample": [oversampling],
            "loss-train": [np.mean(training_loss)],
            "loss-val": [np.mean(validation_loss)],
            "auc-roc-train": [
                metrics_dict_train["auc-roc"]
            ],
            "auc-roc-val": [
                metrics_dict_test["auc-roc"]
            ],
            "auc-pr-train": [
                metrics_dict_train["auc-pr"]
            ],
            "auc-pr-val": [
                metrics_dict_test["auc-pr"]
            ],
            "sens-train": [
                metrics_dict_train["sens"]
            ],
            "sens-val": [metrics_dict_test["sens"]],
            "spec-train": [
                metrics_dict_train["spec"]
            ],
            "spec-val": [metrics_dict_test["spec"]],
            "prec-train": [
                metrics_dict_train["prec"]
            ],
            "prec-val": [metrics_dict_test["prec"]],
            "miss_frac": [miss_frac],
            "num_thres": [num_thres],
            "cat_thres": [cat_thres],
            "config_dict": [json.dumps(best_config)],
            "num_feats": [num_features],
            "cat_feats": [cat_features],
            "X": [X],
            "X_len": [X_len]
        }
    )
