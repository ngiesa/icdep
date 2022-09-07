from numpy import tracemalloc_domain
from extraction_layer.support_classes.meta_manager import MetaManager
from modeling_layer.metrics import get_metrics
from preprocessing_layer.time_manager import TimeManager
from extraction_layer.support_classes.js_converter import JSConverter
from pandas.core.frame import DataFrame
from sklearn.impute import MissingIndicator
import pandas as pd
import numpy as np
from sklearn.utils import resample
import torch
from configs import no_feats


class DataManager:
    def __init__(self) -> None:
        self.tm = TimeManager()
        self.mm = MetaManager()
        self.master_df = pd.read_csv(
            "./data/meta/cohort/master_time_table.csv")
        self.data_holder = {}
        self.no_feats = no_feats

    def merge_with_master_df(self, df: DataFrame = None):
        # merging df with master table reducing on columns for gaining performances
        df = self.master_df.merge(
            df, on=["c_case_id"], suffixes=["_master", ""], how="inner"
        )
        return df

    def locf_within_stay(self, df: DataFrame = None, fillna: bool = False):
        # locf imputation within one stay
        cols = ["c_pat_id", "c_case_id", "c_an_start_ts"]
        df = df.set_index(cols).groupby(cols).ffill().reset_index()
        if fillna:
            return df.filna(0)
        return df

    def convert_time_cons(self, df: DataFrame = None):
        # convert time consisteny variable to binary
        if "c_time_consistent" in list(df.columns):
            return df.assign(
                c_time_consistent=[
                    0 if e == True else 1 for e in df["c_time_consistent"]
                ]
            )
        return df

    def standardize_data(
        self, df: DataFrame = None, metr_dict: dict = None, num_cols: list = []
    ):
        # convert categorical variables to {1, -1}
        df_cat = df.drop(columns=num_cols)
        df_cat = (df_cat - 0.5) * 2
        df_num = df[num_cols]
        # create dict for z-transformation metric
        if metr_dict is None:
            df_mean = df_num.mean()
            df_std = df_num.std()
        else:
            df_mean = metr_dict["mean"]
            df_std = metr_dict["std"]
        # stadardize numerical attributes
        df_num = (df_num - df_mean) / df_std
        # concat both
        df = pd.concat([df_num.reset_index(), df_cat.reset_index()], axis=1).drop(
            columns=["index"]
        )
        return df, {"mean": df_mean, "std": df_std}

    def add_missing_indicator(self, df: DataFrame = None):
        # add indicator for all features included in the feature set
        mi = MissingIndicator(features="all")
        # get all feature names which have missing values
        df_tmp = df.drop(columns=self.no_feats)
        # fit the missing indicator instance
        mi.fit(df_tmp)
        # transform the data to binary flags with speaking column names
        df_miss = pd.DataFrame(
            mi.transform(df_tmp) * 1,
            columns=["{}_miss_indicator".format(
                x) for x in list(df_tmp.columns)],
        )
        miss_cols = list(df_miss.columns)
        miss_cols = [x for x in miss_cols if (
            not "history" in x) & (not "time" in x)]
        # return the concatinated data
        df_miss = df_miss[miss_cols]
        df = pd.concat([df.reset_index(), df_miss.reset_index()], axis=1)
        return df.drop(columns=["index"]), miss_cols

    def cut_na_features(self, fraction: float = 0.5, df: DataFrame = None):
        # get all features in list
        feats = [ele for ele in list(df.columns) if ele not in self.no_feats]
        # count null values per row
        df = df.assign(null_count=df.isnull().sum(axis=1))
        # return rows having higher feat non null count as fraction
        return df[df.null_count < (fraction * len(feats))].drop(
            columns=["null_count"], axis=1
        )

    def select_transformed_values(
        self, df: DataFrame = None, df_trans: DataFrame = None, col: str = "c_value"
    ):
        # merge original and transformed data sets
        df = df.merge(df_trans, on=["c_case_id",
                      "c_op_id"], suffixes=["", "_trans"])
        # filter where transformation condition applies
        df = df[df[col] == df[col + "_trans"]]
        return df

    def load_features(self, features: list = [], model=1):
        models_timestamps = {
            1: {"from": "c_hos_start_ts", "to": "c_an_start_ts"},
            2: {"from": "c_an_start_ts", "to": "c_an_end_ts"},
            3: {"from": "c_an_end_ts", "to": "c_timestamp"},
            4: {"from": "c_hos_start_ts", "to": "c_an_end_ts"},
            5: {"from": "c_an_start_ts", "to": "c_timestamp"},
            6: {"from": "c_hos_start_ts", "to": "c_timestamp"},
        }
        df_master = pd.read_csv("./data/meta/cohort/master_time_table.csv")
        js_con = JSConverter()
        data_holder = {}
        for feat_name in features:
            # iterate through raw json files and search for exact feature name
            path_list = [
                ehr_file
                if ("/{}.json".format(feat_name) in ehr_file) and ("/raw/" in ehr_file)
                else ""
                for ehr_file in self.mm.ehr_raw_files
            ]
            for p in path_list:
                if p != "":
                    print(p)
                    # load data from data path
                    js_con.read_js_file(path=p.split(".json")[0])
                    # make c_value numeric
                    df = js_con.df_str_to_numeric()
                    d = df.head()
                    # filter data on time line if timestamp available
                    if (not "" in set(df.c_start_ts)) & (not "history" in feat_name):
                        master_cols = (
                            ["c_case_id"]
                            + [models_timestamps[model]["from"]]
                            + [models_timestamps[model]["to"]]
                        )
                        df = df.merge(
                            df_master[master_cols], how="inner", on="c_case_id"
                        )
                        df = df[
                            (df.c_start_ts >=
                             df[models_timestamps[model]["from"]])
                            & (df.c_start_ts <= df[models_timestamps[model]["to"]])
                        ]
                        df = df.drop(
                            columns=[models_timestamps[model]["from"]] +
                            [models_timestamps[model]["to"]]
                        )
                    data_holder[feat_name] = df
        self.data_holder = data_holder
        return data_holder

    # assign 0 values for features where just available data was extracted per case
    def assign_zero_cases(self, df: DataFrame = None, time_relevant: bool = False):
        master_data = DataManager().master_df[
            ["c_case_id", "c_pat_id", "c_last_ts", "c_hos_start_ts", "c_op_id"]
        ]
        df = master_data.merge(
            df.drop(columns=["c_pat_id"], axis=1),
            on="c_case_id",
            how="left",
            suffixes=["", "_master"],
        )
        df = df.assign(c_value=df["c_value"].fillna(0))
        # for zero cases, the last ts is used as time stamp where c_start_ts is not empty
        if time_relevant:
            df.c_start_ts.fillna(df.c_last_ts, inplace=True)
            df.c_start_ts.fillna(df.c_hos_start_ts, inplace=True)
        df = df.assign(c_end_ts=df["c_end_ts"].fillna(""))
        df = df.assign(c_op_id=df["c_op_id"].fillna(""))
        df = df.assign(c_unit="")
        return df.drop(columns=["c_id", "c_domain", "c_name"])

    def get_first_value(self, df: DataFrame = None):
        # merging with master data holding annotated information
        df = self.merge_with_master_df(df)
        # reducing on values observed first before pod
        df = df[df.c_start_ts < df.c_timestamp]
        # get minimum time stamp for value before the operation
        df_min = df.groupby(["c_case_id", "c_op_id"])[
            "c_start_ts"].min().reset_index()
        # merging minimum time stamped information with value data
        df = self.select_transformed_values(
            df=df, df_trans=df_min, col="c_start_ts")
        # return and select the attributes
        df = (
            df[["c_value", "c_start_ts", "c_case_id", "c_op_id"]]
            .drop_duplicates()
            .reset_index()
        )
        return df

    def get_highest_value(self, df: DataFrame = None):
        # merging with master data holding annotated information
        df = self.merge_with_master_df(df)
       # get time line 1 from admission
        df = self.tm.get_time_lines(df, 1)
        # get minimum time stamp for value within before the operation
        df_max = df.groupby(["c_case_id", "c_op_id"])[
            "c_value"].max().reset_index()
        # merging df with max value and normal one
        df = self.select_transformed_values(df=df, df_trans=df_max)
        # return and select the attributes
        df = df[["c_value", "c_case_id", "c_op_id"]
                ].drop_duplicates().reset_index()
        return df

    def get_cumulated_value(self, df: DataFrame = None):
        # merging with master data holding annotated information
        df = self.merge_with_master_df(df)
        # get time line 1 from admission
        df = self.tm.get_time_lines(df, 1)
        # get minimum time stamp for value within before the operation
        return df.groupby(["c_case_id", "c_op_id"])["c_value"].sum().reset_index()

    def sample_mean_per_time_unit(self, df: DataFrame = None, time_unit: str = "1H", tl: int = 1):
        df = self.merge_with_master_df(df)
        df = df[(df.c_start_ts >= df.c_hos_start_ts)
                & (df.c_start_ts <= df.c_timestamp)]
        df = df.assign(c_start_ts=pd.to_datetime(df.c_start_ts)).reset_index()
        df = (
            df.groupby(["c_op_id", "c_case_id"])
            .resample(time_unit, on="c_start_ts")["c_value"]
            .mean()
            .reset_index()
        )
        return df

    def perform_random_undersampling(
        self, df: DataFrame = None, incidence_rate: float = 0.5
    ):
        # split pos and neg
        pos = df[df.c_target == 1]
        neg = df[df.c_target == 0]
        # original target information
        print("Origianl N pos:", len(pos))
        print("Original N neg:", len(neg))
        # adapt the incidence rate
        n_neg = int((len(pos) / incidence_rate) - len(pos))
        print("New N neg:", str(n_neg))
        # draw random sample according to incidence rate
        neg = neg.sample(frac=1)
        return neg[0:n_neg].append(pos).sample(frac=1)

    def perform_random_oversampling(
        self, df: DataFrame = None, incidence_rate: float = 0.5
    ):
        # split pos and neg
        pos = df[df.c_target == 1]
        neg = df[df.c_target == 0]
        # original target information
        print("Origianl N pos:", len(pos))
        print("Original N neg:", len(neg))
        # adapt the incidence rate
        n_pos = int((len(neg) / (1 - incidence_rate)) - len(neg))
        print("New N pos:", str(n_pos))
        return neg.append(pos.sample(n=n_pos, replace=True))

    # calc odds as cat missingness as a surrogate to have a feeling of sparsity
    def calc_odds(self, df: DataFrame = None):
        l00 = len(df[(df.c_value == 0) & (df.c_target == 0)])
        l11 = len(df[(df.c_value == 1) & (df.c_target == 1)])
        l10 = len(df[(df.c_value == 1) & (df.c_target == 0)])
        l01 = len(df[(df.c_value == 0) & (df.c_target == 1)])
        if (l00 == 0) | (l11 == 0) | (l10 == 0) | (l01 == 0):
            return 0
        oddsr = (l00 * l11) / (l10 * l01)
        return oddsr

    # assign test train cohort to data
    def get_train_test_df(self, df: DataFrame = None, isbinary=False):
        split = JSConverter()
        split.read_js_file("./data/meta/cohort/train_test_split")

        # getting the data according to the splitted train and test patients
        df_test_pos = df[df.c_pat_id.isin(split.data_obj["test_pos"])]
        df_test_neg = df[df.c_pat_id.isin(split.data_obj["test_neg"])]
        df_train_pos = df[df.c_pat_id.isin(split.data_obj["train_pos"])]
        df_train_neg = df[df.c_pat_id.isin(split.data_obj["train_neg"])]

        # counting zros vs non zero values for information content
        df_test = df_test_pos.append(df_test_neg)
        df_train = df_train_pos.append(df_train_neg)

        odds_ratio_test = 0
        odds_ratio_train = 0
        if isbinary:
            if not df_test.empty:
                odds_ratio_test = abs(self.calc_odds(df_test) - 1) * 2
            if not df_train.empty:
                odds_ratio_train = abs(self.calc_odds(df_train) - 1) * 2

        _, miss_train = self.get_missing(
            df_train,
            len(
                list(
                    dict.fromkeys(
                        split.data_obj["train_neg"] +
                        split.data_obj["train_pos"]
                    )
                )
            ),
        )
        _, miss_test = self.get_missing(
            df_test,
            len(
                list(
                    dict.fromkeys(
                        split.data_obj["test_neg"] + split.data_obj["test_pos"]
                    )
                )
            ),
        )

        # assign missing rates and skip data that is not available according to rate
        test_pos_df, miss_test_pos = self.get_missing(
            df_test_pos, len(list(dict.fromkeys(split.data_obj["test_pos"])))
        )
        test_neg_df, miss_test_neg = self.get_missing(
            df_test_neg, len(list(dict.fromkeys(split.data_obj["test_neg"])))
        )

        train_pos_df, miss_train_pos = self.get_missing(
            df_train_pos, len(list(dict.fromkeys(split.data_obj["train_pos"])))
        )
        train_neg_df, miss_train_neg = self.get_missing(
            df_train_neg, len(list(dict.fromkeys(split.data_obj["train_neg"])))
        )

        return {
            "test_df": test_pos_df.append(test_neg_df),
            "train_df": train_pos_df.append(train_neg_df),
            "miss_train": miss_train,
            "miss_test": miss_test,
            "miss_test_pos": miss_test_pos,
            "miss_test_neg": miss_test_neg,
            "miss_train_pos": miss_train_pos,
            "miss_train_neg": miss_train_neg,
            "non_zero_test_rate": odds_ratio_test,
            "non_zero_train_rate": odds_ratio_train,
        }

    # select features over the missing rate in test, train for y1 and y0
    def get_missing(self, df: DataFrame = None, ref: int = 0):
        pat_count = len(df.c_pat_id.drop_duplicates())
        miss_rate = 1 - pat_count / ref
        if df.empty:
            miss_rate = 1
        return df, miss_rate

    def perform_bootstrap_lr(
        self,
        X_test,
        y_test,
        clf,
        boot_strap_iterations,
        df_val_res,
        model_name,
        i,
        mt,
        mlp=False,
        apply_only=False,
    ):
        print(len(X_test))
        print(len(y_test))

        if torch.is_tensor(X_test):
            X_test = X_test.detach().numpy()
        if torch.is_tensor(y_test):
            y_test = y_test.detach().numpy()

        for boot_id in range(1, boot_strap_iterations + 1):

            if mlp:
                X_test_strap, y_test_strap = resample(
                    X_test, y_test, replace=True, stratify=y_test
                )
                pred = clf(torch.tensor(X_test_strap)).reshape(-1)
                metrics_dict_test = get_metrics(
                    targets=y_test_strap, pred=pred.detach().numpy()
                )
                metrics_dict_test["model"] = model_name + "_mlp"
                metrics_dict_test["type"] = "test"
                metrics_dict_test["nr"] = str(i + 1) + "," + str(boot_id)
                df_val_res = df_val_res.append(
                    pd.DataFrame(metrics_dict_test, index=[0]), ignore_index=True
                )
            elif apply_only:
                pred = X_test.apply(lambda x: clf.predict_outcome(x), axis=1)
                metrics_dict_test = get_metrics(
                    targets=list(y_test), pred=list(pred)
                )
                metrics_dict_test["model"] = model_name + "_org"
                metrics_dict_test["type"] = "test"
                metrics_dict_test["nr"] = str(i + 1) + "," + str(boot_id)
                df_val_res = df_val_res.append(
                    pd.DataFrame(metrics_dict_test, index=[0]), ignore_index=True
                )

            else:
                X_test_strap, y_test_strap = resample(
                    X_test, y_test, replace=True, stratify=y_test
                )
                prob = clf.predict_proba(np.nan_to_num(X_test_strap))
                prob = [x[1] for x in prob]
                metrics_dict_test = get_metrics(
                    targets=y_test_strap, pred=prob,
                )
                metrics_dict_test["model"] = model_name + "_lr"
                metrics_dict_test["type"] = "test"
                metrics_dict_test["nr"] = str(i + 1) + "," + str(boot_id)
                df_val_res = df_val_res.append(
                    pd.DataFrame(metrics_dict_test, index=[0]), ignore_index=True
                )
        return df_val_res
