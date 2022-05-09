from datetime import timedelta
from pandas.core.indexes.base import Index
from preprocessing_layer.data_manager import DataManager
from preprocessing_layer.time_manager import TimeManager
from extraction_layer.support_classes.js_converter import JSConverter
from typing import Dict
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from extraction_layer.support_classes.meta_manager import MetaManager


class DescriptiveStatistic:
    def __init__(self) -> None:
        self.tm = TimeManager()
        self.dm = DataManager()
        self.descr_ehr = pd.DataFrame()

    def __append_ehr_meta_data(self, jso: Dict = {}, unit: str = ""):

        desc = pd.DataFrame({"col": jso["c_value"]})["col"].describe()

        self.descr_ehr = self.descr_ehr.append(
            pd.DataFrame(
                {
                    "id": [jso["c_id"]],
                    "feature": [jso["c_name"]],
                    "set": [jso["c_set"]],
                    "domain": [jso["c_domain"]],
                    "unit": [unit],
                    "tl": [jso["c_time_line"]],
                    "n_data_points": [desc["count"]],
                    "missingness_pos": [jso["c_missing_pos"]],
                    "missingness_neg": [jso["c_missing_neg"]],
                    "missingness_mean": [
                        (jso["c_missing_neg"] + jso["c_missing_pos"]) / 2
                    ],
                    "min": [desc["min"]],
                    "max": [desc["max"]],
                    "mean": [desc["mean"]],
                    "std": [desc["std"]],
                    "25_perc": [desc["25%"]],
                    "50_perc": [desc["50%"]],
                    "75_perc": [desc["75%"]],
                }
            )
        )

    def get_descriptive_ehr_stats(self):

        # TODO apply for prepared features

        jsc = JSConverter()

        for j, ehr_f in enumerate(MetaManager.ehr_prepared_files):

            # skipping nudesc for annotation purposes
            if "/nudesc" in ehr_f:
                continue

            if "/camicu" in ehr_f:
                continue

            # open up json file
            _ = jsc.read_js_file(ehr_f.split(".json")[0])
            # check if values are numeric
            if jsc.is_numeric():
                print(ehr_f)
                unit_list = self.get_unit_list(unit=jsc.data_obj["c_unit"])
                unit = unit_list[0]
                self.__append_ehr_meta_data(jso=jsc.data_obj, unit=unit)

                d = self.descr_ehr

        # store descriptive results
        self.descr_ehr.to_csv("./data/metrics/descriptive/ehr_numeric_descriptions.csv")

    # TODO HANDLE FOR categorical variables
    def get_unit_list(self, unit):
        # check on different raw units per feature
        if type(unit) == type("str"):
            unit_list = [unit]
        else:
            unit_list = list(dict.fromkeys(unit))
        return [unit.strip().lower() for unit in unit_list]

    def describe_col(self, df: DataFrame = None, y: int = 0, col: str = ""):
        d = df[df.c_target == y][col].describe()
        if col == "c_hos_duration":
            d["unit"] = "d"
        elif "_duration" in col:
            d["unit"] = "h"
        elif "age" in col:
            d["unit"] = "y"
        else:
            d["unit"] = ""
        d["y"] = str(y)
        d["name"] = col
        return d

    def get_splitting_information(self):
        split = JSConverter()
        s = split.read_js_file("./data/meta/cohort/train_test_split")
        df_res = pd.DataFrame()
        df = self.dm.master_df
        for id_type in ["c_op_id", "c_case_id", "c_pat_id"]:
            len_pos = len(df[df.c_target == 1][id_type].drop_duplicates())
            len_n = len(df[id_type].drop_duplicates())
            row = {
                "id": id_type,
                "n": len_n,
                "n_pos": len_pos,
                "n_neg": len_n - len_pos,
            }
            df_res = df_res.append(row, ignore_index=True)

        for split_type in ["train", "test"]:
            row = {
                "id": "{}_c_pat_id".format(split_type),
                "n": len(list(dict.fromkeys(s["{}_pos".format(split_type)])))
                + len(list(dict.fromkeys(s["{}_neg".format(split_type)]))),
                "n_pos": len(list(dict.fromkeys(s["{}_pos".format(split_type)]))),
                "n_neg": len(list(dict.fromkeys(s["{}_neg".format(split_type)]))),
            }
            df_res = df_res.append(row, ignore_index=True)

        df_res.to_csv("./data/metrics/descriptive/split_description.csv")

    def get_descriptive_table_1(self):

        df_res = pd.DataFrame()
        df = self.dm.master_df

        # get number of ops, counts and patients
        df_c = pd.DataFrame(
            {
                "n_surgeries": [
                    len((df[df.c_target == 1].c_op_id.drop_duplicates())),
                    len((df[df.c_target == 0].c_op_id.drop_duplicates())),
                ],
                "n_admissions": [
                    len((df[df.c_target == 1].c_case_id.drop_duplicates())),
                    len((df[df.c_target == 0].c_case_id.drop_duplicates())),
                ],
                "n_patients": [
                    len((df[df.c_target == 1].c_pat_id.drop_duplicates())),
                    len((df[df.c_target == 0].c_pat_id.drop_duplicates())),
                ],
                "y": [1, 0],
            },
            index=[1, 2],
        )
        df_c.to_csv("./data/metrics/descriptive/n_descriptions.csv")

        for col in ["c_hos", "c_an", "c_rec"]:
            # calculate duration
            dur_col = "{}_duration".format(col)
            for y in [0, 1]:
                d = self.describe_col(
                    df=df[df["c_time_consistent"] == True][
                        ["c_case_id", "c_pat_id", "c_target"] + [dur_col]
                    ].drop_duplicates(),
                    y=y,
                    col=dur_col,
                )
                df_res = df_res.append(d)

        # increase op count
        df["c_op_count_per_stay"] = [x + 1 for x in df["c_op_count"]]
        df["c_prev_op_count_per_stay"] = df["c_prev_op_count"]

        # get op count information
        for y in [0, 1]:
            d = self.describe_col(
                df=df[
                    ["c_case_id", "c_pat_id", "c_target", "c_op_count_per_stay"]
                ].drop_duplicates(),
                y=y,
                col="c_op_count_per_stay",
            )
            df_res = df_res.append(d)
            d = self.describe_col(
                df=df[
                    ["c_case_id", "c_pat_id", "c_target", "c_prev_op_count_per_stay"]
                ].drop_duplicates(),
                y=y,
                col="c_prev_op_count_per_stay",
            )
            df_res = df_res.append(d)

        other = [
            "type_combination_spinal_epidural_anesthesia",
            "type_infiltration_anesthesia",
            "type_peripheral_block_anesthesia",
            "type_topical_local_anesthesia",
            "type_interscalene_block",
            "type_intravenous_regional_anesthesia",
            "type_n_femoralis_block",
            "type_n_ischiadicus_block",
            "type_pesc_block",
            "type_plexus_brachialis_block",
            "type_psoas_compartment_block",
            "type_stand_by_anesthesia",
            "type_tap_block",
        ]

        # get admission information maybe do with age and gender, emergency admission, icu
        map = {
            "c_prev_admission": {
                "path": "./data/raw/hospitalization/number_of_previous_admissions",
                "col": "c_case_id",
            },
            "c_age": {"path": "./data/raw/demographics/age", "col": "c_pat_id"},
            "c_gender": {"path": "./data/raw/demographics/gender", "col": "c_pat_id"},
            "c_bmi": {"path": "./data/raw/demographics/bmi", "col": "c_pat_id"},
            "c_type_spinal_anesthesia": {
                "path": "./data/raw/anesthesia_procedures/type_spinal_anesthesia",
                "col": "c_op_id",
            },
            "c_type_total_intravenous_anesthesia": {
                "path": "./data/raw/anesthesia_procedures/type_total_intravenous_anesthesia",
                "col": "c_op_id",
            },
            "c_type_general_balanced_anesthesia": {
                "path": "./data/raw/anesthesia_procedures/type_general_balanced_anesthesia",
                "col": "c_op_id",
            },
            "c_type_epidural_anesthesia": {
                "path": "./data/raw/anesthesia_procedures/type_epidural_anesthesia",
                "col": "c_op_id",
            },
            "c_type_analgo_sedation": {
                "path": "./data/raw/anesthesia_procedures/type_analgo_sedation",
                "col": "c_op_id",
            },
            "c_type_other": {"paths": other, "col": "c_op_id"},
        }

        dm = DataManager()
        for col in list(map.keys()):
            # iterate through col map and append top
            if "path" in list(map[col].keys()):
                jsc = JSConverter()
                jsc.read_js_file(map[col]["path"])
                df_col = jsc.to_df()
            else:
                df_col = pd.DataFrame()
                for p in map[col]["paths"]:
                    jsc = JSConverter()
                    jsc.read_js_file("./data/raw/anesthesia_procedures/" + p)
                    df_col = df_col.append(jsc.to_df(), ignore_index=True)
            if col == "c_gender":
                df_col = df_col[df_col.c_value.str.strip() == "M"]
            df_col = dm.merge_with_master_df(df=df_col)
            df_col = df_col.rename({"c_value": col}, axis=1)
            col_name = map[col]["col"]
            for y in [0, 1]:
                d = self.describe_col(
                    df=df_col[[col_name, "c_target"] + [col]].drop_duplicates(),
                    y=y,
                    col=col,
                )
                df_res = df_res.append(d)

        df_res = df_res.sort_values(["name", "y", "unit"])
        df_res["y"] = [1 if x == "1" else 0 for x in df_res.y]

        print(
            "time inconsistency in data is {} from all hospitalizations".format(
                str(
                    len(df[df.c_time_consistent == False].c_case_id.drop_duplicates())
                    / len(df.c_case_id.drop_duplicates())
                )
            )
        )

        df_res.fillna(" ").merge(df_c, on=["y"]).to_csv(
            "./data/metrics/descriptive/description_table_1.csv"
        )
        print("statistic stored")

    def getting_missingness_stats(self):

        df = pd.read_csv(
            "./data/metrics/descriptive/{}.csv".format(
                MetaManager.get_last_available_features()
            ),
            index_col=0,
        )
        df = df.assign(miss_train_mean=(df.miss_train_pos + df.miss_train_neg) / 2)
        types = {
            "categorical": {
                "is_numeric": False,
                "cols": ["odds_ratio_train_normalized"],
            },
            "numerical": {
                "is_numeric": True,
                "cols": ["miss_train_mean", "miss_train"],
            },
        }
        # iterate categorical or numerical missing stats
        for typ in types:

            df_stat = (
                df[(df.numeric == types[typ]["is_numeric"])]
                .groupby(["domain", "tl"])
                .mean()
            )
            # ehr missingness per domain and tl
            df_stat.to_csv(
                "./data/metrics/descriptive/{}_ehr_missingness_per_domain_and_tl.csv".format(
                    typ
                )
            )
            # numeric mean ehr missingness per domain
            df_stat = (
                df[(df.numeric == types[typ]["is_numeric"])]
                .groupby(["domain"])
                .mean()
                .reset_index()[["domain"] + types[typ]["cols"]]
                .sort_values(types[typ]["cols"][0], ascending=False)
            )
            df_stat.to_csv(
                "./data/metrics/descriptive/{}_ehr_mean_train_missingness_per_domain.csv".format(
                    typ
                )
            )
            # numeric mean ehr missingness per tl
            df_stat = (
                df[(df.numeric == types[typ]["is_numeric"])]
                .groupby(["tl"])
                .mean()
                .reset_index()[["tl"] + types[typ]["cols"]]
                .sort_values(types[typ]["cols"][0], ascending=False)
            )
            df_stat.to_csv(
                "./data/metrics/descriptive/{}_ehr_mean_train_missingness_per_tl.csv".format(
                    typ
                )
            )

    def meta_feature_dict(
        self, jsc: JSConverter = None, tl: int = 0, unit: str = "", sets: dict = {}
    ):
        return {
            "feature": jsc.data_obj["c_name"].lower(),
            "domain": jsc.data_obj["c_domain"].lower(),
            "id": jsc.data_obj["c_id"].lower(),
            "tl": tl,
            "unit": unit,
            "numeric": str(jsc.is_numeric() | jsc.is_duration()),
            "n_test_pos": len(sets["test_df"][sets["test_df"].c_target == 1]),
            "n_test_neg": len(sets["test_df"][sets["test_df"].c_target == 0]),
            "n_train_pos": len(sets["train_df"][sets["train_df"].c_target == 1]),
            "n_train_neg": len(sets["train_df"][sets["train_df"].c_target == 0]),
            "miss_test_pos": sets["miss_test_pos"],
            "miss_test_neg": sets["miss_test_neg"],
            "miss_train_pos": sets["miss_train_pos"],
            "miss_train_neg": sets["miss_train_neg"],
            "miss_train": sets["miss_train"],
            "miss_test": sets["miss_test"],
            "odds_ratio_test": sets["non_zero_test_rate"],
            "odds_ratio_train": sets["non_zero_train_rate"],
        }

    def seq_exp_results_dict(
        self,
        loss: str = "focal",
        tl: int = 1,
        q: int = 1,
        training: dict = {},
        data_dict: dict = {},
        metrics_prefix: int = 0,
        time: timedelta = 0,
        application_id: int = 0,
        perceptron_type: str = "MLP",
        cv_round: int = 0,
    ):

        oversample = data_dict["oversample"]
        undersample = data_dict["undersample"]
        missing_indicator = data_dict["missing_indicator"]
        prev_timelines = data_dict["prev_timelines"]
        drop_sparse_rows = data_dict["drop_sparse_rows"]
        mi_reduction = data_dict["mi_reduction"]
        pat_count_pos = data_dict["pat_count_pos"]
        pat_count_neg = data_dict["pat_count_neg"]
        case_count_pos = data_dict["case_count_pos"]
        case_count_neg = data_dict["case_count_neg"]
        op_count_pos = data_dict["op_count_pos"]
        op_count_neg = data_dict["op_count_neg"]

        return {
            "application_id": application_id,
            "q": q,
            "tl": tl,
            "time": time,
            "pat_count_pos": pat_count_pos,
            "pat_count_neg": pat_count_neg,
            "case_count_pos": case_count_pos,
            "case_count_neg": case_count_neg,
            "op_count_pos": op_count_pos,
            "op_count_neg": op_count_neg,
            "cv_round": cv_round,
            "model": perceptron_type,
            "oversampling": oversample,
            "undersampling": undersample,
            "focal_loss": (not oversample) & (not undersample),
            "prev_timelines": prev_timelines,
            "loss": loss,
            "missing_indicator": missing_indicator,
            "drop_sparse_rows": drop_sparse_rows,
            "mi_reduction": mi_reduction,
            "mean_train_auc_pr": np.mean(training.history["auc_pr"]),
            "mean_val_auc_pr": np.mean(training.history["val_auc_pr"]),
            "mean_train_auc_roc": np.mean(training.history["auc_roc"]),
            "mean_val_auc_roc": np.mean(training.history["val_auc_roc"]),
            "mean_train_precision": np.mean(training.history["precision"]),
            "mean_val_precision": np.mean(training.history["val_precision"]),
            "mean_train_recall": np.mean(training.history["recall"]),
            "mean_val_recall": np.mean(training.history["val_recall"])  }
