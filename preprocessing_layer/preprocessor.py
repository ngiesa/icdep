from preprocessing_layer.data_manager import DataManager
from extraction_layer.support_classes.js_converter import JSConverter
import gc
import pandas as pd
from pandas.core.frame import DataFrame
from extraction_layer.support_classes.meta_manager import MetaManager
from preprocessing_layer.time_manager import TimeManager
from preprocessing_layer.test_train_splitter import TrainTestSplitter
from statistic_layer.descriptive_statistic import DescriptiveStatistic
import datetime
from datetime import datetime


class Preprocessor:
    def __init__(self, desc_only=False) -> None:
        self.availability_rate = 5e-2
        self.minimal_non_zero_rate = 1e-4
        self.dm = DataManager()
        self.tm = TimeManager()
        self.desc_only = desc_only
        self.missing_df = pd.DataFrame()
        self.available_df = pd.DataFrame()
        self.tts = TrainTestSplitter()
        self.s = DescriptiveStatistic()

    # assign dummy values for plotting if minority class is empty
    def prepare_plot_df(self, df: DataFrame = None):
        df = df[["c_value", "c_target"]]
        for y in [1.0, 0.0]:
            if len(df[df["c_target"] == y]) == 0:
                df = df.append(
                    pd.DataFrame({"c_value": [0], "c_target": [y]}), ignore_index=True
                )
        return df

    # get unit per list
    def get_unit_list(self, unit):
        # check on different raw units per feature
        if type(unit) == type("str"):
            unit_list = [unit]
        else:
            unit_list = list(dict.fromkeys(unit))
        return [unit.strip().lower() for unit in unit_list]

    # cutting outliers according to input data
    def cut_boundaries(
        self,
        df: DataFrame = None,
        col_name: str = "c_value",
        lower_bound: float = 0.0,
        upper_bound: float = 0.0,
    ):

        # cut value according to defined valid boundaries
        df[col_name] = [
            x
            if (str(x) != "") & (float(x) <= upper_bound) & (float(x) >= lower_bound)
            else float("NaN")
            for x in list(df[col_name])
        ]
        return df[df[col_name].notna()]

    # filter and reassigning converted data
    def convert_data(
        self,
        df: DataFrame = None,
        org_unit: str = "",
        targ_unit: str = "",
        factor: float = 0.0,
    ):
        df["c_unit"] = [u.lower().strip() for u in df["c_unit"]]
        df_tmp = df[df.c_unit == org_unit]
        df_tmp = df_tmp.assign(c_value=float(factor) * df_tmp.c_value)
        df_tmp = df_tmp.assign(c_unit=targ_unit)
        df = df[df.c_unit != org_unit].append(df_tmp, ignore_index=True)
        return df

    # conversions for laboratory values
    def convert_lab_units(
        self, name: str = "", df: DataFrame = None, unit_list: list = []
    ):
        conv = MetaManager.get_lab_conversions()
        name_split = [n.strip().lower() for n in name.split(" ")]
        conv = conv[conv["Original_Unit"].isin(unit_list)]
        conv = conv[(conv.parameter.isin(name_split)) & (conv.carrier.isin(name_split))]
        if len(conv) < 1:
            # nothing to convert
            return df
        for i, c in conv.iterrows():
            df = self.convert_data(
                df=df,
                org_unit=c["Original_Unit"],
                targ_unit=c["SI_Unit"],
                factor=c["Factor"],
            )
        return df

    # conversion for input values
    def convert_input_units(self, df: DataFrame = None, unit_list: list = []):
        conv = MetaManager.get_med_conversions()
        conv["Original_Unit"] = [u.lower().strip() for u in conv["Original_Unit"]]
        conv = conv[conv["Original_Unit"].isin(unit_list)]
        if len(conv) < 1:
            # nothing to convert
            return df
        for i, c in conv.iterrows():
            df = self.convert_data(
                df=df,
                org_unit=c["Original_Unit"],
                targ_unit=c["Target_Unit"],
                factor=c["Factor"],
            )
        return df

    def rename_op_related_times(self, df: DataFrame = None, name: str = ""):
        # handling op times seperately because basic preprocessing is done via the time manager
        name_col_map = {
            "hospitalization_time": "c_hos_{}_ts",
            "recovery_room_time": "c_rec_{}_ts",
            "surgery_time": "c_op_{}_ts",
            "aneasthesia_time": "c_an_{}_ts",
        }
        df["c_start_ts"] = df[name_col_map[name].format("start")]
        df["c_end_ts"] = df[name_col_map[name].format("end")]

        return df

    # unit conversions and boundary selection
    def preprocess_ehr_feature(self, jsc: JSConverter = None):

        # get name and domain

        if type([2, 3, 4]) == type(jsc.data_obj["c_name"]):
            name = jsc.data_obj["c_name"][0]
        else:
            name = jsc.data_obj["c_name"].strip().replace(" ", "_").lower()

        if type([2, 3, 4]) == type(jsc.data_obj["c_domain"]):
            domain = jsc.data_obj["c_domain"][0]
        else:
            domain = jsc.data_obj["c_domain"].strip().replace(" ", "_").lower()

        # basic preprocessing before applying the timelines
        if jsc.is_numeric():
            # preprocess numeric features
            md = MetaManager.get_meta_data()
            md = md[
                md["var_name"].str.replace(" ", "")
                == jsc.data_obj["c_name"].replace(" ", "").lower()
            ]

            # check if meta data could be read
            if len(md) < 1:
                print("meta data was not found for feature")
                return None, None

            # read units
            unit_list = self.get_unit_list(unit=jsc.data_obj["c_unit"])

            # create df on selected keys
            df = jsc.df_str_to_numeric()

            # apply laboratory or medication unit conversions
            if domain == "laboratory":
                df = self.convert_lab_units(
                    df=df, unit_list=unit_list, name=jsc.data_obj["c_name"]
                )
            if domain == "inputs":
                df = self.convert_input_units(df=df, unit_list=unit_list)

            # assign zero for number of diagnosis
            if domain == "comorbidities_and_diagnosis":
                df = self.dm.assign_zero_cases(df, time_relevant=jsc.is_time_relevant())

            # check if bounds are defined in meta data and cut off value
            if (md["UpperBound"].iloc[0] != "") & (md["LowerBound"].iloc[0] != ""):
                df = self.cut_boundaries(
                    df=df,
                    lower_bound=float(md["LowerBound"].iloc[0]),
                    upper_bound=float(md["UpperBound"].iloc[0]),
                )

            # merge with master
            df = self.dm.merge_with_master_df(df=df)

            # process surgery counts seperately by using master data
            if name == "number_of_surgeries":
                df = df.assign(c_value=df["c_op_count"])
                df = df.assign(c_start_ts=[""] * len(df))

            if name == "number_of_previous_surgeries":
                df = df.assign(c_value=df["c_prev_op_count"])
                df = df.assign(c_start_ts=[""] * len(df))

            # calucalte median for body length and heights
            if name in ["body_weight", "body_length"]:
                df = (
                    df.groupby(
                        ["c_case_id", "c_pat_id", "c_op_id", "c_target", "c_unit"]
                    )["c_value"]
                    .median()
                    .reset_index()
                )
                df = df.assign(c_start_ts=[""] * len(df))
                df = df.assign(c_end_ts=[""] * len(df))
                
            return jsc, df

        elif jsc.is_duration():
            # calculate duration according to time lines in time manager later
            df = jsc.to_df()

            # merge with master
            df = self.dm.merge_with_master_df(df=df)

            # preprocess surgery related times seperately
            if name in [
                "hospitalization_time",
                "recovery_room_time",
                "surgery_time",
                "aneasthesia_time",
            ]:
                df = self.rename_op_related_times(df=df, name=name)

            return jsc, df

        elif jsc.is_binary():
            # convert binary features
            df = jsc.to_df()
            if name == "gender":
                df = df.assign(
                    c_value=[
                        1 if str(x).lower().replace(" ", "") == "m" else 0
                        for x in df["c_value"]
                    ]
                )
                df = self.dm.merge_with_master_df(df=df)
                return jsc, df
            else:
                df = df.assign(
                    c_value=[
                        0 if (str(x).isnumeric() and (float(x) == 0.0)) else 1
                        for x in df["c_value"]
                    ]
                )
            # assign zero cases
            df = self.dm.assign_zero_cases(df, time_relevant=jsc.is_time_relevant())
            # merge with master
            df = self.dm.merge_with_master_df(df=df)
            return jsc, df
        return None, None

    def min_max_scaling(self, df, colname):
        mi = min(list(df[colname]))
        ma = max(list(df[colname]))
        df[colname] = [(x - mi) / (ma - mi) for x in df[colname]]
        return df

    def get_time_set_prepared_features(
        self, tl: int = 0, df: DataFrame = None, jsc: JSConverter = None, unit: str = ""
    ):

        print("Time Line: ", str(tl))

        name = jsc.data_obj["c_name"].strip().replace(" ", "_")

        if tl != 0:
            df = self.tm.get_time_lines(
                df, tl=tl, is_duration=jsc.is_duration(), name=name
            )

        # return none if nothing to process
        if (df is None) or (df.empty):
            print("no data in time line")
            return

        # label 1 if at least once the variable was encoded for the patient and surgery
        if ((tl != 0) & (jsc.is_binary())) or (
            jsc.data_obj["c_name"].lower().replace(" ", "")
            == "numberofpreviousadmissions"
        ):
            df = (
                df.groupby(["c_case_id", "c_pat_id", "c_op_id", "c_target", "c_unit"])[
                    "c_value"
                ]
                .max()
                .reset_index()
            )
            df = df.assign(c_start_ts=[""] * len(df))
            df = df.assign(c_end_ts=[""] * len(df))

        # sum up durations like ventilation etc. maybe enhance features with count of support procedures etc.
        if jsc.is_duration():
            df = (
                df.groupby(["c_case_id", "c_pat_id", "c_op_id", "c_target", "c_unit"])[
                    "c_value"
                ]
                .sum()
                .reset_index()
            )
            df = df.assign(c_start_ts=[""] * len(df))
            df = df.assign(c_end_ts=[""] * len(df))

        # get the train test sets
        sets = self.tts.get_train_test_df(df, jsc.is_binary())

        df = df[
            [
                "c_value",
                "c_pat_id",
                "c_case_id",
                "c_op_id",
                "c_start_ts",
                "c_end_ts",
                "c_unit",
                "c_target",
            ]
        ].drop_duplicates()

        # print exclusion criteria
        print(
            "Missing Rates Pos y=1 for Patients in Training Set: ",
            sets["miss_train_pos"],
        )
        print(
            "Missing Rates Neg y=0 for Patients in Training Set: ",
            sets["miss_train_neg"],
        )
        print("Non Zero Rate in Training Set: ", sets["non_zero_train_rate"])

        # check on missign feature and store all missing in seperate df
        if (
            (
                (sets["miss_train_pos"] > 1 - self.availability_rate)
                & (not jsc.is_binary())
            )
            | (
                (sets["miss_train_neg"] > 1 - self.availability_rate)
                & (not jsc.is_binary())
            )
            | (
                jsc.is_binary()
                & (sets["non_zero_train_rate"] < self.minimal_non_zero_rate)
            )
        ):
            print(
                "more than {}% of patients with no data or zero value rate < {}% on training set".format(
                    str(self.availability_rate * 100),
                    str(self.minimal_non_zero_rate * 100),
                )
            )
            self.missing_df = self.missing_df.append(
                self.s.meta_feature_dict(jsc=jsc, tl=tl, unit=unit, sets=sets),
                ignore_index=True,
            )
            return

        # store available data
        self.available_df = self.available_df.append(
            self.s.meta_feature_dict(jsc=jsc, tl=tl, unit=unit, sets=sets),
            ignore_index=True,
        )

        # return if statistics only should be obtained
        if self.desc_only:
            return

        # create train and test jsons for storing in prepared layer
        for i, t_set in enumerate(["train", "test"]):

            # make json for prepared layer
            jsc_prep = JSConverter()
            jsc_prep.map_data_obj(
                c_id=jsc.data_obj["c_id"],
                c_name=jsc.data_obj["c_name"],
                c_description=jsc.data_obj["c_description"],
                c_domain=jsc.data_obj["c_domain"],
                c_link_id=jsc.data_obj["c_link_id"],
                c_value=list(sets["{}_df".format(t_set)]["c_value"]),
                c_case_id=list(sets["{}_df".format(t_set)]["c_case_id"]),
                c_pat_id=list(sets["{}_df".format(t_set)]["c_pat_id"]),
                c_end_ts=list(sets["{}_df".format(t_set)]["c_end_ts"]),
                c_start_ts=list(sets["{}_df".format(t_set)]["c_start_ts"]),
                c_unit=[
                    u.strip().lower()
                    for u in list(sets["{}_df".format(t_set)]["c_unit"])
                ],
            )

            # append additional information
            jsc_prep.add_prep_keys(
                c_time_line=tl,
                c_set=t_set,
                c_missing_pos=sets["miss_{}_pos".format(t_set)],
                c_missing_neg=sets["miss_{}_pos".format(t_set)],
                c_target=list(sets["{}_df".format(t_set)]["c_target"]),
                c_op_id=list(sets["{}_df".format(t_set)]["c_op_id"]),
            )

            # build unit string for storing
            unit_string = unit.replace(" ", "").replace("/", "_per_")
            if unit_string.strip() != "":
                unit_string = "_{}".format(unit_string.strip())

            # create storage string
            store_string = "./data/prepared/{domain}/tl{tl}/{name}_{set}{unit}".format(
                domain=jsc.data_obj["c_domain"].strip().replace(" ", "_").lower(),
                tl=str(tl),
                name=jsc.data_obj["c_name"].strip().replace(" ", "_").lower(),
                set=t_set,
                unit=unit_string,
            )
            print(store_string)
            jsc_prep.store_js_file(store_string)

    def extract_prepared_features(self):

        store_datetime = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

        for i, file in enumerate(MetaManager.ehr_raw_files):

            if "nudesc_backup" in file:
                continue

            if not "nudesc" in file:
                continue

            if "camicu" in file:
                continue

            # open and preprocess information
            jsc = JSConverter()
            jsc.read_js_file(file.split(".json")[0])

            # preprocess ehr
            jsc, df = self.preprocess_ehr_feature(jsc)

            if (df is None) or (df.empty):
                print("raw data not available")
                continue

            units = list(df.c_unit.drop_duplicates())

            # iterate through units
            for unit in units:
                df = df[df.c_unit == unit]
                if jsc.is_time_relevant():
                    for tl in range(1, 4):
                        self.get_time_set_prepared_features(
                            tl=tl, jsc=jsc, df=df, unit=unit
                        )
                else:
                    self.get_time_set_prepared_features(jsc=jsc, df=df, unit=unit)

            jsc = None
            df = None
            gc.collect()

            self.missing_df.to_csv(
                "./data/metrics/descriptive/missing_features_{}.csv".format(
                    store_datetime
                )
            )
            self.available_df.to_csv(
                "./data/metrics/descriptive/available_features_{}.csv".format(
                    store_datetime
                )
            )

    def normalize_odds_ratio(self):

        path = "./data/metrics/descriptive/{}.csv".format(
            MetaManager.get_last_available_features()
        )
        self.available_df = pd.read_csv(path, index_col=0)
        self.available_df["odds_ratio_train"] = (
            self.available_df["odds_ratio_train"] / 2
        )
        self.available_df["odds_ratio_test"] = self.available_df["odds_ratio_test"] / 2
        d = self.available_df
        self.available_df = self.min_max_scaling(
            df=self.available_df, colname="odds_ratio_train"
        )
        self.available_df = self.min_max_scaling(
            df=self.available_df, colname="odds_ratio_test"
        )

        self.available_df["odds_ratio_train_normalized"] = (
            1 - self.available_df["odds_ratio_train"]
        )
        self.available_df["odds_ratio_test_normaized"] = (
            1 - self.available_df["odds_ratio_test"]
        )
        self.available_df.to_csv(path)
