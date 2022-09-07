from re import split

from matplotlib.pyplot import axis
from preprocessing_layer.data_manager import DataManager
import numpy as np
import pandas as pd
import gc
from functools import reduce
from pandas.core.frame import DataFrame
from pandas.core.groupby.generic import DataFrameGroupBy
from scipy.stats.mstats import mquantiles
from extraction_layer.support_classes.js_converter import JSConverter
from extraction_layer.support_classes.meta_manager import MetaManager


class FeatureSelector:
    def __init__(self) -> None:
        self.percentiles = [0.1, 0.5, 0.9]
        self.df_set_list_cat = []
        self.df_set_list_num = []
        self.no_feats = [
            "c_pat_id",
            "c_case_id",
            "c_an_start_ts",
            "c_op_id",
            "c_target",
            "c_time_consistent",
        ]

    def define_feature_sets(self):
        '''
        feature sets are first defined purly on condition that at leat 10% of data is non missing
                Parameters: 
                        None
                Returns:
                        None
        '''
        # reading available features
        df = pd.read_csv(
            "./data/metrics/descriptive/{}.csv".format(
                "available_features_16_08_2022_revision"
            ),
            index_col=0,
        )
        df = df.assign(miss_train_mean=(
            df.miss_train_pos + df.miss_train_neg) / 2)
        if "index" in list(df.columns):
            df = df.drop(columns=["index"])
        # if occure drop duplicated rows
        df = df.drop_duplicates()
        # create feature with all features at TL0
        df[df.tl == 0].to_csv(
            "./data/meta/feature_sets/tl0/feature_set_tl0.csv")
        # application to numeric as well as categorical features
        feat_types = ["categorical", "numeric"]
        i = 3
        for feat_t in feat_types:
            for tl in range(1, 4):
                df_feat = df[((df.tl == tl) | (df.tl == 0))
                             & (df.numeric == (feat_t == "numeric"))]
                # iteration through quantiles and time lines
                if (feat_t == "numeric"):
                    df_q = df_feat[
                        (df_feat["miss_train"] <= 0.9)
                    ]
                else:
                    df_q = df_feat
                df_q = df_q.assign(size=len(df_q))
                df_q = df_q.assign(i=i)
                df_q.reset_index().to_csv(
                    "./data/meta/feature_sets/tl{}/{}_feature_set_i_{}_revised.csv".format(
                        str(tl), feat_t, str(i)
                    )
                )
                if (feat_t == "numeric"):
                    self.df_set_list_num.append(df_q)
                if not (feat_t == "numeric"):
                    self.df_set_list_cat.append(df_q)

    def __get_store_strings(self, r=None, tl: int = 0):
        '''
        helper method to open file for feature
                Parameters: 
                        r (Series) row of data for features, tl (int): time window 
                Returns:
                        path_string_test (str): path of feature in test set, 
                        path_string_train (str): path of feature in train set
        '''
        feature_string = r["feature"].strip().lower().replace(" ", "_")
        domain_string = r["domain"].strip().lower().replace(" ", "_")
        unit_string = r["unit"].replace(" ", "").replace("/", "_per_")
        if unit_string.strip() != "":
            unit_string = "_{}".format(unit_string)

        # exception for tl0
        if r["tl"] == 0.0:
            tl = 0

        # build file and path strings and concatenate
        file_string_test = "{}_test{}".format(
            str(feature_string), str(unit_string))
        file_string_train = "{}_train{}".format(
            str(feature_string), str(unit_string))
        path_string_test = "./data/prepared/{}/tl{}/{}".format(
            domain_string, str(tl), file_string_test
        )
        path_string_train = "./data/prepared/{}/tl{}/{}".format(
            domain_string, str(tl), file_string_train
        )
        return path_string_test, path_string_train

    def __merge_with_df_X(
        self,
        df_X: DataFrame = None,
        df: DataFrame = None,
        feature_name: str = "",
        cols: list = ["c_pat_id", "c_case_id", "c_op_id"],
    ):
        '''
        method merging feature information to master table 
                Parameters: 
                        df_X (df): data for holding features as columns, df (data): actual feature data, 
                        feature_name (str): name of feature, cols (list): list of merge columns
                Returns:
                        data (df): merged data 
        '''
        # merging condition for feature in X dataset
        return df_X.merge(
            df.rename(columns={"c_value": feature_name}), on=cols, how="left"
        ).drop_duplicates()

    def __build_categoric_sets(
        self,
        train_store_path: str = "",
        test_store_path: str = "",
        df_cat: DataFrame = None,
        origin_len_df_train: int = 0,
        tl: int = 0,
        df_train: DataFrame = None,
        df_test: DataFrame = None,
    ):
        '''
        building and storing categoric feature sets
                Parameters: 
                        train_store_path (str), test_store_path (str), df_cat (data): categoric data, 
                        origin_len_df_train (int): length of result for basic validation checkup, 
                        tl (int): time window index, df_train (df), df_test (df)
                Returns:
                        None
        '''
        # iterate through categorical features and appen them to list
        for i, row in enumerate(df_cat.drop_duplicates().iterrows()):
            r = row[1].fillna("")
            path_string_test, path_string_train = self.__get_store_strings(
                r=r, tl=tl)
            for set_t, path_string in zip(
                ["test", "train"], [path_string_test, path_string_train]
            ):
                # check if feature is available at this timeline
                if path_string_train + ".json" not in MetaManager.ehr_prepared_files:
                    print("feature not available for this time line")
                    continue

                jsc = JSConverter()
                _ = jsc.read_js_file(path_string)
                print("Read: ", path_string)
                name = jsc.data_obj["c_name"].strip().lower().replace(" ", "_")
                df = jsc.to_df(
                    col=["c_case_id", "c_pat_id", "c_op_id", "c_value"])
                df = (
                    df.groupby(["c_case_id", "c_pat_id", "c_op_id"])["c_value"]
                    .max()
                    .reset_index()
                )
                if set_t == "train":
                    df_train = self.__merge_with_df_X(
                        df_X=df_train, feature_name=name, df=df, jsc=jsc
                    )
                if set_t == "test":
                    df_test = self.__merge_with_df_X(
                        df_X=df_test, feature_name=name, df=df, jsc=jsc
                    )

                # throw exception if df_train is getting bigger
                if origin_len_df_train < len(df_train):
                    print(len(df_train))
                    print("Error: train set is getting bigger")
                    return

        # Print feature information
        print("Features :", df_train.columns)
        print("# of Features: ", len(df_train.columns))

        # store build dfs
        df_train.fillna(0).to_csv(train_store_path)
        df_test.fillna(0).to_csv(test_store_path)

        # try to free memory
        gc.collect()

    def __build_numeric_sets(
        self,
        train_store_path: str = "",
        test_store_path: str = "",
        df_num: DataFrame = None,
        origin_len_df_train: int = 0,
        tl: int = 0,
        df_train: DataFrame = None,
        df_test: DataFrame = None,
    ):
        '''
        building and storing nuemric feature sets
                Parameters: 
                        train_store_path (str), test_store_path (str), df_cat (data): categoric data, 
                        origin_len_df_train (int): length of result for basic validation checkup, 
                        tl (int): time window index, df_train (df), df_test (df)
                Returns:
                        None
        '''
        # iterate through numeric features and append them to list
        for i, row in enumerate(df_num.drop_duplicates().iterrows()):
            r = row[1].fillna("")
            path_string_test, path_string_train = self.__get_store_strings(
                r=r, tl=tl)
            for set_t, path_string in zip(
                ["test", "train"], [path_string_test, path_string_train]
            ):
                # check if feature is available at this timeline
                if path_string_train + ".json" not in MetaManager.ehr_prepared_files:
                    print("feature not available for this time line")
                    continue
                # opening jsons and preprocessing with percentiles
                jsc = JSConverter()
                _ = jsc.read_js_file(path_string)
                print("Read: ", path_string)

                if jsc.is_for_percentile():
                    # iterate through percentiles and append to df X
                    for perc in self.percentiles:
                        df_perc = jsc.get_percentile(perc)
                        name = "{}_p{}".format(
                            jsc.data_obj["c_name"].strip(
                            ).lower().replace(" ", "_"),
                            str(perc),
                        )
                        if set_t == "train":  # append to train df
                            df_train = self.__merge_with_df_X(
                                df_X=df_train, feature_name=name, df=df_perc, jsc=jsc
                            )
                        if set_t == "test":  # append to test df
                            df_test = self.__merge_with_df_X(
                                df_X=df_test, feature_name=name, df=df_perc, jsc=jsc
                            )
                else:
                    # other features like comorbidities
                    name = jsc.data_obj["c_name"].lower().replace(" ", "_")
                    df = jsc.to_df(
                        col=["c_pat_id", "c_case_id", "c_op_id", "c_value"]
                    ).drop_duplicates()
                    if set_t == "test":
                        df_test = self.__merge_with_df_X(
                            df_X=df_test, feature_name=name, df=df, jsc=jsc
                        )
                    if set_t == "train":
                        df_train = self.__merge_with_df_X(
                            df_X=df_train, feature_name=name, df=df, jsc=jsc
                        )

                # add sum of features if feature contains volumes or amounts
                if jsc.is_for_sum():
                    df_sum = jsc.get_sum()
                    name = "{}_sum".format(
                        jsc.data_obj["c_name"].lower().replace(" ", "_")
                    )
                    if set_t == "train":
                        df_train = self.__merge_with_df_X(
                            df_X=df_train, feature_name=name, df=df_sum, jsc=jsc
                        )
                    if set_t == "test":
                        df_test = self.__merge_with_df_X(
                            df_X=df_test, feature_name=name, df=df_sum, jsc=jsc
                        )

                # calculate stadard deviation from the median for high frequency variables
                if jsc.is_for_std():
                    df_std = jsc.get_std()
                    name = "{}_std".format(
                        jsc.data_obj["c_name"].lower().replace(" ", "_")
                    )
                    if set_t == "train":
                        df_train = self.__merge_with_df_X(
                            df_X=df_train, feature_name=name, df=df_std, jsc=jsc
                        )
                    if set_t == "test":
                        df_test = self.__merge_with_df_X(
                            df_X=df_test, feature_name=name, df=df_std, jsc=jsc
                        )

                # print meta information
                print("Length train df with na total",
                      len(df_train.drop_duplicates()))
                print("Length train df without na total", len(df_train.dropna()))
                print(
                    "Length train df without na y=1",
                    len(df_train[df_train.c_target == 1].dropna()),
                )
                print(
                    "Length train df without na y=0",
                    len(df_train[df_train.c_target == 0].dropna()),
                )

                print("length df train: ", len(df_train))

                # throw exception if df_train is getting bigger
                if (origin_len_df_train + 2) < len(df_train):
                    print("Error: train set is getting much more bigger")
                    return

        # Print feature information
        print("Features :", df_train.columns)
        print("# of Features: ", len(df_train.columns))

        # store build dfs
        df_train.to_csv(train_store_path)
        df_test.to_csv(test_store_path)

        # try to free memory
        gc.collect()

    def build_feature_sets(self):
        '''
        method for building feature sets per time windows
                Parameters: None
                Returns: None
        '''
        master_df = pd.read_csv(
            "./data/meta/cohort/master_time_table_22_08_22.csv", index_col=0)
        sets_t = JSConverter()
        _ = sets_t.read_js_file("./data/meta/cohort/train_test_split")
        # define one X set per q and tl as well as test and train
        df_X = master_df[
            [
                "c_case_id",
                "c_pat_id",
                "c_time_consistent",
                "c_op_id",
                "c_target",
                "c_an_start_ts",
            ]
        ].drop_duplicates()
        df_train = df_X[
            df_X.c_pat_id.isin(
                sets_t.data_obj["train_pos"] + sets_t.data_obj["train_neg"]
            )
        ]
        df_test = df_X[
            df_X.c_pat_id.isin(
                sets_t.data_obj["test_pos"] + sets_t.data_obj["test_neg"]
            )
        ]
        origin_len_df_train = len(df_train)

        # build clinical and tl=0 feature sets
        df_clinic = pd.read_csv(
            "./data/meta/feature_sets/clinical/feature_set_clinical.csv", index_col=0
        )
        df_tl0 = pd.read_csv(
            "./data/meta/feature_sets/tl0/feature_set_tl0.csv", index_col=0
        )

        for tl in list(range(0, 4)):
            for path, df in zip(
                ["clinical/tl{}".format(tl), "tl0"], [df_clinic, df_tl0]
            ):
                # building cat and num feature sets for tl0 and clinical defined features
                self.__build_categoric_sets(
                    train_store_path="./data/interpreted/{}/df_train_categoric_revised.csv".format(
                        path
                    ),
                    test_store_path="./data/interpreted/{}/df_test_categoric_revised.csv".format(
                        path
                    ),
                    tl=tl,
                    df_cat=df[(df.numeric == False) & (df.tl == tl)],
                    df_test=df_test,
                    df_train=df_train,
                    origin_len_df_train=len(df_train),
                )
                self.__build_numeric_sets(
                    train_store_path="./data/interpreted/{}/df_train_numeric_revised.csv".format(
                        path
                    ),
                    test_store_path="./data/interpreted/{}/df_test_numeric_revised.csv".format(
                        path
                    ),
                    tl=tl,
                    df_num=df[(df.numeric == True) & (df.tl == tl)],
                    df_test=df_test,
                    df_train=df_train,
                    origin_len_df_train=len(df_train),
                )

        #  iterate through i defined feature sets and build them
        for tl in list(range(1, 4)):
            # iterating through quantiles
            i = 3
            # opening feature set information
            path_num = "./data/meta/feature_sets/tl{}/numeric_feature_set_i_{}_revised.csv".format(
                str(tl), str(i)
            )
            path_cat = "./data/meta/feature_sets/tl{}/categorical_feature_set_i_{}_revised.csv".format(
                str(tl), str(i)
            )
            df_num = pd.read_csv(path_num, index_col=0)
            df_cat = pd.read_csv(path_cat, index_col=0)
            print("TL: ", tl, "i: ", i)
            print("Length orig. train set (num/cat): ", len(df_train))
            print("Length orig. test set (num/cat): ", len(df_test))
            self.__build_numeric_sets(
                train_store_path="./data/interpreted/tl{}/i{}/df_train_numeric_revised.csv".format(
                    str(tl), str(i)
                ),
                test_store_path="./data/interpreted/tl{}/i{}/df_test_numeric_revised.csv".format(
                    str(tl), str(i)
                ),
                tl=tl,
                df_num=df_num,
                origin_len_df_train=origin_len_df_train,
                df_train=df_train,
                df_test=df_test,
            )
            self.__build_categoric_sets(
                train_store_path="./data/interpreted/tl{}/i{}/df_train_categoric_revised.csv".format(
                    str(tl), str(i)
                ),
                test_store_path="./data/interpreted/tl{}/i{}/df_test_categoric_revised.csv".format(
                    str(tl), str(i)
                ),
                tl=tl,
                df_cat=df_cat,
                df_test=df_test,
                df_train=df_train,
                origin_len_df_train=origin_len_df_train,
            )

    # define a feature selection algorithm based on missingness and univariate performance metrics
    def apply_feature_selection_algorithm(
        self,
        tl_list: list = [],
        missing_fraction=0.1,
        numeric_threshold=0.05,
        categoric_threshold=2,
        availability_threshold=0.1,
    ):
        '''
        application of feature selection algorithm as described in the methods part
                Parameters: 
                        tl_list (list): list of time windows that need to be incorporated, 
                        missing_fraction (float): tunable treshold for missingness per row (surgery), 
                        numeric_threshold (float): tunable threshold for numeric effect size,
                        categoric_threshold (float): tunable threshold for categoric effect size,
                        availability_threshold (float): threshold for general availability of data (patient)
                Returns: None
        '''
        # join all dfs for multiple timeline
        df_num, df_cat = self.get_tl_datasets(tl_list=tl_list)
        univ_num = pd.read_csv(
            "./data/metrics/statistical/univariate_stats_train_numeric_revised.csv", index_col=0
        )
        univ_cat = pd.read_csv(
            "./data/metrics/statistical/univariate_stats_train_categoric_revised.csv",
            index_col=0,
        )
        # reset indexes
        univ_cat = univ_cat.reset_index()
        univ_num = univ_num.reset_index()
        # aggregate univariate numeric statistics according to the selected timelines
        univ_num = univ_num[
            (univ_num.reject == True)
            & (univ_num.tl >= min(tl_list))
            & (univ_num.tl <= max(tl_list))
        ]
        univ_num = univ_num[univ_num["avai"] > availability_threshold]
        univ_num = (
            univ_num.groupby(["feature"])["2|AUC-0.5|"].mean().reset_index()
        )
        feats_num = univ_num[univ_num["2|AUC-0.5|"] > numeric_threshold].sort_values(
            ["2|AUC-0.5|"], ascending=False
        )["feature"]
        # select catgoric feature sets according to the selcted timelines
        feats_cat = univ_cat[
            (univ_cat["|log(OR)|"] > categoric_threshold)
            & (univ_cat.tl >= min(tl_list))
            & (univ_cat.tl <= max(tl_list))
        ]["feature"]
        # select all avaiable columns for any timeline
        final_num_feat_list, final_cat_feat_list, feat_num_list_raw = [], [], []
        for i, feat_num in enumerate(feats_num):
            # search for time variabts of selected features
            f = [x for x in list(df_num.columns) if feat_num in x]
            df_tmp = df_num[f + final_num_feat_list]
            df_tmp = df_tmp.assign(nan_count=df_tmp.isnull().sum(axis=1))
            df_tmp = df_tmp.assign(
                nan_perc=df_tmp.nan_count / (len(df_tmp.columns) - 1)
            )
            row_miss = len(df_tmp[df_tmp.nan_perc > 0.5]) / len(df_tmp)
            if row_miss < missing_fraction:
                # store the final time variate features
                final_num_feat_list = final_num_feat_list + f
                # store the raw features
                feat_num_list_raw.append(feat_num)
        for i, feat_cat in enumerate(feats_cat):
            hist = "history" in feat_cat
            f = [
                x
                for x in list(df_cat.columns)
                if (feat_cat in x) and not (bool("history" in x) != bool(hist))
            ]
            final_cat_feat_list = final_cat_feat_list + f
        final_cat_feat_list = list(dict.fromkeys(final_cat_feat_list))
        final_num_feat_list = self.__reduce_feature_list(final_num_feat_list)
        # print length of feature
        print("Number numerical features: ", len(final_num_feat_list))
        print("Number categorical features: ", len(final_cat_feat_list))
        # remove meaningless columns with constant values one or zero
        df_num = self.__remove_meaningless_cols(
            df_num[final_num_feat_list + self.no_feats]
        )
        df_cat = self.__remove_meaningless_cols(
            df_cat[final_cat_feat_list + self.no_feats]
        )
        final_num_feat_list = [
            x for x in list(df_num.columns) if x not in self.no_feats
        ]
        # select binary categoric variables with OR metric greater than 2
        return df_num, df_cat, final_num_feat_list, feat_num_list_raw

    # open numerical and categorical datasets according to timelines
    def get_tl_datasets(self, tl_list: list = [], ds_type: str = "train"):
        '''
        method for getting data from the time windows
                Parameters: 
                        tl_list (list): list of time window indices, ds_type (str): type of dataset (train or test)
                Returns:
                        df_numeric (df): numeric data per time windows,
                        df_categoric (df): categoric data per time windows
        '''
        # join all dfs for multiple timeline
        df_list_numeric, df_list_categoric = [], []
        for tl in tl_list:
            # open numeric and categoric data frames
            df_tl_numeric = pd.read_csv(
                "./data/interpreted/tl{}/i3/df_{}_numeric.csv".format(
                    str(tl), ds_type),
                index_col=0,
            )
            df_tl_categoric = pd.read_csv(
                "./data/interpreted/tl{}/i3/df_{}_categoric.csv".format(
                    str(tl), ds_type
                ),
                index_col=0,
            )
            # rename columns related to the time lines
            df_tl_numeric = self.__assign_tl_colums(df=df_tl_numeric, tl=tl)
            df_tl_categoric = self.__assign_tl_colums(
                df=df_tl_categoric, tl=tl)
            # append them to df lists
            df_list_numeric.append(df_tl_numeric)
            df_list_categoric.append(df_tl_categoric)
        # join dfs from list on keys
        df_numeric = reduce(
            lambda df1, df2: pd.merge(
                df1, df2, on=self.no_feats, suffixes=["", "_duplicate"]
            ),
            df_list_numeric,
        )
        df_categoric = reduce(
            lambda df1, df2: pd.merge(
                df1, df2, on=self.no_feats, suffixes=["", "_duplicate"]
            ),
            df_list_categoric,
        )
        # drop all redundant features
        df_numeric = df_numeric[
            [x for x in list(df_numeric.columns) if not "_duplicate" in x]
        ]
        df_categoric = df_categoric[
            [x for x in list(df_categoric.columns) if not "_duplicate" in x]
        ]
        return df_numeric, df_categoric

    def __reduce_feature_list(self, feat_list: list = []):
        '''
        reducing the feature sets per feature to the most discriminative one
        '''
        list_reduced_feat = []
        for f in feat_list:
            if not f[:-1] in [x[:-1] for x in list_reduced_feat]:
                list_reduced_feat.append(f)
        return list_reduced_feat

    def __remove_meaningless_cols(self, df):
        '''
        remove columns having just ones or zeros and drop times that do not make sense
        '''
        drop_cols = []
        time_inc_cols = [
            "tl1_surgery_time",
            "tl3_surgery_time",
            "tl1_aneasthesia_time",
            "tl3_aneasthesia_time",
            "tl2_recovery_room_time",
        ]
        for col in list(df.columns):
            if all(v == 0 for v in list(df[col])):
                drop_cols.append(col)
            if all(v == 1 for v in list(df[col])):
                drop_cols.append(col)
            if col in time_inc_cols:
                drop_cols.append(col)
        return df.drop(columns=drop_cols, axis=1)

    def __assign_tl_colums(self, df: DataFrame = None, tl: int = 0):
        '''
       rename columns for time relevance in models
        '''
        tl0_feats = pd.read_csv(
            "./data/meta/feature_sets/tl0/feature_set_tl0.csv", index_col=0
        )["feature"]
        tl0_feats = [x.replace(" ", "_") for x in tl0_feats]
        df.columns = [
            "tl{}_{}".format(tl, x)
            if (x not in self.no_feats) and (x not in tl0_feats)
            else "tl{}_{}".format(0, x)
            if (x not in self.no_feats) and (x in tl0_feats)
            else x
            for x in list(df.columns)
        ]
        return df
