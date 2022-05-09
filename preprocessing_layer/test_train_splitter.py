from preprocessing_layer.time_manager import TimeManager
from extraction_layer.support_classes.js_converter import JSConverter
from pandas.core.frame import DataFrame


class TrainTestSplitter:
    def __init__(self) -> None:
        self.tm = TimeManager()
        self.split_ratio = 0.2
        self.master_df = self.tm.get_times()

    def split_train_test_patients(self):

        mdf = self.master_df

        # split in pos and neg
        pos = mdf[mdf.c_target == 1]["c_pat_id"].drop_duplicates()
        neg = mdf[~mdf.c_pat_id.isin(list(pos))]["c_pat_id"].drop_duplicates()

        # 30 % random cases with at least one pos target variable
        test_pos = pos.sample(frac=self.split_ratio)
        train_pos = pos[~pos.isin(list(test_pos))]
        # 30 % random cases with all 0 as target variable
        test_neg = neg.sample(frac=self.split_ratio)
        train_neg = neg[~neg.isin(list(test_neg))]

        # print sizes
        print("# of pos test patients: ", len(test_pos))
        print("# of neg test patients: ", len(test_neg))
        print("# of pos train patients: ", len(train_pos))
        print("# of neg train patients: ", len(train_neg))

        # store splitted data
        jsc = JSConverter(
            {
                "test_pos": list(test_pos),
                "test_neg": list(test_neg),
                "train_pos": list(train_pos),
                "train_neg": list(train_neg),
            }
        )
        jsc.store_js_file("./data/meta/cohort/train_test_split")

    # calc odds as cat missingness
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
                        split.data_obj["train_neg"] + split.data_obj["train_pos"]
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
