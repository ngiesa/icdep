from matplotlib.pyplot import axis
from numpy import isnan
import pandas as pd
from pandas.core.frame import DataFrame
from extraction_layer.support_classes.meta_manager import MetaManager
from extraction_layer.support_classes.js_converter import JSConverter
from preprocessing_layer.annotator import Annotator


class TimeManager:
    def __init__(self) -> None:
        self.ann = Annotator()
        self.tl_count = 3

    def create_time_df(self, js: dict = {}, abbr: str = ""):
        # parsing jsons for times into dfs
        return pd.DataFrame(
            {
                "c_case_id": js["c_case_id"],
                "c_pat_id": js["c_pat_id"],
                "c_link_id": js["c_link_id"],
                "c_" + abbr + "_start_ts": pd.to_datetime(js["c_start_ts"]),
                "c_" + abbr + "_end_ts": pd.to_datetime(js["c_end_ts"]),
            }
        )

    def get_times(self):

        # check if master table exists and load
        if "master_time_table" in MetaManager().files_written:
            return pd.read_csv("./data/meta/cohort/master_time_table.csv", index_col=0)

        jsc = JSConverter()

        # loading time information in df structure
        df_rec = self.create_time_df(
            js=jsc.read_js_file(path="./data/raw/hospitalization/recovery_room_time"),
            abbr="rec",
        )
        df_hos = self.create_time_df(
            js=jsc.read_js_file(path="./data/raw/hospitalization/hospitalization_time"),
            abbr="hos",
        )
        df_op = self.create_time_df(
            js=jsc.read_js_file(path="./data/raw/hospitalization/surgery_time"),
            abbr="op",
        )
        df_an = self.create_time_df(
            js=jsc.read_js_file(path="./data/raw/hospitalization/aneasthesia_time"),
            abbr="an",
        )

        # load target information
        target = pd.DataFrame(
            jsc.read_js_file(path="./data/raw/scores_and_scales/nudesc")
        )
        target = target.assign(c_timestamp=pd.to_datetime(target.c_timestamp))

        # join op related times
        df_rel = df_rec.merge(df_op, on=["c_case_id", "c_link_id", "c_pat_id"]).merge(
            df_an, on=["c_case_id", "c_link_id", "c_pat_id"]
        )

        # join times and target
        df = df_rel.merge(
            df_hos.drop(columns=["c_pat_id"], axis=1).drop(columns="c_link_id", axis=1),
            on=["c_case_id"],
            how="inner",
        ).merge(target.drop(columns=["c_pat_id"], axis=1), on=["c_case_id"], how="left")

        # do logical imputation for important time stamps
        df = self.impute_time_stamps(df)

        # add extra time window of 24h for admission
        df = df.assign(c_hos_start_ts=pd.to_datetime(df.c_hos_start_ts))
        df = df.assign(
            c_hos_start_ts=df["c_hos_start_ts"] - pd.to_timedelta(1, unit="d")
        )

        # store prev op counts
        df = self.get_prev_surgeries(df)

        # assign last ts for calculating time frames
        print("assigning last ts to time relevant data")
        df["c_last_ts"] = df.apply(lambda x: self.assign_last_ts(x, df), axis=1)

        # focus on delirium in recovery room only
        df = df[
            (df.c_timestamp > df.c_rec_start_ts) & (df.c_timestamp < df.c_rec_end_ts)
        ]

        # add binary annotation variable based on theshold
        df = self.ann.annotate_binary(df)
        df = df.drop_duplicates()

        # renaming the link id in op id
        df = df.rename(columns={"c_link_id": "c_op_id"})

        # store master table as readable csv
        df.to_csv("./data/meta/cohort/master_time_table.csv")

        return df

    def replace_time_order(
        self,
        df: DataFrame = None,
        col_before: str = "",
        col_after: str = "",
        back_replacement: bool = True,
    ):
        # cutting non time consistent data and replacing
        df_tmp = df[df[col_before] > df[col_after]]
        if back_replacement:
            add = {col_before: df_tmp[col_after]}
        else:
            add = {col_after: df_tmp[col_before]}
        df_tmp = df_tmp.assign(**add)
        df = df[df[col_before] <= df[col_after]]
        df = df.append(df_tmp)
        return df

    def impute_time_stamps(self, df: DataFrame = None):

        # impute if op end is null with aneast end
        df.c_op_end_ts.fillna(df.c_an_end_ts, inplace=True)

        # impute if op start is null with aneast start
        df.c_op_start_ts.fillna(df.c_an_start_ts, inplace=True)

        # impute aneast start is null with op start
        df.c_an_start_ts.fillna(df.c_op_start_ts, inplace=True)

        # impute aneast end is null with op end
        df.c_an_end_ts.fillna(df.c_hos_end_ts, inplace=True)

        # impute rec start with an end if null
        df.c_rec_start_ts.fillna(df.c_an_end_ts, inplace=True)

        # impute rec end with hos end if null
        df.c_rec_end_ts.fillna(df.c_hos_end_ts, inplace=True)

        # impute hospitalizaiton start with an start
        df.c_hos_start_ts.fillna(df.c_an_start_ts)

        # do basic admission and discharge replacement
        df = self.replace_time_order(
            df=df, col_before="c_hos_start_ts", col_after="c_an_start_ts"
        )
        df = self.replace_time_order(
            df=df,
            col_before="c_rec_end_ts",
            col_after="c_hos_end_ts",
            back_replacement=False,
        )
        df = df.reset_index()
        df = df.assign(id=[i for i in range(0, len(df))])

        # assign durations in hours or days (hosp)
        for col in ["c_hos", "c_an", "c_rec"]:
            col_start = "{}_start_ts".format(col)
            col_end = "{}_end_ts".format(col)
            dur_col = "{}_duration".format(col)
            df[dur_col] = pd.to_datetime(df[col_end]) - pd.to_datetime(df[col_start])
            df[dur_col] = [x.total_seconds() / 3600 for x in df[dur_col]]
            if col == "c_hos":
                df[dur_col] = [x / 24.0 for x in df[dur_col]]

        # lable data which is time consistent
        df_tmp = df[
            (df.c_an_start_ts < df.c_op_start_ts)
            & (df.c_op_start_ts < df.c_op_end_ts)
            & (df.c_op_end_ts < df.c_an_end_ts)
            & (df.c_an_end_ts < df.c_rec_start_ts)
            & (df.c_rec_start_ts < df.c_rec_end_ts)
            & (df.c_hos_duration < 1000)
            & (df.c_an_duration < 1000)
            & (df.c_rec_duration < 1000)
        ]

        # assign time consistency flag
        df_tmp = df_tmp.assign(c_time_consistent=True)
        df = df[~df.id.isin(df_tmp.id)]
        df = df.assign(c_time_consistent=False)
        df = df.append(df_tmp, ignore_index=True)

        print("total number of rows in time dataframe: ", len(df))

        return df.drop(columns=["id"], axis=1)

    def assign_last_ts(self, row, df):
        # treat last rec room as new stay within one hospitalization
        if row["c_op_count"] == 0:
            # return hos when op count is one per case
            return row["c_hos_start_ts"]
        else:
            # return last rec room end when op count > 0
            df = df[(df.c_case_id == row["c_case_id"])]
            df = df[(df.c_op_count == row["c_op_count"] - 1)]
            return df["c_rec_end_ts"].iloc[0]

    def get_prev_surgeries(self, df):

        op_count = {
            "number of surgeries": "c_op_count",
            "number of previous surgeries": "c_prev_op_count",
        }

        # select relevant columns for counting
        df_surg = df[
            ["c_case_id", "c_link_id", "c_op_start_ts", "c_pat_id"]
        ].drop_duplicates()

        # assign the number of ops within one stay
        df_surg = df_surg.sort_values("c_op_start_ts", ascending=True)
        df_surg = df_surg.assign(
            c_op_count=df_surg.groupby("c_case_id")["c_op_start_ts"].cumcount()
        )

        # sum up the count from all previous stays
        df_surg = df_surg.sort_values("c_op_start_ts", ascending=True)
        df_surg = df_surg.assign(c_prev_op_count=df_surg.groupby("c_pat_id").cumcount())

        # store op count as feature in json
        meta_vars = MetaManager.get_meta_data()
        meta_prev_ops = meta_vars[meta_vars["var_name"].isin(list(op_count.keys()))]

        # iterate through meta vars
        for i, meta_var in meta_prev_ops.iterrows():

            col_name = op_count[meta_var["var_name"]]

            if col_name == "c_op_count":
                c_value = [x + 1 for x in list(df_surg[col_name])]
            else:
                c_value = list(df_surg[col_name])

            # enhance data js object by additional key value pairs
            js_conv = JSConverter()
            js_conv.map_data_obj(
                c_id=meta_var["ID"],
                c_pat_id=list(df_surg.c_pat_id),
                c_case_id=list(df_surg.c_case_id),
                c_link_id=list(df_surg.c_link_id),
                c_start_ts=list(df_surg.c_op_start_ts),
                c_name=meta_var["var_name"],
                c_value=c_value,
                c_domain=meta_var["Domain"],
            )

            js_conv.store_js_file(
                path=js_conv.store_path.format(
                    "{}/{}".format(
                        meta_var["Domain"].replace(" ", "_").lower(),
                        meta_var["var_name"].replace(" ", "_"),
                    )
                )
            )

        df = df.merge(
            df_surg.drop(columns=["c_op_start_ts"], axis=1),
            on=["c_link_id", "c_pat_id", "c_case_id"],
        )
        return df

    def calc_duration_for_tl(self, df: DataFrame = None, end_ts_col: str = ""):
        df = df.assign(c_start_ts=pd.to_datetime(df.c_start_ts))
        df[end_ts_col] = pd.to_datetime(df[end_ts_col])
        df = df.assign(c_value=(df[end_ts_col] - df.c_start_ts))
        df = df.assign(c_value=[x.total_seconds() / 3600.00 for x in df.c_value])
        return df

    def compare_end_ts(self, df: DataFrame = None, end_col: str = ""):
        # if the end ts of the feature is before end ts of time col, then use end ts
        df_1 = df[df.c_end_ts < df[end_col]]
        df_2 = df[df.c_end_ts >= df[end_col]]
        if not df_1.empty:
            df_1 = self.calc_duration_for_tl(df=df_1, end_ts_col="c_end_ts")
        if not df_2.empty:
            df_2 = self.calc_duration_for_tl(df=df_2, end_ts_col=end_col)
        return df_1.append(df_2)

    def get_time_lines(
        self, df: DataFrame = None, tl: int = 1, is_duration=False, name: str = ""
    ):

        df = df.drop_duplicates()

        required_columns = [
            "c_an_start_ts",
            "c_start_ts",
            "c_rec_start_ts",
            "c_an_end_ts",
            "c_timestamp",
            "c_last_ts",
        ]

        if (len(df[(df["c_start_ts"] == "")]) > 0.1 * len(df)) or (
            len(df[df.c_start_ts.isnull()]) > 0.1 * len(df)
        ):
            print("feature does have empty time stamps over 10%")
            return

        if not set(required_columns).issubset(set(list(df.columns))):
            print("df does not contain required time columns")
            return

        # make required cols datetimes
        for col in required_columns:
            add = {col: pd.to_datetime(df[col])}
            df = df.assign(**add)

        if tl == 1:
            #  data before aneasthesia for an operation
            df = df[(df["c_start_ts"] < df["c_an_start_ts"])]
            if not is_duration:
                df = df[(df["c_start_ts"] >= df["c_last_ts"])]
            if name.replace(" ", "") in ["surgery_time", "anesthesia_time"]:
                print("inconsistent time")
                return
            if is_duration & (not df.empty):
                if name != "hospitalization_time":
                    df = df[(df["c_start_ts"] >= df["c_last_ts"])]
                df = self.compare_end_ts(
                    df=df, end_col="c_an_start_ts"
                )  # TODO check with new defined timelines
            return df

        if tl == 2:
            #  data until the end of the aneasthesia from the begin of aneasthesia
            df = df[(df["c_start_ts"] < df["c_an_end_ts"])]
            # TODO think of all other durations
            if not is_duration:
                df = df[(df["c_start_ts"] >= df["c_an_start_ts"])]
            if name.replace(" ", "") in ["recovery_room_time"]:
                print("inconsistent time")
                return
            if is_duration & (not df.empty):
                if name != "hospitalization_time":
                    df = df[(df["c_start_ts"] >= df["c_last_ts"])]
                df = self.compare_end_ts(df=df, end_col="c_an_end_ts")
            return df

        if tl == 3:
            #  data before occurence of postoperative delirium and after the aneasthesia end
            df = df[(df["c_start_ts"] < df["c_timestamp"])]
            if not is_duration:
                df = df[(df["c_start_ts"] >= df["c_an_end_ts"])]
            if is_duration & (not df.empty):
                if name != "hospitalization_time":
                    df = df[(df["c_start_ts"] >= df["c_last_ts"])]
                df = self.compare_end_ts(df=df, end_col="c_timestamp")
            return df
