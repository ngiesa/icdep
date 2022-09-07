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
        '''
        asks users to provide credentials as inputs
                Parameters: 
                        js (dict): json for variable, abbr: shortname for column (str)
                Returns:
                        time data (df): A dataframe with time related columns 
        '''
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

    def create_master_table(self):
        '''
        stores and returns metadata about admission, discharge surgery times and the NuDesc as target variable
        every row describes one surgery with corresponding time information used for T1, T2, T3
                Parameters: 
                        None
                Returns:
                        master table (df): A dataframe with 
        '''

        # instanze of class to parse and chek on json variables stored
        jsc = JSConverter()

        # loading time information in df structure
        df_rec = self.create_time_df(
            js=jsc.read_js_file(
                path="./data/raw/hospitalization/recovery_room_time"),
            abbr="rec",
        )
        df_hos = self.create_time_df(
            js=jsc.read_js_file(
                path="./data/raw/hospitalization/hospitalization_time"),
            abbr="hos",
        )
        df_op = self.create_time_df(
            js=jsc.read_js_file(
                path="./data/raw/hospitalization/surgery_time"),
            abbr="op",
        )
        df_an = self.create_time_df(
            js=jsc.read_js_file(
                path="./data/raw/hospitalization/aneasthesia_time"),
            abbr="an",
        )

        # flag for including cam after recovery room as well
        option_cam = False

        # load target information
        target_nu = pd.DataFrame(
            jsc.read_js_file(path="./data/raw/scores_and_scales/nudesc")
        ).rename(columns={"c_start_ts": "c_timestamp"})
        target_cam = pd.DataFrame(
            jsc.read_js_file(path="./data/raw/scores_and_scales/camicu")
        ).rename(columns={"c_start_ts": "c_timestamp"})

        if option_cam:
            target = pd.concat([target_cam, target_nu])
        else:
            target = target_nu

        target = target.assign(c_timestamp=pd.to_datetime(target.c_timestamp))

        # join op related times
        df_rel = df_rec.merge(df_op, on=["c_case_id", "c_link_id", "c_pat_id"]).merge(
            df_an, on=["c_case_id", "c_link_id", "c_pat_id"]
        )

        # join op times and target
        df = df_rel.merge(
            df_hos.drop(columns=["c_pat_id"], axis=1).drop(
                columns="c_link_id", axis=1),
            on=["c_case_id"],
            how="inner",
        ).merge(target.drop(columns=["c_pat_id"], axis=1), on=["c_case_id"], how="left")

        # add binary annotation variable based on theshold
        df = self.ann.annotate_binary(df)

        # do logical imputation for important time stamps
        df = self.impute_time_stamps(df)

        # store prev op counts
        df = self.get_surgery_info(df)

        # assign last ts for calculating time frames
        print("assigning last ts to time relevant data")
        df["c_last_ts"] = df.apply(
            lambda x: self.assign_last_ts(x, df), axis=1)

        # focus on delirium in recovery room only
        df = df[
            (df.c_timestamp >= df.c_rec_start_ts) & (
                df.c_timestamp <= df.c_rec_end_ts)
        ]

        # renaming the link id in op id
        df = df.rename(columns={"c_link_id": "c_op_id"})

        # group by and set c_timestamp for TL3 to first assessment point
        df_grouped = df.groupby(["c_op_id"])["c_target"].max().reset_index() \
            .merge(df.groupby("c_op_id")["c_timestamp"].min(), on="c_op_id").reset_index()

        df = df.drop(columns=["c_target", "c_timestamp"]
                     ).merge(df_grouped, on="c_op_id")
        df = df.drop_duplicates()

        # store master table as readable csv
        df.to_csv("./data/meta/cohort/master_time_table_22_08_22.csv")

        return df

    def replace_time_order(
        self,
        df: DataFrame = None,
        col_before: str = "",
        col_after: str = "",
        back_replacement: bool = True,
    ):
        '''
        replace if one time column can be substituted with the other one when inconsistent
                Parameters: 
                        df (df): the master table, col_before (str): column name pre, col_after (str): column name post,
                        back_replacement (bool): indicator of replacement should be performed in a backward sense
                Returns:
                        df (df): master table with switched time related columns if required
        '''
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
        '''
        performs basic imputation for inconsistent or missing time stamps
                Parameters: 
                        df (df): dataframe with time related columns per surgery
                Returns:
                        master table (df): dataframe with imputed time columns per surgery
        '''
        # impute if op end is null with aneast end
        df.c_op_end_ts.fillna(df.c_an_end_ts, inplace=True)

        # impute if op start is null with aneast start
        df.c_op_start_ts.fillna(df.c_an_start_ts, inplace=True)

        # impute aneast start is null with op start
        df.c_an_start_ts.fillna(df.c_op_start_ts, inplace=True)

        # impute aneast end is null with op end
        df.c_an_end_ts.fillna(df.c_op_end_ts, inplace=True)

        # impute rec end with hos end if null
        df.c_rec_end_ts.fillna(df.c_hos_end_ts, inplace=True)

        # if hospital start ts is null set to 12h before an start
        df.c_hos_start_ts.fillna(pd.to_datetime(
            df.c_an_start_ts) - pd.Timedelta(hours=12), inplace=True)

        # do basic admission and discharge replacement, not important for T1, T2, T3
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
            df[dur_col] = pd.to_datetime(
                df[col_end]) - pd.to_datetime(df[col_start])
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
        '''
        for multiple surgeries per stay, get last timestamp either after the last surgery or admission
                Parameters: 
                        row (Series): the row related to one surgery, df (df): the master table
                Returns:
                        the last timestamp which is used for setting up T1, T2, T3
        '''
        # treat last rec room as new stay within one hospitalization
        if row["c_op_count"] == 0:
            # return hos when op count is one per case
            return row["c_hos_start_ts"]
        else:
            # return last rec room end when op count > 0
            df = df[(df.c_case_id == row["c_case_id"])]
            df = df[(df.c_op_count == row["c_op_count"] - 1)]
            return df["c_rec_end_ts"].iloc[0]

    def get_surgery_info(self, df):
        '''
        get additional features like number of suergeries per stay or per patient
                Parameters: 
                        df (df): the master table
                Returns:
                        df (df): master table with surgery information
        '''
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
        df_surg = df_surg.assign(
            c_prev_op_count=df_surg.groupby("c_pat_id").cumcount())

        # get meta information describes in own method
        meta_vars = MetaManager.get_meta_data()
        meta_prev_ops = meta_vars[meta_vars["var_name"].isin(
            list(op_count.keys()))]

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

            # store results as json variables as well
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
        '''
        calcualte durations e.g. intubation time per time window
                Parameters: 
                        df (df): data, end_ts_col (str): name of end column
                Returns:
                        data (df): data with duration lengths in hours
        '''
        df = df.assign(c_start_ts=pd.to_datetime(df.c_start_ts))
        df[end_ts_col] = pd.to_datetime(df[end_ts_col])
        df = df.assign(c_value=(df[end_ts_col] - df.c_start_ts))
        df = df.assign(c_value=[x.total_seconds() /
                       3600.00 for x in df.c_value])
        return df

    def compare_end_ts(self, df: DataFrame = None, end_col: str = ""):
        '''
        for duration features e.g. intubation time, the end timestamp is important
                Parameters: 
                        df (df): master table, end_col (str): name of time columns of duration end 
                Returns:
                        data (df): data with duration end information
        '''
        # if the end ts of the feature is before end ts of time col, then use end ts of feature
        df_1 = df[df.c_end_ts < df[end_col]]
        df_2 = df[df.c_end_ts >= df[end_col]]
        if not df_1.empty:
            df_1 = self.calc_duration_for_tl(df=df_1, end_ts_col="c_end_ts")
        if not df_2.empty:
            df_2 = self.calc_duration_for_tl(df=df_2, end_ts_col=end_col)
        return df_1.append(df_2)

    def get_time_lines(
        self, df: DataFrame = None, tl: int = 1, is_duration=False, feature_name: str = ""
    ):
        '''
        lookup in defined time windows T1, T2, T3 for feature availability
                Parameters: 
                        df (df): master table, tl (int): time window index, 
                        is_duration (bool): indicator of feature descibes a time period e.g. intubation time,
                        feature_name (str): name of feature 
                Returns:
                        data (df): feature data from corresponding time window
        '''

        df = df.drop_duplicates()

        required_columns = [
            "c_an_start_ts",
            "c_start_ts",
            "c_rec_start_ts",
            "c_an_end_ts",
            "c_timestamp",
            "c_last_ts"
        ]

        if (len(df[(df["c_start_ts"] == "")]) > 0.1 * len(df)) or (
            len(df[df.c_start_ts.isnull()]) > 0.1 * len(df)
        ):
            print("feature does have empty time stamps over 10%")
            return

        if not set(required_columns).issubset(set(list(df.columns))):
            print("df does not contain required time columns")  # TODO
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
            if feature_name.replace(" ", "") in ["surgery_time", "anesthesia_time"]:
                print("inconsistent time")
                return
            if is_duration & (not df.empty):
                if feature_name != "hospitalization_time":
                    df = df[(df["c_start_ts"] >= df["c_last_ts"])]
                df = self.compare_end_ts(
                    df=df, end_col="c_an_start_ts"
                )
            return df

        if tl == 2:
            #  data until the end of the aneasthesia from the begin of aneasthesia
            df = df[(df["c_start_ts"] < df["c_an_end_ts"])]
            if not is_duration:
                df = df[(df["c_start_ts"] >= df["c_an_start_ts"])]
            if feature_name.replace(" ", "") in ["recovery_room_time"]:
                print("inconsistent time")
                return
            if is_duration & (not df.empty):
                if feature_name != "hospitalization_time":
                    df = df[(df["c_start_ts"] >= df["c_last_ts"])]
                df = self.compare_end_ts(df=df, end_col="c_an_end_ts")
            return df

        if tl == 3:
            #  data before occurence of postoperative delirium and after the aneasthesia end and first NuDesc
            df = df[(df["c_start_ts"] < df["c_timestamp"])]
            if not is_duration:
                df = df[(df["c_start_ts"] >= df["c_an_end_ts"])]
            if is_duration & (not df.empty):
                if feature_name != "hospitalization_time":
                    df = df[(df["c_start_ts"] >= df["c_last_ts"])]
                df = self.compare_end_ts(df=df, end_col="c_timestamp")
            return df
