import pandas as pd
from pandas.core.frame import DataFrame


class DataStandardizer:
    def __init__(self) -> None:
        pass

    def standardize_data(
        self, df: DataFrame = None, metr_dict: dict = None, num_cols: list = []
    ):
            
        '''
        performing z transformation on data train and test indepentently migitating bias 
                Parameters: 
                        df (df): data, metr_dict (dict): standardization metrics, num_cols (list): list of numeric columns
                Returns:
                        df (df): standardized data, d (dict): dict with standatdization metrics
        '''
        # convert categorical variables to {1, -1}
        df_cat = df.drop(columns=num_cols)
        df_cat = (df_cat - 0.5) * 2
        df_num = df[num_cols]
        # create dict for z-transformation matric
        if metr_dict is None:
            df_mean = df_num.mean()
            df_std = df_num.std()
        else:
            df_mean = metr_dict["mean"]
            df_std = metr_dict["std"]
        # stadardize numerical attributes
        df_num = (df_num - df_mean) / df_std
        df_num = df_num.fillna(0)
        # concat both
        df = pd.concat([df_num.reset_index(), df_cat.reset_index()], axis=1).drop(
            columns=["index"]
        )
        return df, {"mean": df_mean, "std": df_std}
