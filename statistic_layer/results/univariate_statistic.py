from extraction_layer.support_classes.js_converter import JSConverter
from pandas.core.frame import DataFrame
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from math import e, sqrt
from numpy import log as ln
from configs import no_feats

class UnivariateStatistic:
    def __init__(self) -> None:
        pass

    def read_feature_sets(self, i=3):

        '''
        executes effect statistics and stores results
                Parameters: 
                        None
                Returns:
                        None
        '''      

        # declare result dfs
        univ_numeric = pd.DataFrame()
        univ_categoric = pd.DataFrame()
        for tl in range(1, 4):
            df_num_res = pd.DataFrame()
            df_tl_num = pd.read_csv(
                "./data/interpreted/tl{}/i{}/df_train_numeric_revised.csv".format(
                    str(tl), str(i)
                ),
                index_col=0,
            )
            # univariate testing for numerical variables
            for col in list(df_tl_num.columns):
                if col in no_feats:
                    continue
                U1, p, AUC, av, n1, n2, = self.get_mwu_from_df(
                    df=df_tl_num, col=col)
                mi_score = self.get_mi_from_df(df=df_tl_num, col=col)
                df_num_res = df_num_res.append(
                    {
                        "feature": col,
                        "tl": tl,
                        "i": i,
                        "U1": U1,
                        "p": p,
                        "AUC": AUC,
                        "2|AUC-0.5|": 2 * abs(AUC - 0.5),
                        "avai": av,
                        "miss": 1 - av,
                        "n_pos": n1,
                        "n_neg": n2,
                        "MI": mi_score,
                        "10MI": 10 * mi_score,
                    },
                    ignore_index=True,
                )
            reject, pvals_corrected = fdrcorrection(np.array(df_num_res["p"]))
            df_num_res["p_corr"] = pvals_corrected
            df_num_res["reject"] = reject
            univ_numeric = univ_numeric.append(df_num_res)
            # univariate odds ratio testing for categorical variables
            df_tl_cat = pd.read_csv(
                "./data/interpreted/tl{}/i{}/df_train_categoric_revised.csv".format(
                    str(tl), str(i)
                ),
                index_col=0,
            )
            for col in list(df_tl_cat.columns):
                if col in no_feats:
                    continue
                df = df_tl_cat[["c_target"] + [col]]
                df = df.assign(c_value=df[col])
                # calculate the odds ratio
                l00 = len(df[(df.c_value == 0) & (df.c_target == 0)])
                l11 = len(df[(df.c_value == 1) & (df.c_target == 1)])
                l10 = len(df[(df.c_value == 1) & (df.c_target == 0)])
                l01 = len(df[(df.c_value == 0) & (df.c_target == 1)])
                if (l00 < 2) | (l11 < 2) | (l10 < 2) | (l01 < 2):
                    odds_ratio = 1
                    lower_ci = 0
                    upper_ci = 0
                else:
                    odds_ratio = (l00 * l11) / (l10 * l01)
                    lower_ci = pow(
                        e,
                        ln(odds_ratio)
                        - 1.96 * sqrt(1 / l00 + 1 / l11 + 1 / l10 + 1 / l01),
                    )
                    upper_ci = pow(
                        e,
                        ln(odds_ratio)
                        + 1.96 * sqrt(1 / l00 + 1 / l11 + 1 / l10 + 1 / l01),
                    )
                univ_categoric = univ_categoric.append(
                    {
                        "feature": col,
                        "tl": tl,
                        "i": i,
                        "OR": odds_ratio,
                        "|log(OR)|": abs(np.log(odds_ratio)),
                        "lower_ci": lower_ci,
                        "upper_ci": upper_ci,
                        "n_pos": len(df_tl_cat[df_tl_cat.c_target == 1]),
                        "n_neg": len(df_tl_cat[df_tl_cat.c_target == 0]),
                    },
                    ignore_index=True,
                )
        # store results
        univ_numeric.to_csv(
            "./data/metrics/statistical/univariate_stats_train_numeric_revised.csv"
        )
        univ_categoric.to_csv(
            "./data/metrics/statistical/univariate_stats_train_categoric_revised.csv"
        )

    def get_mwu_from_df(self, df: DataFrame = None, col: str = ""):
        '''
        analyze mutual information 
                Parameters: 
                        df (df): data, col (str): feature column
                Returns:
                        U1 (float): U statistics, p (float): p-value, fraction (float), 
                        n1_len (int): length 1 class, n2_len (int): length 2 class, 
        '''
        total_len = len(df)
        n1 = np.array(df[df.c_target == 1][col].dropna())
        n2 = np.array(df[df.c_target == 0][col].dropna())
        U1, p = mannwhitneyu(n1, n2, alternative="two-sided")
        AUC = U1 / (len(n1) * len(n2))
        return U1, p, AUC, (len(n2) + len(n1)) / total_len, len(n1), len(n2)

    def get_mi_from_df(self, df: DataFrame = None, col: str = "", k=5):
        '''
        analyze mutual information 
                Parameters: 
                        df (df): data, col (str): feature column
                Returns:
                        mutal information score (float)
        '''
        df = df[[col] + ["c_target"]].dropna()
        X = np.array(df[col])
        y = np.array(df.c_target)
        return mutual_info_classif(X.reshape(-1, 1), y, n_neighbors=k)[0]
