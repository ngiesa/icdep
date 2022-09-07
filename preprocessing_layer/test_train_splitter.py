from preprocessing_layer.time_manager import TimeManager
from extraction_layer.support_classes.js_converter import JSConverter
import pandas as pd

class TrainTestSplitter:
    def __init__(self) -> None:
        self.tm = TimeManager()
        self.split_ratio = 0.2
        self.master_df = pd.read_csv("./data/meta/cohort/master_time_table_grouped.csv", index_col=0)

    def split_train_test_patients(self):
        '''
        splitting data into test and train sets and stores result
                Parameters: 
                        None
                Returns:
                        None
        '''

        mdf = self.master_df

        # split in pos and neg
        pos = mdf[mdf.c_target == 1]["c_pat_id"].drop_duplicates()
        neg = mdf[~mdf.c_pat_id.isin(list(pos))]["c_pat_id"].drop_duplicates()

        test_pos = pos.sample(frac=self.split_ratio)
        train_pos = pos[~pos.isin(list(test_pos))]
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

 