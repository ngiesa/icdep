
from extraction_layer.support_classes.js_converter import JSConverter


class Validator:

    def __init__(self) -> None:
        pass

    def validate_prepared_feature(self, train_path=""):

        # opening split information
        split = JSConverter()
        _ = split.read_js_file("./data/meta/cohort/train_test_split")

        # split training pat and testing pats
        train_pat_pos = split.data_obj['train_pos'] + \
            split.data_obj['train_neg']
        train_pat = list(dict.fromkeys(train_pat_pos))

        # opening file from path for training data
        jsc = JSConverter()
        _ = jsc.read_js_file(train_path)

        valid = set(list(jsc.to_df().c_pat_id.drop_duplicates())
                    ).issubset(set(train_pat))
        print("{} valid: {}".format(train_path, str(valid)))

        return valid
