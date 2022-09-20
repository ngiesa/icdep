from ray import tune

model_variants = {
    "model_1": {
        "tl": [1],
        "drop_50": [False],
        "oversample": [False],
        "loss_type": ["bce", "focal"]
    },
    "model_2": {
        "tl": [2],
        "drop_50": [False],
        "oversample": [False],
        "loss_type": ["bce", "focal"]
    },
    "model_3": {
        "tl": [3],
        "drop_50": [False],
        "oversample": [False],
        "loss_type": ["bce", "focal"]
    },
    "model_4": {
        "tl": [1, 2],
        "drop_50": [False],
        "oversample": [False],
        "loss_type": ["bce", "focal"]
    },
    "model_5": {
        "tl": [2, 3],
        "drop_50": [False],
        "oversample": [False],
        "loss_type": ["bce", "focal"]
    },
    "model_6": {
        "tl": [1, 2, 3],
        "drop_50": [False],
        "oversample": [False],
        "loss_type": ["bce", "focal"]
    },
}

no_feats = [
    "c_pat_id",
    "c_case_id",
    "c_an_start_ts",
    "c_op_id",
    "c_target",
    "c_time_consistent",
]

# configs for hyperparameter search
config_mlp = {
    "batch_size": tune.choice([64, 128, 256]),
    "learning_rate": tune.choice([1e-2, 1e-3, 1e-4]),
    "activation": tune.choice(["relu", "sig"]),
    "node": tune.grid_search([8, 16, 32, 64, 128]),
    "layer": tune.grid_search([4, 8, 12]),
}

config_trees = {
    "tree_type": tune.grid_search(["random_forest", "boosted_trees"]),
    "n_estimators": tune.grid_search([1, 10, 100, 1000]),
    "max_depth": tune.choice([2, 4, 8]),
    "min_sample_split": tune.choice([2, 4, 8, 16]),
    "max_leaf_nodes": tune.choice([2, 4, 16, None])
}

# define columns for cv hyperparameter evaluation
grouping_columns = ["miss_frac", "num_thres", "cat_thres", "num_feats", "cat_feats", "X", "X_len",
                    "cv_hyperparam", "model", "ml_type", "loss_type", "config_dict"]
validation_columns = ["loss-val", "loss-train", "auc-roc-train",
                      "auc-roc-val", "auc-pr-train", "auc-pr-val",
                      "sens-train", "sens-val", "spec-train",
                      "spec-val", "prec-train", "prec-val"]

# helper dict for evaluation process
model_timelines = {
    0: {"from": "hospitalization start", "to": "anesthesia start", "m": "M1"},
    1: {"from": "anesthesia start", "to": "anesthesia end", "m": "M2"},
    2: {"from": "anesthesia end", "to": "Nu-DESC evaluation", "m": "M3"},
    3: {"from": "hospitalization start", "to": "anesthesia end", "m": "M12", },
    4: {"from": "anesthesia start", "to": "Nu-DESC evaluation", "m": "M23", },
    5: {
        "from": "hospitalization start",
                "to": "Nu-DESC evaluation",
                "m": "M123",
    },
}

models_timestamps = {
    1: {"from": "c_hos_start_ts", "to": "c_an_start_ts"},
    2: {"from": "c_an_start_ts", "to": "c_an_end_ts"},
    3: {"from": "c_an_end_ts", "to": "c_timestamp"},
    4: {"from": "c_hos_start_ts", "to": "c_an_end_ts"},
    5: {"from": "c_an_start_ts", "to": "c_timestamp"},
    6: {"from": "c_hos_start_ts", "to": "c_timestamp"},
}
