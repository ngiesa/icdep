class Annotator:
    def __init__(self) -> None:
        self.nu_desc_threshold = 1

    def annotate_binary(self, df):

        # using the set nu threshold for binary annotation
        df = df.assign(
            c_target=[1 if x >= self.nu_desc_threshold else 0 for x in df["c_value"]]
        )

        # splitting in pos and neg groups that are distinct in terms of hospital stays
        df_pos = df[df.c_target == 1]
        df_neg = df[(df.c_target == 0) & (df.c_value != 1)]
        df_neg = df_neg[~df_neg.c_pat_id.isin(df_pos.c_pat_id)]

        # print group sizes and return joined cohort
        print(
            "# positives NuDesc threshold at", self.nu_desc_threshold, ":", len(df_pos)
        )
        print(
            "# negatives NuDesc threshold at", self.nu_desc_threshold, ":", len(df_neg)
        )

        return df_neg.append(df_pos).rename(columns={"c_value": "c_nudesc"})
